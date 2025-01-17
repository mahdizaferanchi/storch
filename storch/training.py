from collections import defaultdict
from contextlib import nullcontext
from storch import configurable

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from .exec_info import ddp, ddp_local_rank, device


class SimpleTrainer:
    def __init__(self, model, loss_fn, optimizer, metrics, state=None) -> None:
        super().__init__()
        self.model = model
        self.model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.metrics.to(device)
        self.callbacks = defaultdict(list)
        self.epoch = 1
        if state:
            self.load_state_dict(state)

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def train_step(self, x, y):
        preds = self.model(x)
        self.loss = self.loss_fn(preds, y)

        self.optimizer.zero_grad(set_to_none=True)
        self.loss.backward()
        self.optimizer.step()

        self.trigger_callbacks("train_step_end")

    @torch.no_grad()
    def test_step(self, x, y):
        preds = self.model(x)
        self.loss = self.loss_fn(preds, y)

        self.metrics.update(preds=preds.detach(),
                            target=y.detach(), value=self.loss)
        self.trigger_callbacks("test_step_end")

    def train(self, dataloader):
        self.model.train()
        for x, y in dataloader:
            self.train_step(x, y)
        self.trigger_callbacks("train_end")

    def test(self, dataloader, reset=True):
        self.model.eval()
        if reset:
            self.metrics.reset()
        for x, y in dataloader:
            self.test_step(x, y)
        self.trigger_callbacks("test_end")

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch = state["epoch"]


class Trainer(SimpleTrainer):
    def __init__(
        self, model, loss_fn, optimizer, metrics, state=None, dtype="float16", 
        grad_clip=0., gradient_accumulation_steps=1, get_lr=None
    ) -> None:
        super().__init__(model, loss_fn, optimizer, metrics)
        self.raw_model = model
        if state:
            self.load_state_dict(state)
        if ddp:
            self.model = DDP(self.raw_model, device_ids=[ddp_local_rank])
        device_type = "cuda" if "cuda" in device else "cpu"
        enable_scaler = (device_type == 'cuda') and (dtype == "float16")
        self.scaler = torch.cuda.amp.GradScaler(enabled=enable_scaler)
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        self.ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

        self.grad_clip = grad_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.get_lr = get_lr
        self.iter_num = 0

    def train_step(self, x, y):
        if self.get_lr:
            lr = self.get_lr(self.iter_num)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        for micro_step in range(self.gradient_accumulation_steps):
            if ddp:
                # explanation here: https://github.com/karpathy/nanoGPT/blob/master/train.py
                self.model.require_backward_grad_sync = (
                    micro_step == self.gradient_accumulation_steps - 1)
            with self.ctx:
                preds = self.model(x)
                self.loss = self.loss_fn(preds, y)
            self.scaler.scale(self.loss).backward()

        if self.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.iter_num += 1
        self.trigger_callbacks("train_step_end")

    @torch.no_grad()
    def test_step(self, x, y):
        with self.ctx:
            preds = self.model(x).float()
            self.loss = self.loss_fn(preds, y)

        self.metrics.update(preds=preds.detach(),
                            target=y.detach(), value=self.loss)
        self.trigger_callbacks("test_step_end")

    def state_dict(self):
        return {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }

    def load_state_dict(self, state):
        self.raw_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch = state["epoch"]


class Distiller(Trainer):
    def __init__(self, model, eval_model, loss_fn, optimizer, metrics, loss_metrics, state=None, dtype="float16",
                 grad_clip=0., gradient_accumulation_steps=1, get_lr=None) -> None:
        self.eval_model = eval_model.to(device)
        super().__init__(model, loss_fn, optimizer, metrics, state, dtype,
                         grad_clip, gradient_accumulation_steps, get_lr)
        self.loss_metrics = loss_metrics
        self.loss_metrics.to(device)

    @torch.no_grad()
    def evaluate_loss_step(self, x, y):
        with self.ctx:
            preds = self.model(x)
            self.loss = self.loss_fn(preds, y)

        self.loss_metrics.update(value=self.loss)

    @torch.no_grad()
    def test_step(self, x, y):
        with self.ctx:
            preds = self.eval_model(x).float()

        self.metrics.update(preds=preds.detach(), target=y.detach())
        self.trigger_callbacks("test_step_end")

    def evaluate_loss(self, dataloader):
        self.model.eval()
        for x, y in dataloader:
            self.evaluate_loss_step(x, y)

    def state_dict(self):
        return {
            "model": self.eval_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "iter_num": self.iter_num,
        }

    def load_state_dict(self, state):
        self.eval_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch = state["epoch"]
        self.iter_num = state["iter_num"]

trainer_constructor = configurable(Trainer)
distiller_constructor = configurable(Distiller)