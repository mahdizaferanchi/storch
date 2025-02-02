from pathlib import Path
from torch.optim import Adam as OptClass
from storch import configurable, use_config
from torch.nn import CrossEntropyLoss as LossFnClass
from datetime import timedelta
from storch.training import trainer_constructor as TrainerClass
import argparse
import sys
inf = sys.maxsize

parser = argparse.ArgumentParser(description="Configuration for training")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
parser.add_argument("--id", type=str, default="", help="ID for the experiment")
parser.add_argument("-f", type=str, default="")


args = parser.parse_args()

OptClass = configurable(OptClass)
LossFnClass = configurable(LossFnClass)
_id=args.id
wandb_data_dir = Path("./data").mkdir(exist_ok=True)
learning_rate = args.learning_rate

local_run = _id.startswith("local")

config = dict(
    project="dislogic",
    _id=_id,
    epochs=1,
    local_run=local_run,
    model_initial="scratch",
    checkpoints_dir=".",
    decay_lr=True,
    init=dict(
        id=_id,
        dir=wandb_data_dir,
    ),
    GPT2Config=dict(
        use_cache=False,
        return_dict=False,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,),
    optimizer_constructor=OptClass,
    loss_fn_constructor=LossFnClass,
    memmaped_tokens_data=dict(
        path="../input/wikitext-pretokenized",
        block_size=1024,
        batch_size=1,
    ),
    EnumData=dict(
        steps=5,
        ignore_depletion=False,
        reset_on_start=True,
        train_eval=dict(steps=10), # namespace
        val=dict(steps=15), # namespace
    ),
    get_logger=dict(
        time_interval=timedelta(minutes=10),
        progress_ratio_interval=0.1,
        progress_steps_interval=inf,
        ntfy_path=None,
    ),
    cosine_schedule_lr_getter=dict(
        learning_rate=learning_rate,
        min_lr=1e-4,
        warmup_iters=200,
        lr_decay_iters=60_000,
    ),
    configure_optimizers=dict(
        weight_decay=1e-1,
    ),
)
dynamic_config = {
    OptClass.__name__: dict(lr=learning_rate),
    TrainerClass.__name__: dict(
        grad_clip=1.,
        gradient_accumulation_steps=10,
    ),
}
config = config | dynamic_config
use_config(config)
from types import SimpleNamespace
config = SimpleNamespace(**config)

import torch.utils.data as tdu
import inspect
from torch.utils.data import Dataset, IterableDataset
from torch.distributed import destroy_process_group
from pathlib import Path
import numpy as np
import random
import torch
import math
from torch import nn
from storch import configurable

class FirstOut(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]

class MemMapTokensDataset(Dataset):
    def __init__(self, source, split, block_size=1024):
        self.source = Path(source)
        self.split = Path(f"{split}.bin")
        self.path = self.source / self.split
        if not self.path.exists():
            raise FileNotFoundError("Couldn't find data file")
        self.block_size = block_size

    def __len__(self):
        data = np.memmap(self.path, dtype=np.uint16, mode='r')
        return (len(data) - self.block_size)

    def __getitem__(self, i):
        data = np.memmap(self.path, dtype=np.uint16, mode='r')
        tokens = torch.from_numpy(
            (data[i:i+1+self.block_size]).astype(np.int64))
        return tokens


class MemMapTokensDatasetRA(IterableDataset):
    def __init__(self, dset):
        self.dset = dset
        self.length = len(dset)

    def __iter__(self):
        return self

    def __next__(self):
        idx = random.randint(0, self.length - 1)
        return self.dset[idx]


@configurable
def memmap_collate(batch, pin=False):
    batch = torch.stack(batch)
    if pin:
        batch = batch.pin_memory()
    x = batch[:, :-1]
    y = batch[:, 1:]  # .clone()
    # y[:, 0:512] = -100
    # y = y.to(x.device)
    return x, y

@configurable
def memmaped_tokens_data(path, block_size, batch_size):
    _train_set = MemMapTokensDataset(path, "train", block_size=block_size)
    train_set = MemMapTokensDatasetRA(_train_set)

    _test_set = MemMapTokensDataset(path, "val", block_size=block_size)
    test_set = MemMapTokensDatasetRA(_test_set)

    train_loader = tdu.DataLoader(
        train_set, batch_size=batch_size, collate_fn=memmap_collate)
    test_loader = tdu.DataLoader(
        test_set, batch_size=batch_size, collate_fn=memmap_collate)
    return train_loader, test_loader

@configurable
def cosine_schedule_lr_getter(it, learning_rate, min_lr, warmup_iters, lr_decay_iters):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

@configurable
def configure_optimizers(model, optimizer_constructor, weight_decay, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = optimizer_constructor(optim_groups, **extra_args)

    return optimizer

from transformers import GPT2Config, GPT2LMHeadModel
import torchmetrics as tm
from storch.training import trainer_constructor as Trainer
from storch import (get_logger, EnumData, epochs_loop,
                    get_minutes_left, configurable,
                    master_process, wandb_load_state,
                    wandb_log_state, log_message, log_mets,
                    device)
import wandb
from storch.exec_info import ddp, master_process

local_run = config.local_run

if not local_run:
    wandb_init = configurable(wandb.init)
    if master_process:
        wandb_init(allow_val_change=True, resume="allow")
        wandb.run.log_code(".")

    state = wandb_load_state()
state = None

GPT2Config = configurable(GPT2Config)
hfconfig = GPT2Config()
initial = config.model_initial
if initial == "scratch":
    model = GPT2LMHeadModel(config=hfconfig)
else:
    model = GPT2LMHeadModel.from_pretrained(initial, config=hfconfig)
model = FirstOut(model).to(device)

device_type = "cuda" if "cuda" in device else "cpu"
optimizer = configure_optimizers(model, config.optimizer_constructor, device_type=device_type)

_loss_fn = config.loss_fn_constructor()
def loss_fn(preds, target):
    return _loss_fn(preds.view(-1, preds.size(-1)), target.reshape(-1))

metrics = tm.MetricCollection({
    "perp": tm.text.Perplexity(ignore_index=-100),
    "loss": tm.MeanMetric(),
})

trainer = Trainer(model, loss_fn, optimizer, metrics,
                  get_lr=cosine_schedule_lr_getter)

_train_loader, _val_loader = memmaped_tokens_data()

get_logger = configurable(get_logger)

train_loader = EnumData(_train_loader, logger=get_logger(), _c="train")
train_eval_loader = EnumData(_val_loader, logger=get_logger(), _c="train_eval")
val_loader = EnumData(_val_loader, logger=get_logger(), _c="val")

if local_run:
    log_message = print
    log_mets = print
    wandb_log_state = lambda *args, **kw: None

initial_epoch = trainer.epoch 
for epoch in epochs_loop(trainer.epoch, config.epochs, get_minutes_left() - 10):
    try:
        log_message(f"\n== epoch {trainer.epoch} ==")
        trainer.train(train_loader)
        trainer.test(train_eval_loader)
        log_mets(trainer.metrics.compute(), epoch, "train")
        trainer.test(val_loader)
        log_mets(trainer.metrics.compute(), epoch, "val", True)
        trainer.epoch += 1
    except Exception as e:
        if epoch == initial_epoch:
            raise e
        print(f'encountered: {e}')
        print('trying to exit script and save checkpoint...')
        break
        
if local_run:
    from pathlib import Path
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"{config._id}_state"
    if master_process:
        state_file = checkpoints_dir / f"state_{trainer.epoch - 1}_{config._id}"
        torch.save(trainer.state_dict(), state_file)
else:
    wandb_log_state(trainer)
    if master_process:
        wandb.finish()
if ddp:
    destroy_process_group()