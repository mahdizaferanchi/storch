import os
import dbm
import requests
import torch
from datetime import datetime, timedelta
from pathlib import Path
import wandb
import time
from .exec_info import ddp, master_process, device
import sys
from typing import Any
from torch.distributed import barrier

inf = sys.maxsize


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_minutes_left():
    try:
        limit_dt = datetime.fromisoformat(os.environ["EXPIRE_TIME"])
        left = limit_dt - datetime.now()
        return left.seconds / 60
    except KeyError:
        print("Could not get EXPIRE_TIME env variable. Assuming 12 hours left.")
        return 12 * 60


def log_mets(info, step, split, commit=False):
    if master_process:
        info_to_log = {f"{split}_{key}": info[key] for key in info.keys()}
        wandb.log(info_to_log, step, commit)


def epochs_loop(first_epoch, last_epoch, limit):
    start = time.time()
    for i in range(first_epoch, last_epoch + 1):
        yield i
        temp = i - first_epoch + 1
        growth_ratio = (temp + 1) / temp
        minutes_so_far = (time.time() - start) / 60
        if growth_ratio * minutes_so_far > limit:
            break


def move_to(obj, device):
    """Accepts `torch.tensor` or a python object containg nested dicts, lists and tuples
    that eventually contain tensors at the bottom level. Moves the tensors to device"""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, tuple):
        return tuple([move_to(item, device) for item in obj])
    else:
        raise TypeError("Invalid type for move_to")

def log_message(message):
    if master_process:
        now = datetime.now()
        print(f"[{now.isoformat()}]", end=" ")
        print(message)


class PrintLogger:
    def __init__(
        self,
        time_interval=timedelta(days=100),
        progress_ratio_interval=inf,
        progress_steps_interval=inf,
        ntfy_path=None,
    ) -> None:
        self.last_log_time = datetime.now()
        self.last_progress_ratio = 0
        self.last_progress_steps = 0
        self.progress_ratio_interval = progress_ratio_interval
        self.progress_steps_interval = progress_steps_interval
        self.time_interval = time_interval
        self.ntfy_path = ntfy_path

    def set_enumerator_len(self, len):
        self.enumerator_len = len

    def __call__(self, enumerator) -> Any:
        now = datetime.now()
        if enumerator.step == 1:
            self.out("\nnew iteration start")
            self.log(enumerator)
        if now - self.last_log_time > self.time_interval:
            self.log(enumerator)
        elif hasattr(self, "enumerator_len"):
            progress_ratio = enumerator.step / self.enumerator_len
            if progress_ratio - self.last_progress_ratio > self.progress_ratio_interval:
                self.log(enumerator)
        elif enumerator.step - self.last_progress_steps > self.progress_steps_interval:
            self.log(enumerator)

    def log(self, enumerator):
        now = datetime.now()
        if hasattr(self, "enumerator_len"):
            self.out(
                f"step {enumerator.step} of {self.enumerator_len} ({100 * enumerator.step / self.enumerator_len: .2f}%)"
            )
        else:
            self.out(f"step {enumerator.step}")
        self.last_log_time = now
        if hasattr(self, "enumerator_len"):
            progress_ratio = enumerator.step / self.enumerator_len
            self.last_progress_ratio = progress_ratio
        self.last_progress_steps = enumerator.step

    def out(self, message):
        log_message(message)
        if self.ntfy_path:
            requests.post(f"https://ntfy.sh/{self.ntfy_path}", data=message)


class DummyLogger:
    def __init__(
        self,
        time_interval=timedelta(days=100),
        progress_ratio_interval=inf,
        progress_steps_interval=inf,
        ntfy_path=None,
    ) -> None:
        pass

    def set_enumerator_len(self, len):
        pass

    def __call__(self, enumerator) -> Any:
        pass

    def log(self, enumerator):
        pass

    def out(self, message):
        pass


def get_logger(project, _id, time_interval,
                progress_ratio_interval, progress_steps_interval):

    Logger = PrintLogger if master_process else DummyLogger
    return Logger(
        time_interval=time_interval,
        progress_ratio_interval=progress_ratio_interval,
        progress_steps_interval=progress_steps_interval,
        ntfy_path=f"{project}_{_id}",
    )


def sync_barrier():
    if ddp:
        barrier()

def wandb_init(project, _id, config):
    if master_process:
        wandb.init(
            project=project,
            id=_id,
            config=config,
            allow_val_change=True,
            dir=get_project_root() / "data",
            resume="allow",
        )

def wandb_load_state(_id):
    checkpoints_dir = get_project_root() / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    load_artifact_name = f"{_id}_state"
    if master_process:
        with dbm.open('torch', 'c') as db:
            try:
                trainer_artifact = wandb.use_artifact(f"{load_artifact_name}:latest")
                trainer_artifact.download(root=checkpoints_dir)
                db['got_chkpt'] = 'true'
            except:
                db['got_chkpt'] = 'false'
    sync_barrier() 
    with dbm.open('torch', 'c') as db:
        if db['got_chkpt'].decode() == 'true':
            state = torch.load(checkpoints_dir / load_artifact_name, map_location=device)
        else:
            state = None
    return state

def wandb_log_state(trainer, _id):
    """Logs `trainer.state_dict()`"""
    checkpoints_dir = get_project_root() / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"{_id}_state"
    if master_process:
        trainer_artifact = wandb.Artifact(name=artifact_name, type="model")
        state_file = checkpoints_dir / f"state_{trainer.epoch - 1}_{_id}"
        torch.save(trainer.state_dict(), state_file)
        trainer_artifact.add_file(state_file, name=artifact_name)
        wandb.log_artifact(
            trainer_artifact, aliases=["latest", f"epoch{trainer.epoch - 1}"]
        )
    sync_barrier()