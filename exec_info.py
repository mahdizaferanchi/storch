import os
import torch
from torch.distributed import init_process_group

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ddp_world_size = 1
    ddp_rank = None
    ddp_local_rank = None

