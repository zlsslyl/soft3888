import argparse
from pathlib import Path
import torch
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import json
import signal
import enum

def is_master(global_rank):
    return global_rank == 0

def save_checkpoint(path, step):
    print("Saving checkpoint")
    torch.save({"step": step}, path)

def load_checkpoint(path) -> int:
    print("loading checkpoint")
    checkpoint = torch.load(path)
    return checkpoint["step"]

is_at_cycle_end = False

def cycle_end_signal_handler(signum, frame):
    print("HANDLING SIGNAL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
    global is_at_cycle_end
    is_at_cycle_end = True

signal.signal(signal.SIGINT, cycle_end_signal_handler)
signal.signal(signal.SIGTERM, cycle_end_signal_handler)

class BreakReason(enum.Enum):
    STEPS_COMPLETED = 1
    CYCLE_END = 2

def main(args):
    checkpoint_path = args.checkpoint_path
    max_steps = args.max_steps

    dist.init_process_group("nccl")
    global_rank = dist.get_rank()
    device_id = global_rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    torch.cuda.set_device(device_id)

    starting_step = 0
    if checkpoint_path.exists():
        starting_step = load_checkpoint(checkpoint_path)
    if starting_step >= max_steps:
        return

    for i in range(starting_step, max_steps):
        print(f"Step {i} of {max_steps}")
        time.sleep(1) # doing work

        if is_at_cycle_end:
            break

    if is_master(global_rank):
        save_checkpoint(checkpoint_path, i)
        print("Training Complete")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()
    main(args)