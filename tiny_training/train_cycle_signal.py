import argparse
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import signal
import torchvision
import atexit

def is_master(global_rank):
    return global_rank == 0

def save_checkpoint(*, path, model, optimizer, step):
    print("Saving checkpoint")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }, path)

def load_checkpoint(*, path, model, optimizer, device) -> int:
    print("loading checkpoint")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"]

done = False
 

def shandler(signum, frame):
    print("setting self.done=True")
    global done
    done = True
signal.signal(signal.SIGINT, shandler)
signal.signal(signal.SIGTERM, shandler)

def main(args):
    checkpoint_path = args.checkpoint_path
    print(f"Checkpoint Path: {checkpoint_path}")
    max_steps = args.max_steps

    dist.init_process_group("nccl")
    global_rank = dist.get_rank()
    device_id = global_rank % torch.cuda.device_count()
    print(f"Global Rank: {global_rank}, Device Count {torch.cuda.device_count()}, Device ID: {device_id}")
    world_size = dist.get_world_size()
    torch.cuda.set_device(device_id)

    model = torchvision.models.resnet18().to(device_id)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)

    starting_step = 0
    if checkpoint_path.exists():
        starting_step = load_checkpoint(path=checkpoint_path, model=model, optimizer=optimizer, device=f"cuda:{device_id}")
    if starting_step >= max_steps:
        return

    model = DDP(model, device_ids=[device_id])
    batch = torch.randn(64, 3, 224, 224).to(device_id)

    atexit.register(lambda: save_checkpoint(path=checkpoint_path, model=model.module, optimizer=optimizer, step=i))

    if done:
        raise Exception("Cycle Finished: Did not setup fast enough")
    for i in range(starting_step, max_steps):
        if is_master(global_rank):
            print(f"Step {i} of {max_steps}")

        optimizer.zero_grad()
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()

        if done:
            print(f"Step {i} of {max_steps} was interrupted")
            break

    if is_master(global_rank):
        save_checkpoint(path=checkpoint_path, model=model.module, optimizer=optimizer, step=i)
        print("Training Complete")

    dist.barrier()
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()
    main(args)