# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
"""
Modified from
https://github.com/pytorch/audio/blob/main/examples/pipeline_tacotron2/train.py
for Strong Compute ISC

Changes: 
- Assumes that processed dataset is found in a safetensors file
- Removed mp.spawn and replaced with torchrun
"""

import argparse
import logging
import os
import random
from datetime import datetime
from functools import partial
from time import time

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from model.tacotron2 import SyncedTacotron2
from tqdm import tqdm

plt.switch_backend("agg")

from pathlib import Path

from cycling_utils import InterruptableDistributedSampler, atomic_torch_save
from safetensors.torch import load_model, save_model, load_file 


from loss import Tacotron2Loss
from text.text_preprocessing import get_symbol_list

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(os.path.basename(__file__))

from parser_utils import parse_args


class SafetensorsLJSPEECH(Dataset):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.file_list = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.safetensors')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        sf = load_file(filename)
        # unpack the safetensors dict
        batch = (
            sf["text_padded"],
            sf["batch_text_lengths"],
            sf["mel_specgram_padded"],
            sf["mel_specgram_lengths"],
            sf["gate_padded"]
        ) 
        return batch

def adjust_learning_rate(epoch, optimizer, learning_rate, anneal_steps, anneal_factor):
    """Adjust learning rate base on the initial setting."""
    p = 0
    if anneal_steps is not None:
        for _, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p = p + 1

    if anneal_factor == 0.3:
        lr = learning_rate * ((0.1 ** (p // 2)) * (1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate * (anneal_factor**p)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def batch_to_gpu(batch):
    (
        text_padded,
        text_lengths,
        mel_specgram_padded,
        mel_specgram_lengths,
        gate_padded,
    ) = batch
    text_padded = to_gpu(text_padded).long()
    text_lengths = to_gpu(text_lengths).long()
    mel_specgram_padded = to_gpu(mel_specgram_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    mel_specgram_lengths = to_gpu(mel_specgram_lengths).long()
    x = (text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths)
    y = (mel_specgram_padded, gate_padded)
    return x, y


def training_step(model, train_batch, batch_idx):
    (
        text_padded,
        text_lengths,
        mel_specgram_padded,
        mel_specgram_lengths,
    ), y = batch_to_gpu(train_batch)
    y_pred = model(text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths)
    y[0].requires_grad = False
    y[1].requires_grad = False
    losses = Tacotron2Loss()(y_pred[:3], y)
    return losses[0] + losses[1] + losses[2], losses


def validation_step(model, val_batch, batch_idx):
    (
        text_padded,
        text_lengths,
        mel_specgram_padded,
        mel_specgram_lengths,
    ), y = batch_to_gpu(val_batch)
    y_pred = model(text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths)
    losses = Tacotron2Loss()(y_pred[:3], y)
    return losses[0] + losses[1] + losses[2], losses


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if rt.is_floating_point():
        rt = rt / world_size
    else:
        rt = rt // world_size
    return rt


def log_additional_info(writer, model, loader, epoch):
    model.eval()
    data = next(iter(loader))
    with torch.no_grad():
        (
            text_padded,
            text_lengths,
            mel_specgram_padded,
            mel_specgram_lengths,
        ), _ = batch_to_gpu(data)
        y_pred = model(
            text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths
        )
        mel_out, mel_out_postnet, gate_out, alignment = y_pred

    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(mel_out[0].cpu().numpy())
    writer.add_figure("trn/mel_out", fig, epoch)
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(mel_out_postnet[0].cpu().numpy())
    writer.add_figure("trn/mel_out_postnet", fig, epoch)
    writer.add_image(
        "trn/gate_out", torch.tile(gate_out[:1], (10, 1)), epoch, dataformats="HW"
    )
    writer.add_image("trn/alignment", alignment[0], epoch, dataformats="HW")


def train(rank, world_size, args):
    if rank == 0 and args.logging_dir:
        if not os.path.isdir(args.logging_dir):
            os.makedirs(args.logging_dir)
        filehandler = logging.FileHandler(os.path.join(args.logging_dir, "train.log"))
        filehandler.setLevel(logging.INFO)
        logger.addHandler(filehandler)

        writer = SummaryWriter(log_dir=args.logging_dir)
    else:
        writer = None

    torch.manual_seed(0)

    torch.cuda.set_device(rank)

    symbols = get_symbol_list(args.text_preprocessor)
    
    if rank == 0:
        logger.info("Initialising model")
    
    model = SyncedTacotron2(
        mask_padding=args.mask_padding,
        n_mels=args.n_mels,
        n_symbol=len(symbols),
        n_frames_per_step=args.n_frames_per_step,
        symbol_embedding_dim=args.symbols_embedding_dim,
        encoder_embedding_dim=args.encoder_embedding_dim,
        encoder_n_convolution=args.encoder_n_convolution,
        encoder_kernel_size=args.encoder_kernel_size,
        decoder_rnn_dim=args.decoder_rnn_dim,
        decoder_max_step=args.decoder_max_step,
        decoder_dropout=args.decoder_dropout,
        decoder_early_stopping=(not args.decoder_no_early_stopping),
        attention_rnn_dim=args.attention_rnn_dim,
        attention_hidden_dim=args.attention_hidden_dim,
        attention_location_n_filter=args.attention_location_n_filter,
        attention_location_kernel_size=args.attention_location_kernel_size,
        attention_dropout=args.attention_dropout,
        prenet_dim=args.prenet_dim,
        postnet_n_convolution=args.postnet_n_convolution,
        postnet_kernel_size=args.postnet_kernel_size,
        postnet_embedding_dim=args.postnet_embedding_dim,
        gate_threshold=args.gate_threshold,
    ).cuda(rank)

    if rank == 0:
        logger.info("converting batchnorm in model")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if rank == 0:
        logger.info("Finished initialising model")

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 0

    checkpoint_dir_path = Path(args.checkpoint_dir)
    checkpoint_dir_path = checkpoint_dir_path.expanduser()

    model_checkpoint_path = checkpoint_dir_path / "model.sf"
    train_state_checkpoint_path = checkpoint_dir_path / "train_state.pt"
    temp_checkpoint_dir_path = Path(args.checkpoint_dir + "_temp",  )

    loader_params = {
        "batch_size": 1, # batch sized is determined by the pre train script 
        "num_workers": args.workers,
        "prefetch_factor": 1024,
        "persistent_workers": True,
        "shuffle": False,
        "pin_memory": True,
        "drop_last": False,
    }

    batched_train_set =  SafetensorsLJSPEECH("ljspeech_batches")
    batched_train_sampler = InterruptableDistributedSampler(batched_train_set)
    batched_train_loader = DataLoader(batched_train_set, sampler=batched_train_sampler, **loader_params)

    if checkpoint_dir_path.is_dir():
        if model_checkpoint_path.is_file() and train_state_checkpoint_path.is_file():
            
            logger.info("Loading the model checkpoint")
            load_model(model, str(model_checkpoint_path))
            
            logger.info(f"Loading train state checkpoint data")
            train_state_checkpoint = torch.load(train_state_checkpoint_path)
            start_epoch = train_state_checkpoint["epoch"]
            optimizer.load_state_dict(train_state_checkpoint["optimizer"])
            batched_train_sampler.load_state_dict(train_state_checkpoint["sampler"])
        else: 
            raise Exception("Not found both a model and a checkpoint file!")
    else:
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    dist.barrier()
    model.train()

    for epoch in range(start_epoch, args.epochs):
        start = time()

        trn_loss, counts = 0, 0

        if rank == 0:
            batch_iterator = tqdm(
                enumerate(batched_train_loader), desc=f"Epoch {epoch}", total=len(batched_train_loader)
            )
        else:
            batch_iterator = enumerate(batched_train_loader)

        for i, batch in batch_iterator:
            adjust_learning_rate(
                epoch,
                optimizer,
                args.learning_rate,
                args.anneal_steps,
                args.anneal_factor,
            )
            model.zero_grad()
            loss, losses = training_step(model, batch, i)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batched_train_sampler.advance(1)

            if rank == 0 and writer:
                
                print({"loss":loss}) 

                global_iters = epoch * len(batched_train_loader)

                writer.add_scalar("trn/mel_loss", losses[0], global_iters)
                writer.add_scalar("trn/mel_postnet_loss", losses[1], global_iters)
                writer.add_scalar("trn/gate_loss", losses[2], global_iters)

            trn_loss += loss * len(batch[0])
            counts += len(batch[0])

            if rank == 0 and (global_iters % args.checkpoint_freq + 1) == 0:
                logger.info("saving checkpoint")
                
                temp_checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
                save_model(model, str(temp_checkpoint_dir_path / "model.sf"))
                atomic_torch_save(
                    {
                        "epoch": epoch,
                        "optimizer": optimizer.state_dict(),
                        "sampler": batched_train_sampler.state_dict()
                    },
                    temp_checkpoint_dir_path / "train_state.pt",
                )
                os.replace(temp_checkpoint_dir_path, checkpoint_dir_path)
                logger.info("saved checkpoint")
            
        trn_loss = trn_loss / counts

        trn_loss = reduce_tensor(trn_loss, world_size)
        if rank == 0:
            logger.info(f"[Epoch: {epoch}] time: {time()-start}; trn_loss: {trn_loss}")
            if writer:
                writer.add_scalar("trn_loss", trn_loss, epoch)

        # to do - add validation 

    dist.destroy_process_group()


def main(args):
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    if global_rank == 0:
        logger.info("Start time: {}".format(str(datetime.now())))
        logger.info(f"# available GPUs: {world_size}")

    torch.manual_seed(0)
    random.seed(0)

    train(global_rank, world_size, args)
    
    if global_rank == 0:
        logger.info(f"End time: {datetime.now()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Tacotron 2 Training")
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    main(args)
