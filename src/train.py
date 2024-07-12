from argparse import Namespace
from dataclasses import asdict
import os
from time import time

import torch
import wandb
from torch.nn.functional import cross_entropy
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.config import Config
from src.model import Transformer
from src.utils import (
    get_dataset,
    get_device,
    log,
    pack_sequences,
    process_and_tokenize,
    setup_tokenizer,
)


def train(args: Namespace) -> None:
    wandb.init(
        # setup wandb project
        project="gpt2-reproduction",
        # track config
        config=asdict(Config()).update({k + "_args": v for k, v in vars(args).items()}),
    )

    device = get_device(args)
    cpus = os.cpu_count()
    assert cpus is not None

    dataset = get_dataset(args)
    tokenizer = setup_tokenizer(args)

    dataset = dataset.map(
        lambda data: process_and_tokenize(data, tokenizer),
        batched=True,
        num_proc=cpus - 2,
        writer_batch_size=750,
    )
    dataset = dataset.map(
        lambda data: pack_sequences(data, tokenizer),
        batched=True,
        num_proc=cpus - 2,
        writer_batch_size=750,
        remove_columns=["attention_mask"],
    )
    train_data = dataset["train"].with_format("torch", device=device)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)  # type: ignore
    val_data = dataset["test"].with_format("torch", device=device)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)  # type: ignore

    uncompiled_model = Transformer()
    uncompiled_model.to(device)
    uncompiled_model.train()

    if args.resume:
        checkpoint = torch.load(os.path.join(args.checkpoint_path, "pretrain_ckpt.pt"))
        state_dict = checkpoint["model"]
        uncompiled_model.load_state_dict(state_dict)
        print(
            f"Resuming from checkpoint, iteration: {checkpoint['iteration']}, best_val_loss: {checkpoint['best_val_loss']:.4f}"
        )

        model = torch.compile(uncompiled_model)
        assert isinstance(model, torch.nn.Module)

        # Setup an optimizer
        optimizer = uncompiled_model.setup_optimizer(0.1, args.lr, args.gpu)
        optimizer.load_state_dict(checkpoint["optimizer"])
        linear_warmup = LinearLR(
            optimizer,
            start_factor=min(
                (1 / args.warmup_iterations * checkpoint["iteration"]), 1.0
            ),
            total_iters=max(args.warmup_iterations - checkpoint["iteration"], 0),
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.max_iterations,
            eta_min=args.lr * 0.1,
            last_epoch=checkpoint["iteration"],
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[linear_warmup, cosine_scheduler],
            milestones=[args.warmup_iterations],
        )
        scheduler.load_state_dict(checkpoint["scheduler"])

        iteration_start = checkpoint["iteration"] + 1
        best_val_loss = checkpoint["best_val_loss"]

    else:
        model = torch.compile(uncompiled_model)
        assert isinstance(model, torch.nn.Module)

        # Setup an optimizer
        optimizer = uncompiled_model.setup_optimizer(0.1, args.lr, args.gpu)
        linear_warmup = LinearLR(
            optimizer,
            start_factor=1 / args.warmup_iterations,
            total_iters=args.warmup_iterations,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.max_iterations, eta_min=args.lr * 0.1
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[linear_warmup, cosine_scheduler],
            milestones=[args.warmup_iterations],
        )

        iteration_start = 1
        best_val_loss = float("inf")

    # Setup training loop and associated variables
    train_data_iterator = iter(train_dataloader)
    val_data_iterator = iter(val_dataloader)

    for iteration in range(iteration_start, args.max_iterations + 1):
        # Reset dataloader on epoch boundary
        try:
            batch = next(train_data_iterator)
        except StopIteration:
            train_data_iterator = iter(train_dataloader)
            batch = next(train_data_iterator)

        # Get tokens
        tokens = batch["input_ids"].to(device=device, non_blocking=True)  # (B x T)
        input_sequence, target_sequence = tokens[:, :-1], tokens[:, 1:]
        num_tokens = input_sequence.shape[0] * input_sequence.shape[1]

        s = time()
        optimizer.zero_grad()
        # we get bfloat16 for free!
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(input_sequence)
            loss = cross_entropy(
                logits.view(-1, logits.size(-1)),  # convert to (B * T x M)
                target_sequence.reshape(-1),  # convert to (B * T)
                ignore_index=Config.padding_idx,
            )
        loss.backward()
        norm = clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        e = time()

        if iteration % args.log_at == 0:
            # update metrics
            log(
                {
                    "iteration": iteration,
                    "norm": norm,
                    "lr": scheduler.get_last_lr()[0],
                    "time": (e - s) * 1000,
                    "tok/ps": num_tokens / (e - s),
                }
            )

        # Evaluate if needed
        if iteration % args.evaluate_at == 0:
            s = time()
            model.eval()
            val_losses = []
            # evaluate
            with torch.no_grad():
                for _ in range(200):
                    try:
                        val_batch = next(val_data_iterator)
                    except StopIteration:
                        train_data_iterator = iter(val_dataloader)
                        val_batch = next(val_data_iterator)
                    tokens = val_batch["input_ids"].to(
                        device=device, non_blocking=True
                    )  # (B x T)
                    input_sequence, target_sequence = tokens[:, :-1], tokens[:, 1:]
                    logits = model(input_sequence)
                    loss = cross_entropy(
                        logits.view(-1, logits.size(-1)),  # convert to (B * T x M)
                        target_sequence.reshape(-1),  # convert to (B * T)
                        ignore_index=Config.padding_idx,
                    )
                    val_losses.append(loss.item())
            mean_val_loss = sum(val_losses) / len(val_losses)
            e = time()

            print(
                f"iteration: {iteration}, val_loss: {mean_val_loss:.4f}, val loop time: {(e - s) * 1000:.2f}ms",
                flush=True,
            )

            log(
                {
                    "iteration": iteration,
                    "val_loss": mean_val_loss,
                    "val_time": (e - s) * 1000,
                }
            )

            if mean_val_loss < best_val_loss:
                # save a checkpoint
                path = os.path.join(args.checkpoint_path, "pretrain_ckpt.pt")
                # we use uncompiled model to avoid `_orig_model` issues in loading
                # the weights are the same so it should be fine
                checkpoint = {
                    "model": uncompiled_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": args,
                    "iteration": iteration,
                    "best_val_loss": best_val_loss,
                }
                torch.save(
                    checkpoint,
                    path,
                )
                print(f"Saved checkpoint to {path}")
                best_val_loss = mean_val_loss

            model.train()

        scheduler.step()
