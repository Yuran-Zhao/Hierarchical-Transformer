#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 Alan Zhao. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# """
# Pre-train the BERT on a dataset without using HuggingFace Trainer.
# """
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.
from sklearn.metrics import classification_report
import argparse
import logging
import math
import os
import pdb
import random

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.nn import DataParallel
from torch.utils.data.dataloader import DataLoader,Dataset
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)

HIDDEN_SIZE = 1280
# CLASS_NAMES = ["Adware", "Backdoor", "PWS", "Rogue", "Trojan", "TrojanDownloader", "TrojanDropper", "TrojanSpy", "VirTool", "Virus", "Worm"]
CLASS_NAMES = ["Adware_Win32_GameVance", "Adware_Win32_Rugo", "Backdoor_Win32_Cycbot_B", "Backdoor_Win32_Cycbot_G", "Backdoor_Win32_Hupigon",
    "Backdoor_Win32_Rbot", "PWS_Win32_Ceekat_gen_A", "PWS_Win32_Lolyda_BF", "PWS_Win32_OnLineGames_CPL", "PWS_Win32_Zbot", "Rogue_Win32_FakeRean",
    "Rogue_Win32_Winwebsec", "TrojanDownloader_Win32_Renos_LX", "TrojanDownloader_Win32_Renos_MJ", "TrojanDownloader_Win32_Renos_NS",
    "TrojanDownloader_Win32_Renos_ON", "TrojanDownloader_Win32_Renos_PG", "TrojanDownloader_Win32_Renos_PT", "Trojan_Win32_C2Lop_E",
    "Trojan_Win32_Delf_KP", "Trojan_Win32_Koutodoor_F", "VirTool_Win32_DelfInject_gen_X", "Worm_Win32_Vobfus", "Worm_Win32_Vobfus_GZ"]
# MAX_LENGTH = 500  # NOTE  NOT FOR SURE !

from bottom_transformer import BottomTransformer
from middle_transformer import MiddleTransformer
from process_data.utils import CURRENT_DATA_BASE_FOR_TOP
from top_data_collator import TopDataCollator, MyDataset
from top_transformer import TopTransformer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the bottom Transformer in I-MAD"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=40,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=5000,
        help="Number of steps before evaluating the model.",
    )

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    
    # train_info_file = "/home/ming/malware/HCNN/dataset_infos/train_infos_sm.txt"
    train_info_file = "/home/ming/malware/HCNN/dataset_infos/train_infos1.txt"
    with open(train_info_file, 'r') as f:
        train_file_info = f.readlines()
    train_dataset = MyDataset(train_file_info)
    
    # test_info_file = "/home/ming/malware/HCNN/dataset_infos/test_infos_sm.txt"
    test_info_file = "/home/ming/malware/HCNN/dataset_infos/test_infos1.txt"
    with open(test_info_file, 'r') as f:
        test_file_info = f.readlines()
    random.shuffle(test_file_info)
    test_dataset = MyDataset(test_file_info)
    
    eval_dataset = MyDataset(test_file_info[:200])

    model = TopTransformer(
        # middle_transformer,
        d_model=HIDDEN_SIZE,
        n_head=8,
        num_layers=6,
        device="cuda:0",
    )
    model = DataParallel(model)

    loss_function = torch.nn.NLLLoss()

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        loss_function,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model,
        optimizer,
        loss_function,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
    )
    # model, optimizer, train_dataloader = accelerator.prepare(
    #     model, optimizer, train_dataloader
    # )

    # model.to("cuda:0")

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Evalute every {args.eval_every_steps} steps")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            input_vec = batch.get('vectors', None)
            # print(input_vec.shape)

            outputs = model(input_vec)

            labels = batch.get("labels", None)

            loss = loss_function(outputs, labels)

            loss = loss.sum()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
                
            if completed_steps % args.eval_every_steps == 0:
                model.eval()
                total, correct = 0, 0
                losses = []
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        input_vec = batch.get('vectors', None)
                        outputs = model(input_vec)
                        labels = batch.get("labels", None)
                        loss = loss_function(outputs, labels)

                        predictions = torch.argmax(outputs, dim=-1)

                    loss = loss.sum()

                    losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

                    correct += (predictions == labels).sum().item()
                    total += labels.shape[0]

                losses = torch.cat(losses)
                # losses = losses[: len(eval_dataset)]

                logger.info(
                    f"steps {completed_steps}, loss: {torch.mean(losses).item()}, accuracy: {correct / total}"
                )
                model.train()
            
            # delete caches
            del input_vec, labels, outputs, loss
            torch.cuda.empty_cache()

    total, correct = 0, 0
    losses = []
    total_pred_y = []
    total_right_y = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            input_vec = batch.get('vectors', None)
            outputs = model(input_vec)
            labels = batch.get("labels", None)
            loss = loss_function(outputs, labels)

            predictions = torch.argmax(outputs, dim=-1)
            total_pred_y += predictions.tolist()
            total_right_y += labels.tolist()

        loss = loss.sum()

        losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

        correct += (predictions == labels).sum().item()
        total += labels.shape[0]

    losses = torch.cat(losses)
    # losses = losses[: len(eval_dataset)]

    logger.info(
        # f"Final result on test dataset: loss: {torch.mean(losses).item()}, accuracy: {correct / total}"
        classification_report(np.array(total_right_y), np.array(total_pred_y), target_names=list(CLASS_NAMES), digits=4)
    )

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(
            unwrapped_model.state_dict(),
            os.path.join(args.output_dir, "top_transformer_state_dict"),
        )
        print(
            "The Top Transformer has been save to {}".format(
                os.path.join(args.output_dir, "top_transformer_state_dict")
            )
        )


if __name__ == "__main__":
    main()
