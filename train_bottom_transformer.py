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

import argparse
import logging
import math
import os
import pdb
import random

import datasets
import numpy as np
import tokenizers
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from tokenizers.processors import TemplateProcessing
from torch.nn import DataParallel
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BatchEncoding,
    BertConfig,
    BertForMaskedLM,
    BertModel,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
    set_seed,
)

HIDDEN_SIZE = 96
MAX_LENGTH = 251

from bottom_data_collator import BottomDataCollatorForMaskedAssemblyModel
from bottom_transformer import BottomTransformer
from process_data.utils import CURRENT_DATA_BASE_FOR_BOTTOM

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

    # we take control of the load of dataset by oursevles
    # there will be several json file for training
    # `raw_dataset` has two features:
    #   `text`: "sentA\tsentB"
    #   `is_next`: 0 or 1
    # raw_datasets = load_dataset(
    #     "json",
    #     data_files={
    #         "train": "/home/ming/malware/inst2vec_bert/data/test_lm/inst.json",
    #         "validation": "/home/ming/malware/inst2vec_bert/data/test_lm/inst.json",
    #     },
    #     field="data",
    # )

    # the format in train_files should be
    # {"data" =
    #   [
    #       {"text": [ inst1, inst2, ..., inst_n ]},
    #       {"text": [ inst1, inst2, ..., inst_n ]},
    #       ...
    #       {"text": [ inst1, inst2, ..., inst_n ]},
    #   ]
    # }
    # each inst is like [opcode, operand_1, operand_2]
    train_files = [
        os.path.join(
            CURRENT_DATA_BASE_FOR_BOTTOM, "inst_of_block.{}.clean.json".format(i)
        )
        for i in range(29)
    ]
    valid_file = os.path.join(CURRENT_DATA_BASE_FOR_BOTTOM, "inst_of_block.test.json")

    raw_datasets = load_dataset(
        "json",
        data_files={"train": train_files, "validation": valid_file,},
        field="data",
    )

    # train_files = "/home/ming/malware/inst2vec_bert/H-Transformer/data/test.data.json"

    # raw_datasets = load_dataset(
    #     "json", data_files={"train": train_files}, field="data",
    # )

    # we use the tokenizer previously trained on the dataset above
    tokenizer = tokenizers.Tokenizer.from_file(
        "/home/ming/malware/inst2vec_bert/data/asm_bert/tokenizer-inst.all.json"
    )

    # NOTE: have to promise the `length` here is consistent with the one used in `train_my_tokenizer.py`
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=3
    )
    tokenizer.post_processor = TemplateProcessing(single="$A",)

    # # NOTE: `max_position_embeddings` here should be consistent with `length` above
    # # use transformer, so the num_hidden_layers = 1
    # config = BertConfig(
    #     vocab_size=tokenizer.get_vocab_size(),
    #     hidden_size=HIDDEN_SIZE,
    #     num_hidden_layers=1,
    #     num_attention_heads=8,
    #     intermediate_size=HIDDEN_SIZE * 4,
    #     max_position_embeddings=32,
    # )

    # bottom_embedding = BottomEmbedding(opcode_size, operand_size, HIDDEN_SIZE // 3)
    # bottom_embedding = DataParallel(bottom_embedding)

    # # initalize a new BERT for Masked Assemble Model
    # model = BertModel(config)
    # model = DataParallel(model)

    # bottom_headlayer = BottomHeadLayer(opcode_size, operand_size, HIDDEN_SIZE // 3)
    # bottom_headlayer = DataParallel(bottom_headlayer)

    model = BottomTransformer(
        tokenizer.get_vocab_size(),
        tokenizer.get_vocab_size(),
        tokenizer.token_to_id("[PAD]"),
        d_model=HIDDEN_SIZE,
        n_head=8,
        num_layers=6,
        max_length=MAX_LENGTH,
        device="cuda:0",
    )
    model = DataParallel(model)

    loss_function = torch.nn.NLLLoss(ignore_index=-100)

    # Preprocessing the datasets.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # First we aplly `tokenize_function` on the dataset.

    def tokenize_function(examples):
        # pdb.set_trace()
        blocks = examples["text"]
        # print(blocks)
        encoded_inputs = {}
        results = [tokenizer.encode_batch(block) for block in blocks]
        # NOTE
        # Assumption: every instruction consists of (opcode, operand_1, operand_2)
        # so only need first three ids
        encoded_inputs["input_ids"] = [
            [result.ids[:3] for result in block] for block in results
        ]
        # encoded_inputs["token_type_ids"] = [
        #     [result.type_ids[0] for result in block] for block in results
        # ]
        encoded_inputs["special_tokens_mask"] = [
            [result.special_tokens_mask[0] for result in block] for block in results
        ]
        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type="np", prepend_batch_axis=False,
        )
        return batch_outputs

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = BottomDataCollatorForMaskedAssemblyModel(
        tokenizer=tokenizer
    )  # , mlm_probability=args.mlm_probability)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
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
    ) = accelerator.prepare(
        model, optimizer, loss_function, train_dataloader, eval_dataloader,
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
            # pdb.set_trace()
            # input_ids `(batch_size, max_length, 3)`
            input_ids = batch.pop("input_ids", None)
            # masks `(batch_size, max_length)`
            masks = batch.pop("special_tokens_mask", None)
            predictions = (
                model(input_ids, masks, header=True).permute(0, 3, 1, 2).contiguous()
            )

            labels = batch.pop("labels", None)
            # labels = labels.unsqueeze(dim=-1).permute(0, 3, 1, 2).contiguous()
            loss = loss_function(predictions, labels)
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
                losses = []
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        input_ids = batch.pop("input_ids", None)
                        masks = batch.pop("special_tokens_mask", None)
                        predictions = (
                            model(input_ids, masks, header=True)
                            .permute(0, 3, 1, 2)
                            .contiguous()
                        )
                        labels = batch.pop("labels", None)
                        loss = loss_function(predictions, labels)
                    loss = loss.sum()

                    losses.append(
                        accelerator.gather(loss.repeat(args.per_device_eval_batch_size))
                    )

                losses = torch.cat(losses)
                # losses = losses[: len(eval_dataset)]
                try:
                    perplexity = math.exp(torch.mean(losses))
                except OverflowError:
                    perplexity = float("inf")

                logger.info(f"steps {completed_steps}: perplexity: {perplexity}")
                model.train()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(
            unwrapped_model.state_dict(),
            os.path.join(args.output_dir, "bottom_transformer_state_dict"),
        )
        print(
            "The Bottom Transformer has been save to {}".format(
                os.path.join(args.output_dir, "bottom_transformer_state_dict")
            )
        )

        # pdb.set_trace()
        # unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()
