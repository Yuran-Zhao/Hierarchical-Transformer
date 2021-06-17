import argparse
import os
from itertools import chain

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer

from process_data.utils import CURRENT_DATA_BASE, ORIGINAL_DATA_BASE, read_file

BASE_PATH = "/home/ming/malware/inst2vec_bert/bert/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a word level tokenizer for ASM_BERT"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1016,
        help="The size of vocabulary used to train the tokenizer.",
    )
    parser.add_argument(
        "--padding_length",
        type=int,
        default=3,
        help="The length will be padded to by the tokenizer.",
    )
    args = parser.parse_args()

    return args


def train_tokenizer(args, dataset):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )

    tokenizer.train_from_iterator(dataset, trainer)

    return tokenizer


def save_tokenizer(tokenizer, tokenizer_file):
    tokenizer.save(tokenizer_file)


def load_tokenizer(tokenizer_file):
    if not os.path.exists(tokenizer_file):
        print("{} doesn't exist, will be retrained...".format(tokenizer_file))
        return None
    print("The tokenizer has already been trained.")
    return Tokenizer.from_file(tokenizer_file)


def post_process(tokenizer):
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        # pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        # special_tokens=[
        #     ("[CLS]", tokenizer.token_to_id("[CLS]")),
        #     ("[SEP]", tokenizer.token_to_id("[SEP]")),
        # ],
    )
    return tokenizer


def tokenizer_encode(tokenizer, data):
    return tokenizer.encode_batch(data)


def main(tokenizer_file=""):
    args = parse_args()

    tokenizer = load_tokenizer(tokenizer_file)

    if tokenizer is not None:
        return

    # json_files = [
    #     os.path.join(CURRENT_DATA_BASE, "inst.1.{}.json".format(i)) for i in range(128)
    # ]
    # dataset = load_dataset("json", data_files=json_files, field="data")

    text_files = [
        os.path.join(CURRENT_DATA_BASE, "inst_of_block.{}".format(i)) for i in range(59)
    ]

    dataset = []
    for f in text_files:
        tmp = read_file(f)
        dataset += [block[:-1].split("\t") for block in tmp]

    print("Get {} instructions".format(len(dataset)))

    print("Trainging tokenizer...")
    tokenizer = train_tokenizer(args, chain.from_iterable(dataset))
    tokenizer = post_process(tokenizer)
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[PAD]"),
        pad_token="[PAD]",
        length=args.padding_length,
    )
    save_tokenizer(tokenizer, tokenizer_file)


if __name__ == "__main__":
    main(os.path.join(CURRENT_DATA_BASE, "tokenizer-inst.all.json"))
