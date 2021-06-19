import json
import os
import pdb
from itertools import combinations
from multiprocessing import Pool, Process
from random import randint

from tqdm import tqdm

from utils import (  # ORIGINAL_DATA_BASE_FOR_MIDDLE,
    CURRENT_DATA_BASE_FOR_MIDDLE,
    get_one_function_all_blocks,
)


def worker(index):
    filename = os.path.join(CURRENT_DATA_BASE_FOR_MIDDLE, "func.{}.json".format(index))
    print("Process {} is ready to process {}".format(os.getpid(), filename))
    with open(filename, "r") as f:
        examples = json.load(f)
    examples = examples["data"]
    results = []
    for example in tqdm(examples):
        if (
            len(example["func1"]) <= 50
            and len(example["func2"]) <= 50
            and all(len(block) <= 50 for block in example["func1"])
            and all(len(block) <= 50 for block in example["func2"])
        ):
            results.append(example)

    print(
        "Process {} get {} examples and write to {}".format(
            os.getpid(),
            len(results),
            os.path.join(CURRENT_DATA_BASE_FOR_MIDDLE, "func.{}.json".format(index)),
        )
    )
    with open(
        os.path.join(CURRENT_DATA_BASE_FOR_MIDDLE, "func.{}.json".format(index)), "w"
    ) as f:
        json.dump({"data": results}, f)


def main():
    p = Pool(36)
    for index in range(105):
        p.apply_async(worker, args=(index,))

    print("Waiting for all sub-processes done...")
    p.close()
    p.join()
    print("All sub-processes done.")


if __name__ == "__main__":
    main()
