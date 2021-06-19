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
    filename = os.path.join(
        CURRENT_DATA_BASE_FOR_MIDDLE, "same_func.{}.json".format(index)
    )
    print("Process {} is ready to process {}".format(os.getpid(), filename))
    with open(filename, "r") as f:
        postives = json.load(f)

    i = index
    while i == index:
        i = randint(0, 104)
    with open(
        os.path.join(CURRENT_DATA_BASE_FOR_MIDDLE, "same_func.{}.json".format(i)), "r"
    ) as f:
        negtives = json.load(f)
    results = []
    neg_length = len(negtives)
    for example in tqdm(postives):
        i = randint(1, 2)
        first = example["func{}".format(i)]
        n = randint(0, neg_length - 1)
        j = randint(1, 2)
        second = negtives[n]["func{}".format(j)]
        results.append({"is_cloned": 0, "func1": first, "func2": second})
    results = postives + results

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
