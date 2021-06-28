import json
import os
import pdb
from itertools import combinations
from multiprocessing import Pool, Process

from tqdm import tqdm

from utils import (
    CURRENT_DATA_BASE_FOR_TOP,
    ORIGINAL_DATA_BASE_FOR_TOP,
    get_one_function_all_blocks,
)

BASE = "coreutils-build-O"


def worker(binary_name):
    print("Process {}...".format(binary_name))
    functions = os.listdir(binary_name)
    results = []
    for f in tqdm(functions):
        results.append(get_one_function_all_blocks(os.path.join(binary_name, f)))

    return {"text": results, "is_malware":1}


def main():
    examples = []
    for index, binary_name in tqdm(
        enumerate(os.listdir(ORIGINAL_DATA_BASE_FOR_TOP))
    ):
        examples.append(worker(os.path.join(ORIGINAL_DATA_BASE_FOR_TOP, binary_name)))

    with open(os.path.join(CURRENT_DATA_BASE_FOR_TOP, 'test.json'), 'w') as f:
        json.dump({"data":examples}, f)


if __name__ == "__main__":
    main()
    # worker("[", 0)
