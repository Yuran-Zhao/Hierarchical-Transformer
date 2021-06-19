import json
import os
import pdb
from itertools import combinations
from multiprocessing import Pool, Process

from tqdm import tqdm

from utils import (
    CURRENT_DATA_BASE_FOR_MIDDLE,
    ORIGINAL_DATA_BASE_FOR_MIDDLE,
    get_one_function_all_blocks,
)

BASE = "coreutils-build-O"


def worker(project_name, index):
    print("Process {} is ready to process on {}".format(os.getpid(), project_name))
    dirs = [
        os.path.join(ORIGINAL_DATA_BASE_FOR_MIDDLE, BASE + str(i)) for i in range(4)
    ]

    function_names = [set(os.listdir(os.path.join(d, project_name))) for d in dirs]

    examples = []
    for i in range(4):
        for j in range(i + 1, 4):
            functions = function_names[i] & function_names[j]
            for func_name in tqdm(functions):
                func1 = get_one_function_all_blocks(
                    os.path.join(os.path.join(dirs[i], project_name), func_name)
                )
                func2 = get_one_function_all_blocks(
                    os.path.join(os.path.join(dirs[j], project_name), func_name)
                )
                examples.append({"is_cloned": 1, "func1": func1, "func2": func2})
    target_json = os.path.join(
        CURRENT_DATA_BASE_FOR_MIDDLE, "same_func.{}.json".format(index)
    )
    print(
        "Process {} get {} pairs of functions and write to {}".format(
            os.getpid(), len(examples), target_json,
        )
    )
    with open(target_json, "w") as fout:
        json.dump(examples, fout)


def main():
    p = Pool(36)
    for index, project_name in tqdm(
        enumerate(os.listdir(os.path.join(ORIGINAL_DATA_BASE_FOR_MIDDLE, BASE + "0")))
    ):
        p.apply_async(worker, args=(project_name, index,))

    print("Waiting for all sub-processes done...")
    p.close()
    p.join()
    print("All sub-processes done.")


if __name__ == "__main__":
    main()
    # worker("[", 0)
