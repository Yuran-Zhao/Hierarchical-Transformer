import json
import os
from multiprocessing import Pool

from tqdm import tqdm

from utils import CURRENT_DATA_BASE, read_file


def worker(filename):
    examples = []
    blocks = read_file(filename)
    for block in tqdm(blocks):
        examples.append({"text": block[:-1].split("\t")})
    print("Writing to {}...".format(filename + ".json"))
    results = {"data": examples}
    with open(filename + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f)


def main():

    total_p = Pool(36)

    for i in range(30):
        total_p.apply_async(
            worker,
            args=(os.path.join(CURRENT_DATA_BASE, "inst_of_block.{}.clean".format(i)),),
        )

    print("Waiting for all sub-processes done...")
    total_p.close()
    total_p.join()
    print("All sub-processes done.")


if __name__ == "__main__":
    main()
