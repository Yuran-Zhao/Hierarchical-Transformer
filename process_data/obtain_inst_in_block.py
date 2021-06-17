import os
from multiprocessing import Pool, Process

from tqdm import tqdm

from utils import ORIGINAL_DATA_BASE, get_one_function_all_blocks

BASE = 900
TARGET = "./inst_of_block"


# def worker(subdirs, index):
#     with open(TARGET + ".{}".format(index), "w", encoding="utf-8") as fout:
#         print(
#             "Sub-sub-process {} is writing to {}...".format(
#                 os.getpid(), TARGET + ".{}".format(index)
#             )
#         )
#         for d in tqdm(subdirs):
#             for f in tqdm(os.listdir(d)):
#                 f = os.path.join(d, f)
#                 blocks = get_one_function_all_blocks(f)
#                 for block in blocks:
#                     fout.write("\t".join(block) + "\n")


def dir_worker(subdirs, index):
    for subdir in tqdm(subdirs):
        print("Sub-process {} start working on {}".format(os.getpid(), subdir))
        filenames = [os.path.join(subdir, filename) for filename in os.listdir(subdir)]
        with open(TARGET + ".{}".format(index), "a", encoding="utf-8") as fout:
            for filename in tqdm(filenames):
                blocks = get_one_function_all_blocks(filename)
                for block in blocks:
                    fout.write("\t".join(block) + "\n")


# def obtain(dir_path):
#     for subdir in os.listdir(dir_path):
#         worker(os.path.join(dir_path, subdir), subdir + ".block")


# def get_total_functions_files():
#     filenames = []
#     for f in tqdm(["linux32_0{}xxxx".format(i) for i in range(6)]):
#         print("Processing {}...".format(f))
#         dir_path = os.path.join(ORIGINAL_DATA_BASE, f)
#         for subdir in tqdm(os.listdir(dir_path)):
#             subdir = os.path.join(dir_path, subdir)
#             filenames += [
#                 os.path.join(subdir, filename) for filename in os.listdir(subdir)
#             ]
#     print("There are {} functions in the datasets".format(len(filenames)))
#     return filenames


def main():
    dirs = []
    for f in [
        os.path.join(ORIGINAL_DATA_BASE, "linux32_0{}xxxx".format(i)) for i in range(6)
    ]:
        dirs += [os.path.join(f, sub_dir) for sub_dir in tqdm(os.listdir(f))]
    print("We get {} sub-dirs".format(len(dirs)))

    total_p = Pool(36)

    for i in range(64):
        total_p.apply_async(dir_worker, args=(dirs[i * BASE : (i + 1) * BASE], i,))

    print("Waiting for all sub-processes done...")
    total_p.close()
    total_p.join()
    print("All sub-processes done.")


if __name__ == "__main__":
    main()
    # main()
