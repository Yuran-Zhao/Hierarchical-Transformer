import os
from multiprocessing import Pool, Queue

from tqdm import tqdm

from utils import CURRENT_DATA_BASE_FOR_BOTTOM, read_file

max_length_q = Queue()
cnt_q = Queue()


def worker(filename, index):
    print("Process {} is working on {}".format(os.getpid(), filename))
    blocks = read_file(filename)
    cnt = 0
    # max_length = 0
    rets = []
    for block in tqdm(blocks):
        length = len(block[:-1].split("\t"))
        # max_length = max(max_length, length)
        if length > 250 or length < 5:
            continue
        rets.append(block)
        cnt += length
    with open(filename + ".clean", "w", encoding="utf-8") as fout:
        for block in rets:
            fout.write(block)
    print(
        "{} has {} blocks containing {} instructions".format(filename, len(rets), cnt)
    )
    # max_length_q.put(max_length)
    # cnt_q.put(cnt)
    # print("The max_length in {} is {}".format(filename, max_length))


def main():

    total_p = Pool(36)

    for i in range(59):
        total_p.apply_async(
            worker,
            args=(
                os.path.join(
                    CURRENT_DATA_BASE_FOR_BOTTOM, "inst_of_block.{}".format(i)
                ),
                i,
            ),
        )

    print("Waiting for all sub-processes done...")
    total_p.close()
    total_p.join()
    print("All sub-processes done.")
    max_length = []
    for i in range(59):
        max_length.append(max_length_q.get())
    print("The max_length in all datasets is {}".format(max(max_length)))
    cnt = []
    for i in range(59):
        cnt.append(cnt_q.get())
    print("The total number of instructions in all datasets is {}".format(sum(cnt)))


if __name__ == "__main__":
    main()
