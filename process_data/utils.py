import json
import os
import pickle
import sys

from datatype import Function

sys.path.insert(0, "/home/ming/malware/inst2vec_bert")

ORIGINAL_DATA_BASE_FOR_BOTTOM = "/home/ming/malware/data/elfasm/"
CURRENT_DATA_BASE_FOR_BOTTOM = (
    "/home/ming/malware/inst2vec_bert/H-Transformer/process_data"
)
ORIGINAL_DATA_BASE_FOR_MIDDLE = "/home/ming/malware/prog_32bit/coreutils-asm"
CURRENT_DATA_BASE_FOR_MIDDLE = (
    "/home/ming/malware/inst2vec_bert/H-Transformer/data/middle"
)


def read_file(filename):
    print("Reading data from {}...".format(filename))
    with open(filename, "r", encoding="utf-8") as fin:
        return fin.readlines()


def write_file(sents, filename):
    print("Writing data to {}...".format(filename))
    with open(filename, "w", encoding="utf-8") as fout:
        for sent in sents:
            fout.write(sent)


def get_one_function_all_blocks(function_path, is_pkl=False):
    if is_pkl:
        raise NotImplementedError
        return None
    else:
        with open(function_path) as f:
            fn = Function.load(f.read())
        # for index, block in enumerate(fn.blocks):
        #     with open("block.{}".format(index), "w", encoding="utf-8") as fout:
        #         for inst in block.insts:
        #             fout.write(" ".join(inst.fg_tokens()) + "\n")
        # print(
        #     [
        #         len([" ".join(inst.tokens()) for inst in block.insts])
        #         for block in fn.blocks
        #     ]
        # )
        return [
            [" ".join(inst.tokens()) for inst in block.insts] for block in fn.blocks
        ]


# 得到一个function的所有指令
# 每个指令包含多个fg_token(Fine Grained Token)
# 返回值为二维数组，大小为(`insts_num`, `fg_tokens_num`)
# [['push', 'ebp', '<space>'], ['push', 'edi', '<space>'], ['mov', 'dword', '[', 'esp', '+', '4', ']', 'eax']]
def get_one_function_all_fg_tokens(function_path, is_pkl=False):
    # is_pkl 暂时不能用，默认False就可以了
    if is_pkl:
        fn = Function()
        with open(function_path, "rb") as fr:
            fn = pickle.load(fr)
        return fn.fg_tokens()
    else:
        with open(function_path) as f:
            fn = Function.load(f.read())
        return fn.fg_tokens()


# 得到一个function种可能出现的`walk_num`个顺序序列，即随机游走
# 每个顺序序列为一串顺序执行的指令
# 每个指令拆分为多个fg_token(Fine Grained Token)
# 返回值为三维数组，大小为(`walk_num`, `insts_num`, `fg_tokens_num`)，这里好像写得不对，直接看输出吧hhhh
def get_one_function_random_walk_fg_tokens(function_path, is_pkl=False, walk_num=3):
    fn = None
    # is_pkl 暂时不能用，默认False就可以了
    if is_pkl:
        fn = Function()
        with open(function_path, "rb") as fr:
            fn = pickle.load(fr)
    else:
        with open(function_path) as f:
            fn = Function.load(f.read())

    walks = fn.random_walk(walk_num)
    walk_fg_tokens = []
    for walk in walks:
        fg_tokens = []
        insts = walk
        for inst in insts:
            fg_tokens.append([inst.op] + inst.fg_args)
        walk_fg_tokens.append(fg_tokens)
    return walk_fg_tokens


# def get_one_function_all_insts(function_path, is_pkl=False):
#     if is_pkl:
#         fn = Function()
#         with open(function_path, 'rb') as fr:
#             fn = pickle.load(fr)
#         return fn.tokens()
#     else:
#         with open(function_path) as f:
#             fn = Function.load(f.read())
#         return fn.tokens()

# def get_one_function_random_walk_insts(function_path, is_pkl=False, num=3):
#     fn = None
#     if is_pkl:
#         fn = Function()
#         with open(function_path, 'rb') as fr:
#             fn = pickle.load(fr)
#     else:
#         with open(function_path) as f:
#             fn = Function.load(f.read())

#     walks = fn.random_walk()
#     walk_insts = []
#     for walk in walks:
#         tokens = []
#         insts = walk
#         for inst in insts:
#             tokens.append([inst.op] + inst.args)
#         walk_insts.append(tokens)
#     return walk_insts

if __name__ == "__main__":
    get_one_function_all_blocks(
        "/home/ming/malware/data/elfasm/linux32_00xxxx/linux32_b_000001/fdd6ebaf79a90c237c413551dad0a578f59348c1ba397ba9bbb54cb3f9f65767"
    )
