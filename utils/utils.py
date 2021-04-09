import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def dir_parser(root, dir_name):
    dir_root = os.path.join(root, dir_name)
    if not os.path.exists(dir_root):
        os.makedirs(dir_root)
    return dir_root
