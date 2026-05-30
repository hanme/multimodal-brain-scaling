import argparse


def str2bool(v):
    """
    Converts a string representation of a boolean value to its corresponding boolean value.

    Args:
        v (str): The string representation of the boolean value.

    Returns:
        bool: The corresponding boolean value.
    """
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    elif v in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
