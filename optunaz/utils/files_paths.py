import os


def move_up_directory(path, n=1):
    """Function, to move up "n" directories for a given "path"."""
    # add +1 to take file into account
    if os.path.isfile(path):
        n += 1
    for _ in range(n):
        path = os.path.dirname(os.path.abspath(path))
    return path


def attach_root_path(path):
    """Function to attach the root path of the module for a given "path"."""
    ROOT_DIR = move_up_directory(os.path.abspath(__file__), n=2)
    return os.path.join(ROOT_DIR, path)
