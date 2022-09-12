from h5py import File, Group
from rich import print

from .typing import *


@typecheck
def log(self, msg: str):
    """Log a formatted message to stdout."""
    if isinstance(self, str):
        name = f"[red]{self}[/red]"
    else:
        name = f"[magenta]{self.__class__.__name__}[/magenta]"
    print(f"[white]:: {name}: [green]{msg}[/green][/white]")


@typecheck
def pack(file: File, path: str, obj: Any):
    """Save an object to a given path in an open HDF file."""
    if type(obj) == bsr_matrix:
        file[path + "/data"] = obj.data
        file[path + "/indices"] = obj.indices
        file[path + "/indptr"] = obj.indptr
    else:
        file[path] = obj


@typecheck
def unpack(file: File, path: str) -> Any:
    """Load an object at a given path in an open HDF file."""
    if isinstance(file[path], Group):
        data = file[path + "/data"][...]
        indices = file[path + "/indices"][...]
        indptr = file[path + "/indptr"][...]

        return bsr_matrix((data, indices, indptr))
    else:
        return file[path][...]
