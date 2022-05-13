from h5py import File, Group
from rich import print

from .typing import *


@typechecked
def log(self, msg: str):
    """Log a formatted message to stdout."""
    print(f"[white]:: [magenta]{self.__class__.__name__}[/magenta]: [green]{msg}[/green][/white]")


@typechecked
def pack(file: File, path: str, obj: Any):
    """Save an object to a given path in an open HDF file."""
    if type(obj) == Sparse:
        file[path + "/data"] = obj.data
        file[path + "/indices"] = obj.indices
        file[path + "/indptr"] = obj.indptr
    else:
        file[path] = obj


@typechecked
def unpack(file: File, path: str) -> Any:
    """Load an object at a given path in an open HDF file."""
    if isinstance(file[path], Group):
        data = file[path + "/data"][...]
        indices = file[path + "/indices"][...]
        indptr = file[path + "/indptr"][...]

        return Sparse((data, indices, indptr))
    else:
        return file[path][...]
