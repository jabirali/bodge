from h5py import File, Group

from .typing import *


def pack(file: File, path: str, obj: Any):
    """Save an object to a given path in an open HDF file."""
    if type(obj) == Sparse:
        file[path + "/data"] = obj.data
        file[path + "/indices"] = obj.indices
        file[path + "/indptr"] = obj.indptr
    else:
        file[path] = obj


def unpack(file: File, path: str):
    """Load an object at a given path in an open HDF file."""
    if isinstance(file[path], Group):
        data = file[path + "/data"][...]
        indices = file[path + "/indices"][...]
        indptr = file[path + "/indptr"][...]

        return Sparse((data, indices, indptr))
    else:
        return file[path][...]
