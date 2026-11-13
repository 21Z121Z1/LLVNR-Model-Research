"""ctypes bridge for the local Metal prefix implementation."""
import ctypes
import json
from pathlib import Path

import numpy as np


class MetalPrefix:
    def __init__(self,folder,library=None):
        self.folder=Path(folder)
        library=library or Path(__file__).resolve().parents[1]/"common/libllvnr_precision.dylib"
        self.lib=ctypes.CDLL(str(library))
        self.lib.llvnr_precision_create.argtypes=[ctypes.c_char_p];self.lib.llvnr_precision_create.restype=ctypes.c_void_p
        self.lib.llvnr_precision_buffer.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_size_t)]
        self.lib.llvnr_precision_buffer.restype=ctypes.c_void_p
        self.lib.llvnr_precision_execute.argtypes=[ctypes.c_void_p];self.lib.llvnr_precision_execute.restype=ctypes.c_int
        self.lib.llvnr_precision_release.argtypes=[ctypes.c_void_p]
        self.lib.llvnr_precision_error.restype=ctypes.c_char_p
        self.handle=self.lib.llvnr_precision_create(str(self.folder).encode())
        if not self.handle:raise RuntimeError(self.lib.llvnr_precision_error().decode())
        self.plan=json.loads((self.folder/"prefix.json").read_text())
        self.shapes={n["name"]:n["shape"] for n in self.plan["nodes"]}
    def buffer(self,name):
        size=ctypes.c_size_t()
        pointer=self.lib.llvnr_precision_buffer(self.handle,name.encode(),ctypes.byref(size))
        if not pointer:raise KeyError(name)
        return np.ctypeslib.as_array((ctypes.c_uint16*(size.value//2)).from_address(pointer)).view(np.float16).reshape(self.shapes[name])
    def run(self,inputs,capture=False):
        for n in self.plan["nodes"]:
            if n["input"]:np.copyto(self.buffer(n["name"]),inputs[n["name"]])
        if self.lib.llvnr_precision_execute(self.handle):raise RuntimeError(self.lib.llvnr_precision_error().decode())
        names=self.shapes if capture else self.plan["outputs"]
        return {name:self.buffer(name).copy() for name in names}
    def close(self):
        if self.handle:self.lib.llvnr_precision_release(self.handle);self.handle=None
