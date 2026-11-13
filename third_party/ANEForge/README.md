# ANEForge vendored subset

This directory contains the minimal MIT-licensed ANEForge source needed by the
public LLVNR exporter and runtime. It is vendored from ANEForge commit
`eba339e590d64c01e222dc72ea2e0c9e8988a4b8` so a clean checkout does not need a
private research tree or a separately installed Python package.

Included:

- graph lowering and MIL/BLOBFILE emission (`_compile.py`, `_blob.py`, `graph.py`);
- the ctypes E5RT runtime (`_runtime.py`);
- the source-only dispatcher build path (`build.py`, `_lib/`);
- the small support modules imported by those files.

The dispatcher dylib is intentionally not committed. On Apple Silicon macOS it
is compiled locally from `aneforge/_lib/ane_e5rt_dispatch.mm` on first use, or
explicitly with:

```sh
sh third_party/ANEForge/aneforge/_lib/build.sh
```

The upstream MIT license is in `LICENSE`; attribution is in `NOTICE`.
