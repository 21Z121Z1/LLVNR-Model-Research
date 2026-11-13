# LLVNR Demo (weightless)

This directory contains the model-free macOS demo source. It intentionally has no `Resources/Models`, `Resources/Catalog`, tuning, `CameraSetup.plist`, or generated `.llvnrkit`.

For the user workflow, local kit format, import behavior and signing limitations, read [WEIGHTLESS_DISTRIBUTION.md](WEIGHTLESS_DISTRIBUTION.md).

Build dependencies:

- macOS 14+
- Xcode / Swift 6
- Apple Silicon for the ANE/Metal runtime
- Python 3 for the kit tools

The release ZIP is not stored in Git. It is attached to the repository's `v1.0` Release and must be verified against its SHA-256 before use.
