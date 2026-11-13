# v1.0

## Release labels

- Repository release tag: `v1.0`
- App bundle: `0.4.0` build `6`
- Architecture: arm64
- Distribution: ad-hoc signed, not Developer ID signed, not notarized

`v1.0` is the research repository release label. It does not replace the App bundle version.

## Included asset

```text
LLVNR-Weightless-0.4.0-build6-arm64-test-20261113-v1.0.zip
LLVNR-Weightless-0.4.0-build6-arm64-test-20261113-v1.0.sha256
```

SHA-256 of the ZIP:

```text
29ac50b69a96ead04f893883b6bcc7e474cd62b187767c8c99088cc1cac247a5
```

The archive contains:

- `LLVNR Demo Weightless.app`
- local kit builder/verifier/schema
- `extract_ipsw_assets.py` local IPSW/OTA extractor
- `README.md`
- `LICENSE-ANEForge.txt`
- package verification report

It contains no LLVNR model, weight, tuning, `CameraSetup.plist`, generated `.llvnrkit`, photo, video, certificate, provisioning profile, or Apple firmware.

## What changed

- Model-free runtime and `.llvnrkit` import path.
- Directory and ZIP import.
- Transactional installation; a failed kit does not replace the current kit.
- App source and release audit remove `Resources/Models`, `Resources/Catalog`, tuning and firmware-derived files.
- Release scan rejects local paths/usernames, model suffixes and generated kits.
- Added `extract_ipsw_assets.py`, tested against the caller-provided local V63 OTA and verified 12 model.hwx plus 6 tuning files.
- Demo UI includes direct-angle video preview, centered framing, reduced bottom controls, retained backward seek cache and source display metadata for HDR comparison.

## How to use

1. Verify the SHA-256.
2. Unzip and move the weightless App to `/Applications`.
3. Generate a local kit from material you are authorized to use.
4. Run `python3 kit/verify_kit.py <kit-or-zip>`.
5. Import the kit through the App menu.

## Verification boundary

The archive was built and locally checked on Apple Silicon. Ad-hoc signing allows local execution but does not provide Apple notarization or a Developer ID trust chain. Other Macs may need Gatekeeper's “Open Anyway” flow. Model inference, camera behavior, HDR appearance and performance must be rechecked on the target machine.
