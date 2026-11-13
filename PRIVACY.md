# Privacy and local data

LLVNR Demo is a local research tool. The public repository does not contain
model weights, tuning data, firmware, test photos, videos, certificates,
provisioning profiles, private keys, or generated `.llvnrkit` files. The App
does not upload model assets, camera frames, video frames, reports, or hashes to
a network service.

## What stays on the Mac

| Data | Default location / behavior |
| --- | --- |
| Imported model kit | `~/Library/Application Support/LLVNRDemo/CurrentModelKit` |
| Raw run reports | `~/Library/Application Support/LLVNRDemo/Runs/<timestamp>/` |
| Per-frame JSONL | `frames.jsonl` inside the run directory |
| Run summary | `summary.json` inside the run directory |
| Compiled ANE cache | `~/Library/Caches/com.llvnr.mac.demo/<manifest-hash>` |
| UI preferences | `UserDefaults` for bundle `com.llvnr.mac.demo` |
| Camera / video input | processed in memory; exported only when the user chooses an output path |

A model kit is private model-derived material. Do not commit it, attach it to a
public issue, or upload it unless you have separately confirmed the rights to
the underlying material.

## Sharing a report

Newer builds exclude the hostname and local run directory from
`summary.json` by default. Older builds or error messages may still contain
local paths. Use the local sanitizer before sharing anything:

```bash
python3 tools/sanitize_run_report.py \
  "$HOME/Library/Application Support/LLVNRDemo/Runs/<timestamp>" \
  --output /tmp/llvnr-report-shareable
```

The sanitizer removes host/user/path-like fields and omits raw per-frame rows
unless `--include-frames` is explicitly supplied. It performs no upload.

To retain host and local paths for private debugging, launch the App with:

```bash
LLVNR_INCLUDE_LOCAL_DIAGNOSTICS=1 open -n "LLVNR Demo.app"
```

Do not use that mode for reports you intend to publish.

## Cleanup / uninstall

```bash
rm -rf "$HOME/Library/Application Support/LLVNRDemo"
rm -rf "$HOME/Library/Caches/com.llvnr.mac.demo"
defaults delete com.llvnr.mac.demo 2>/dev/null || true
```

Deleting these paths removes the imported kit, run reports, compiled ANE cache,
and UI preferences. It does not delete any original firmware or local source
material you extracted elsewhere.

## Repository hygiene

Before publishing a source snapshot or release, run:

```bash
python3 tools/audit_release.py .
python3 tools/check_readme_links.py .
```

The audit rejects model suffixes, media, certificates, private keys, local
paths, and generated model kits. Passing the audit does not grant rights to
third-party model or tuning data.
