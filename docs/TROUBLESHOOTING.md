# Troubleshooting

## The App is blocked by Gatekeeper

The published `v1.0` App is ad-hoc signed, not Developer ID signed, and not
notarized. On a downloaded copy, first try opening it once, then use
**System Settings → Privacy & Security → Open Anyway**. Verify the release
checksum before bypassing the warning. A future public binary release should add
Developer ID signing and notarization; do not commit certificates, private keys,
or provisioning profiles to the repository.

## Camera permission was denied

Open **System Settings → Privacy & Security → Camera**, enable LLVNR Demo, then
restart the App. The camera path is not available in a headless session. A
paced replay from a local directory can be used without camera permission.

## `ipsw` is missing

Install the external `ipsw` CLI and ensure it is on `PATH`, or pass its path:

```bash
python3 demo/kit/extract_ipsw_assets.py /path/to/local.aea \
  --ipsw /path/to/ipsw \
  --variant V63 \
  --output /path/to/private/extracted
```

The extractor does not download firmware. Use a local file you are authorized
to process.

## Extraction is incomplete

The V63 path expects 12 `LLV-HD-*.bundle` models and 6
`LLVTuningParameters.plist` files. Check the counts and `EXTRACT-MANIFEST.json`.
`--allow-partial` is for discovery only; it is not a complete release package.

## Catalog generation fails

- `no tuning plist maps to model ...`: the plist `ModelInfo.Name` does not match
  the model bundle name. Check the extraction variant and plist.
- `unsupported LLVNR model ID`: the bundle name is not one of the supported
  canvas/model patterns.
- `model has no passing exported runtime assets`: run the public exporter with
  `--validation ane` and inspect `export_validation.json`.
- `asset source model hash mismatch`: the catalog and asset tree came from
  different `model.hwx` files.

## Kit verification fails

The verifier rejects:

- path traversal, absolute paths, backslashes, duplicate ZIP members, symlinks,
  encrypted ZIP members, and oversized archives;
- raw `.hwx`, `.net`, `.weights`, `.shape`, or `.espressif` files;
- files not listed in `kit.json`;
- catalog/model identity, manifest, source hash, or runtime file mismatches.

Rebuild the kit instead of editing `kit.json` or `manifest.json` by hand.

## ANE compilation or loading fails

Check Apple Silicon hardware, macOS version, the model kit's target assumptions,
and whether another experiment holds the local ANE lock. The App does not
silently switch to CPU or Metal. New kits carry `target_family` / `target_arch`;
a mismatch means the kit was exported for another ANE family and must be
re-exported and revalidated on this Mac. H13 is the only locally verified
generation; see [ANE support](ANE-SUPPORT.md).

## The canvas or model does not match

The runtime expects a supported padded canvas (`2304x1296` or `2464x1388`) and
the four named inputs `cur_frame`, `his_frame`, `dnr_zoom_map`, and
`fusion_zoom_map`. A manifest with a different canvas or port set is rejected.

## Report contains a hostname or local path

Newer builds omit those fields by default. For an older report, use
`tools/sanitize_run_report.py` as described in [Privacy](../PRIVACY.md). Do not
publish raw `frames.jsonl`; it can contain timing and local execution details.

## Swift build fails

Use macOS 14+ with Swift 6 and Apple Silicon:

```bash
cd demo
swift build -c release --product LLVNRDemo
```

The build intentionally emits warnings for existing concurrency and deprecated
API uses. A warning is not a successful runtime acceptance test.
