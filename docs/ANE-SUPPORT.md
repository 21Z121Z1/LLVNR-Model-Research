# ANE generation support

This page records what has actually been executed. A successful compile call is
not evidence that another ANE generation is supported.

## Current evidence matrix

| Target | Evidence in this repository | Status |
| --- | --- | --- |
| H13 / M1-class ANE | Direct MIL compilation and execution through the vendored ANEForge runtime; CPU/ANE synthetic comparison passed for all 12 locally recovered V63 graphs during the 2026-11-13 work. Whole-frame CPU versus native task CPU was exact for Wide 2x 2304. | Verified on the local M1 Pro for the tested graphs; not a universal hardware guarantee. |
| H14-H18 | No target hardware execution recorded. Compile-only checks may return success while producing host-target bytes. | Unverified; do not claim support. |
| H19 | The source model is H19 firmware material, but no H19 target execution or numeric parity run is available. | Source-only; do not claim support. |

The App currently compiles its runtime graph with the host ANE device mask
(`0x4`) and does not automatically fall back to CPU or Metal if compilation or
loading fails. A failure is surfaced as a model runtime error rather than a
silent backend switch. Newly exported manifests record the host compiler family
and target architecture (`target_family` / `target_arch`); the App rejects a
manifest whose recorded target does not match the host. ANE export also fails
closed when the host M-series generation cannot be identified. Older kits
without these fields remain importable for legacy testing, but that
compatibility is not evidence of cross-generation support.

## Why `TargetArchitecture` success is not enough

The local experiments exercised `ane_e5rt_compile_check` with
`TargetArchitecture=h13` through `h19`. The compiler returned `rc=0` for every
target, but the generated `.e5` payloads were identical. That is consistent
with a host-target fallback and is not proof that H14-H19 code was produced or
can execute on those chips.

The App therefore does not expose a "force another ANE generation" switch. A
future support claim requires, at minimum:

1. compilation for an explicit target with a verifiable target fingerprint;
2. load and execution on real target hardware;
3. CPU/reference numeric comparison on fixed inputs;
4. camera/video end-to-end output comparison;
5. a documented failure path when the target is unavailable.

Cross-generation numerical differences may be legitimate FP16 differences, but
they still need a measured acceptance threshold. CI on a host Mac is not a
substitute for target hardware.

## Safe expectation for users

- Use H13/M1-class hardware only for the currently verified path.
- Treat H14-H19 as experimental and unsupported until the matrix above gains
  real hardware evidence.
- If the App reports an ANE compile/load failure, do not infer that the model
  is portable to another chip. Use the private report and the troubleshooting
  guide.
