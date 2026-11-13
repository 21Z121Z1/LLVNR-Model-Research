# App precision runtime source

`PrecisionRuntime.mm` is the Objective-C++ source included by the demo's
`Sources/ANERuntime/PrecisionRuntime.mm`. Keep it at this path so the source
tree's relative include remains unchanged.

This directory intentionally does not contain a prebuilt dylib, model weights,
or firmware assets. The research Python bridge expects a locally built
`libllvnr_precision.dylib` only when that optional path is exercised.
