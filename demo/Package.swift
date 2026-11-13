// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LLVNRDemo",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "LLVNRDemo", targets: ["LLVNRDemo"])],
    targets: [
        .target(name: "ANERuntime", path: "Sources/ANERuntime",
                exclude: ["LICENSE-ANEForge"], publicHeadersPath: "include",
                cxxSettings: [.unsafeFlags(["-std=c++17", "-fblocks"])],
                linkerSettings: [.linkedFramework("Foundation"), .linkedFramework("Metal")]),
        .executableTarget(name: "LLVNRDemo", dependencies: ["ANERuntime"],
            resources: [.copy("Resources")],
            swiftSettings: [.swiftLanguageMode(.v5)],
            linkerSettings: [.linkedFramework("AVFoundation"), .linkedFramework("MetalKit"), .linkedFramework("CoreVideo")])
    ]
)
