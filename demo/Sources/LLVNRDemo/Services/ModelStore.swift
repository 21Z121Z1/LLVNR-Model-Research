import Foundation
import CryptoKit

/// Installs user-produced model kits outside the application bundle. The
/// public app can therefore ship without firmware-derived model assets.
final class ModelStore {
    private struct KitFile: Decodable { let sha256: String; let bytes: Int }
    private struct KitModel: Decodable {
        let manifest_sha256: String
        let source_model_sha256: String
    }
    private struct KitRecord: Decodable {
        let schema_version: Int
        let kit_id: String
        let model_ids: [String]
        let catalog_sha256: String
        let total_bytes: Int
        let files: [String: KitFile]
        let models: [String: KitModel]
    }

    static let appSupportRoot: URL = {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LLVNRDemo", isDirectory: true)
    }()
    static let installedRoot = appSupportRoot.appendingPathComponent("CurrentModelKit", isDirectory: true)

    static var installedRootIfPresent: URL? {
        FileManager.default.fileExists(atPath: installedRoot.appendingPathComponent("kit.json").path)
            ? installedRoot : nil
    }

    /// Accepts either an unpacked `.llvnrkit` directory or the ZIP produced by
    /// `kit/build_kit.py`. Installation is transactional: an invalid kit never
    /// replaces the currently installed one.
    static func install(from source: URL) throws -> ModelCatalog {
        let manager = FileManager.default
        let source = source.standardizedFileURL
        guard source.isFileURL else { throw PipelineError.failure("模型资产包必须是本地目录或 ZIP") }
        if source.path == installedRoot.path || source.path.hasPrefix(installedRoot.path + "/") {
            throw PipelineError.failure("不能导入当前已安装的模型资产包")
        }

        let values = try source.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey])
        guard values.isSymbolicLink != true else {
            throw PipelineError.failure("模型资产包不能是符号链接")
        }
        let isDirectory = values.isDirectory == true
        let isArchive = values.isRegularFile == true && source.pathExtension.lowercased() == "zip"
        guard isDirectory || isArchive else {
            throw PipelineError.failure("请选择 .llvnrkit 文件夹或 .zip 压缩包")
        }

        try manager.createDirectory(at: appSupportRoot, withIntermediateDirectories: true)
        let container = appSupportRoot.appendingPathComponent("Installing-\(UUID().uuidString)", isDirectory: true)
        let stagedKit = container.appendingPathComponent("kit", isDirectory: true)
        try? manager.removeItem(at: container)
        try manager.createDirectory(at: container, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: container) }

        if isArchive {
            let extracted = container.appendingPathComponent("extracted", isDirectory: true)
            try extractZip(source, to: extracted)
            let root = try locateKitRoot(in: extracted)
            try manager.moveItem(at: root, to: stagedKit)
        } else {
            try manager.copyItem(at: source, to: stagedKit)
        }

        let kit = try verifyKit(at: stagedKit)
        let queuedCatalog = try ModelCatalog.load(from: stagedKit)
        guard Set(queuedCatalog.models.map(\.id)) == Set(kit.model_ids) else {
            throw PipelineError.failure("模型资产包目录与 kit 清单不一致")
        }
        try queuedCatalog.verifyAll(root: stagedKit)

        let backup = appSupportRoot.appendingPathComponent("PreviousModelKit-\(UUID().uuidString)", isDirectory: true)
        if manager.fileExists(atPath: installedRoot.path) {
            try manager.moveItem(at: installedRoot, to: backup)
        }
        do {
            try manager.moveItem(at: stagedKit, to: installedRoot)
        } catch {
            if manager.fileExists(atPath: backup.path) {
                try? manager.moveItem(at: backup, to: installedRoot)
            }
            throw error
        }
        try? manager.removeItem(at: backup)
        return try ModelCatalog.load()
    }

    static func removeInstalled() throws {
        let manager = FileManager.default
        if manager.fileExists(atPath: installedRoot.path) {
            try manager.removeItem(at: installedRoot)
        }
    }

    @discardableResult
    private static func verifyKit(at root: URL) throws -> KitRecord {
        guard root.standardizedFileURL.path != "/" else {
            throw PipelineError.failure("拒绝从文件系统根目录安装模型资产包")
        }
        try rejectRawFirmware(in: root)
        let actualFiles = try fileInventory(in: root)
        let kitURL = root.appendingPathComponent("kit.json")
        let kit: KitRecord = try JSONDecoder().decode(KitRecord.self, from: Data(contentsOf: kitURL))
        guard kit.schema_version == 1,
              kit.kit_id.hasPrefix("llvnr-"),
              !kit.model_ids.isEmpty,
              Set(kit.model_ids).count == kit.model_ids.count,
              !kit.files.isEmpty,
              kit.total_bytes >= 0,
              kit.files.count == actualFiles.count - 1,
              Set(kit.files.keys) == Set(actualFiles.keys).subtracting(["kit.json"]),
              Set(kit.models.keys) == Set(kit.model_ids) else {
            throw PipelineError.failure("模型资产包版本或文件清单无效")
        }

        for (relative, record) in kit.files {
            guard record.bytes >= 0, let actual = actualFiles[relative],
                  actual.bytes == record.bytes, actual.sha256 == record.sha256 else {
                throw PipelineError.failure("模型资产包文件校验失败：\(relative)")
            }
        }
        let listedBytes = actualFiles
            .filter { $0.key != "kit.json" }
            .values.reduce(0, { $0 + $1.bytes })
        guard kit.total_bytes == listedBytes else {
            throw PipelineError.failure("模型资产包字节总数不一致")
        }

        let catalogURL = root.appendingPathComponent("Catalog/catalog.json")
        guard digest(try Data(contentsOf: catalogURL)) == kit.catalog_sha256 else {
            throw PipelineError.failure("模型资产包 catalog 校验失败")
        }
        let catalog = try ModelCatalog.load(from: root)
        let catalogByID = Dictionary(uniqueKeysWithValues: catalog.models.map { ($0.id, $0) })
        guard Set(catalogByID.keys) == Set(kit.model_ids),
              catalogByID[catalog.default_model_id] != nil else {
            throw PipelineError.failure("模型资产包 catalog 模型清单不一致")
        }

        for modelID in kit.model_ids {
            let modelParts = modelID.split(separator: "/", omittingEmptySubsequences: false)
            guard modelParts.count == 1, modelParts[0] != ".", modelParts[0] != ".." else {
                throw PipelineError.failure("模型资产包包含无效模型 ID")
            }
            let manifestURL = root.appendingPathComponent("Models/\(modelID)/manifest.json")
            let manifestData = try Data(contentsOf: manifestURL)
            let manifest = try JSONDecoder().decode(ModelManifest.self, from: manifestData)
            guard let record = kit.models[modelID],
                  digest(manifestData) == record.manifest_sha256,
                  manifest.source_model_sha256 == record.source_model_sha256,
                  manifest.source_model_sha256 == catalogByID[modelID]?.sha256 else {
                throw PipelineError.failure("模型资产包 manifest 身份校验失败：\(modelID)")
            }
            let modelRoot = root.appendingPathComponent("Models/\(modelID)")
            var modelFiles = try fileInventory(in: modelRoot)
            modelFiles.removeValue(forKey: "manifest.json")
            guard Set(modelFiles.keys) == Set(manifest.files.keys) else {
                throw PipelineError.failure("模型资产包运行文件清单不一致：\(modelID)")
            }
            for (relative, expected) in manifest.files {
                _ = try safePath(relative, under: modelRoot)
                guard modelFiles[relative]?.sha256 == expected else {
                    throw PipelineError.failure("模型资产包运行文件校验失败：\(modelID)/\(relative)")
                }
            }
        }
        return kit
    }

    private struct FileRecord { let sha256: String; let bytes: Int }

    private static func fileInventory(in root: URL) throws -> [String: FileRecord] {
        let manager = FileManager.default
        guard let enumerator = manager.enumerator(at: root, includingPropertiesForKeys: [
            .isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey
        ], options: []) else {
            throw PipelineError.failure("无法读取模型资产包")
        }
        var result: [String: FileRecord] = [:]
        let rootPrefix = root.standardizedFileURL.path + "/"
        while let path = enumerator.nextObject() as? URL {
            let values = try path.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            if values.isSymbolicLink == true {
                throw PipelineError.failure("模型资产包不能包含符号链接")
            }
            guard values.isRegularFile == true else { continue }
            let standardized = path.standardizedFileURL.path
            guard standardized.hasPrefix(rootPrefix) else {
                throw PipelineError.failure("模型资产包包含越界路径")
            }
            let relative = String(standardized.dropFirst(rootPrefix.count))
            let data = try Data(contentsOf: path)
            result[relative] = FileRecord(sha256: digest(data), bytes: data.count)
        }
        return result
    }

    private static func safePath(_ relative: String, under root: URL) throws -> URL {
        let parts = relative.split(separator: "/", omittingEmptySubsequences: false)
        guard !relative.hasPrefix("/"), !relative.contains("\\"),
              !parts.isEmpty, !parts.contains(""), !parts.contains(".."), !parts.contains(".") else {
            throw PipelineError.failure("模型资产包包含不安全路径")
        }
        return root.appendingPathComponent(relative)
    }

    private static func rejectRawFirmware(in root: URL) throws {
        let bannedSuffixes = Set(["hwx", "net", "weights", "shape", "espressif"])
        let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey],
            options: []
        )
        while let path = enumerator?.nextObject() as? URL {
            let values = try path.resourceValues(forKeys: [.isSymbolicLinkKey, .isRegularFileKey])
            if values.isSymbolicLink == true {
                throw PipelineError.failure("模型资产包不能包含符号链接")
            }
            if values.isRegularFile == true, bannedSuffixes.contains(path.pathExtension.lowercased()) {
                throw PipelineError.failure("模型资产包不能包含原始固件文件：\(path.lastPathComponent)")
            }
        }
    }

    private static func extractZip(_ archive: URL, to destination: URL) throws {
        let listing = try runProcess("/usr/bin/unzip", ["-Z1", archive.path])
        guard let text = String(data: listing, encoding: .utf8) else {
            throw PipelineError.failure("无法读取 ZIP 文件清单")
        }
        for rawEntry in text.split(whereSeparator: \.isNewline) {
            let entry = String(rawEntry)
            let parts = entry.split(separator: "/", omittingEmptySubsequences: false)
            guard !entry.hasPrefix("/"), !entry.contains("\\"),
                  !parts.isEmpty, !parts.contains(""), !parts.contains(".."), !parts.contains(".") else {
                throw PipelineError.failure("ZIP 包含不安全路径：\(entry)")
            }
        }
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        _ = try runProcess("/usr/bin/ditto", ["-x", "-k", archive.path, destination.path])
    }

    private static func locateKitRoot(in extracted: URL) throws -> URL {
        let manager = FileManager.default
        if manager.fileExists(atPath: extracted.appendingPathComponent("kit.json").path) {
            return extracted
        }
        let children = try manager.contentsOfDirectory(
            at: extracted,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        let candidates = try children.filter { url in
            let values = try url.resourceValues(forKeys: [.isDirectoryKey])
            return values.isDirectory == true
                && manager.fileExists(atPath: url.appendingPathComponent("kit.json").path)
        }
        guard candidates.count == 1, let root = candidates.first else {
            throw PipelineError.failure("ZIP 必须包含一个模型资产包目录")
        }
        return root
    }

    @discardableResult
    private static func runProcess(_ executable: String, _ arguments: [String]) throws -> Data {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let output = Pipe()
        let error = Pipe()
        process.standardOutput = output
        process.standardError = error
        try process.run()
        let outputData = output.fileHandleForReading.readDataToEndOfFile()
        let errorData = error.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let detail = String(data: errorData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? "未知错误"
            throw PipelineError.failure("无法解压模型资产包：\(detail)")
        }
        return outputData
    }

    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
