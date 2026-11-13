import Foundation
import CryptoKit

struct ModelCatalog: Decodable {
    struct FileRecord: Decodable { let sha256: String; let bytes: Int }
    struct Entry: Decodable, Identifiable {
        let id: String, name: String, title: String, family: String, zoom: String, model_version: String
        let width: Int, height: Int, bytes: Int, sha256: String, original_file: String
        let task_count: Int, ne_groups: Int, lut_groups: Int, output_names: [String], preset_ids: [String]
        let runtime_directory: String?, status: String, status_detail: String
        let runtime_manifest_sha256: String?
        var canRun: Bool { status == "ported_local" && runtime_directory != nil }
    }
    struct Preset: Decodable, Identifiable {
        let id: String, name: String, model_name: String, file: String
        let input_modulation: [String: [Double]], output_modulation: [String: [Double]]
    }
    let version: Int, platform: String, firmware_build: String, default_model_id: String
    let models: [Entry], presets: [Preset], files: [String: FileRecord]

    static var resourceRoot: URL? {
        if let installed = ModelStore.installedRootIfPresent { return installed }
        return Bundle.module.url(forResource:"Resources",withExtension:nil)
    }
    static var directory: URL? { resourceRoot?.appendingPathComponent("Catalog") }
    static func load() throws -> ModelCatalog {
        guard let root=resourceRoot else {throw PipelineError.failure("缺少模型目录")}
        return try load(from: root)
    }
    static func load(from root: URL) throws -> ModelCatalog {
        let directory=root.appendingPathComponent("Catalog")
        let result=try JSONDecoder().decode(Self.self,from:Data(contentsOf:directory.appendingPathComponent("catalog.json")))
        guard result.version==2, !result.models.isEmpty,
              Set(result.models.map(\.id)).count==result.models.count,
              result.models.contains(where:{$0.id==result.default_model_id && $0.canRun}),
              result.models.allSatisfy({entry in
                  !entry.preset_ids.isEmpty && entry.preset_ids.allSatisfy { id in
                      result.presets.contains { $0.id==id && $0.model_name==entry.name }
                  }
              }) else {throw PipelineError.failure("模型目录或 tuning 对应关系不完整")}
        return result
    }
    func entry(_ id: String?) throws -> Entry {
        guard let entry=models.first(where:{$0.id==(id ?? default_model_id)}) else {
            throw PipelineError.failure("找不到指定模型")
        }
        return entry
    }
    func verifyFile(_ name: String, root: URL? = nil) throws {
        let base = (root ?? Self.resourceRoot)?.appendingPathComponent("Catalog")
        guard let directory=base,let record=files[name],
              !name.split(separator:"/").contains(".."),!name.hasPrefix("/") else {
            throw PipelineError.failure("模型文件清单无效")
        }
        let data=try Data(contentsOf:directory.appendingPathComponent(name))
        let hash=SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()
        guard data.count==record.bytes,hash==record.sha256 else {throw PipelineError.failure("目录资产校验失败：\(name)")}
    }
    func verifyAll(root: URL? = nil) throws {
        for name in files.keys.sorted() {try verifyFile(name,root:root)}
        for entry in models {_=try runtimeManifest(entry,root:root)}
    }
    func runtimeManifest(_ entry: Entry, root: URL? = nil) throws -> (URL,ModelManifest) {
        guard entry.canRun,let resource=entry.runtime_directory,
              let expected=entry.runtime_manifest_sha256,
              let root=root ?? Self.resourceRoot else {
            throw PipelineError.failure("缺少所选模型的运行资产")
        }
        let url=root.appendingPathComponent(resource)
        guard FileManager.default.fileExists(atPath:url.path) else {
            throw PipelineError.failure("缺少所选模型的运行资产")
        }
        let data=try Data(contentsOf:url.appendingPathComponent("manifest.json"))
        let digest=SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()
        guard digest==expected else {throw PipelineError.failure("模型清单校验失败")}
        let manifest=try JSONDecoder().decode(ModelManifest.self,from:data)
        guard manifest.source_model_sha256==entry.sha256,
              manifest.padded_width==entry.width,manifest.padded_height==entry.height else {
            throw PipelineError.failure("模型身份或画布尺寸不符")
        }
        return (url,manifest)
    }
    func validationReport() throws -> [String:Any] {
        try verifyAll()
        let endpoint=try Self.interpolate([40,0.25,60,0.15,300,0.1],at:40)
        let middle=try Self.interpolate([40,0.25,60,0.15,300,0.1],at:50)
        guard abs(endpoint-0.25)<1e-12,abs(middle-0.2)<1e-12,
              try Self.interpolate([0,0.5],at:100)==0.5 else {
            throw PipelineError.failure("tuning LUT 插值检查失败")
        }
        for preset in presets {
            _=try presetText(preset)
            for table in Array(preset.input_modulation.values)+Array(preset.output_modulation.values) {
                _=try Self.interpolate(table,at:100)
            }
        }
        return ["passed":true,"models":models.count,"ultrawide_models":models.filter{$0.family=="SuperWide"}.count,
                "presets":presets.count,"files_verified":files.count,"lut_interpolation_verified":true,
                "runnable_models":models.filter(\.canRun).map(\.id),
                "unported_models":models.filter{!$0.canRun}.map(\.id)]
    }
    func presetText(_ preset: Preset, root: URL? = nil) throws -> String {
        try verifyFile(preset.file,root:root)
        guard let directory=((root ?? Self.resourceRoot)?.appendingPathComponent("Catalog")) else {
            throw PipelineError.failure("缺少 tuning 目录")
        }
        let data=try Data(contentsOf:directory.appendingPathComponent(preset.file))
        let plist=try PropertyListSerialization.propertyList(from:data,format:nil)
        let json=try JSONSerialization.data(withJSONObject:plist,options:[.prettyPrinted,.sortedKeys])
        return String(decoding:json,as:UTF8.self)
    }
    /// Firmware LUTs contain interleaved x,y pairs. Clamp outside the endpoints.
    static func interpolate(_ lut: [Double], at x: Double) throws -> Double {
        guard x.isFinite,lut.count>=2,lut.count%2==0,lut.allSatisfy(\.isFinite) else {
            throw PipelineError.failure("tuning LUT 数据不完整")
        }
        let pairs=stride(from:0,to:lut.count,by:2).map{(lut[$0],lut[$0+1])}
        guard zip(pairs,pairs.dropFirst()).allSatisfy({$0.0.0<$0.1.0}) else {
            throw PipelineError.failure("tuning LUT 横坐标必须递增")
        }
        if x<=pairs[0].0 {return pairs[0].1}
        for (a,b) in zip(pairs,pairs.dropFirst()) where x<=b.0 {
            return a.1+(b.1-a.1)*(x-a.0)/(b.0-a.0)
        }
        return pairs.last!.1
    }
}
