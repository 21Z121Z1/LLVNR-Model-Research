import Foundation
import Metal
import CryptoKit
import ANERuntime

final class ModelRuntime {
    let manifest: ModelManifest
    let device: MTLDevice
    private var handle: UnsafeMutableRawPointer?
    private var lease: Int32 = -1
    let inputs: [String: MTLBuffer]
    let outputs: [String: MTLBuffer]

    init(device: MTLDevice,modelID: String?=nil) throws {
        self.device=device
        let catalog=try ModelCatalog.load()
        let selected=try catalog.entry(modelID)
        let (directory,metadata)=try catalog.runtimeManifest(selected)
        manifest=metadata
        let expectedInputs:Set<String>=["cur_frame","his_frame","dnr_zoom_map","fusion_zoom_map"]
        guard manifest.version==2,manifest.node_count>0,manifest.coefficient_group_count>0,
              manifest.width==1920,manifest.height==1080,
              Set(manifest.inputs.map(\.name))==expectedInputs,
              Set(manifest.outputs.map(\.name)).isSuperset(of:["out_yuv","state_yuv"]),
              manifest.inputs.allSatisfy({$0.shape.count==4 && $0.shape.allSatisfy{$0>0} && $0.bytes==2*$0.shape.reduce(1,*)}),
              manifest.outputs.allSatisfy({$0.shape.count==4 && $0.shape.allSatisfy{$0>0} && $0.bytes==2*$0.shape.reduce(1,*)}) else {
            throw PipelineError.failure("模型输入输出清单无效")
        }
        if (manifest.target_family == nil) != (manifest.target_arch == nil) {
            throw PipelineError.failure("模型 ANE 目标字段不完整")
        }
        if let target = manifest.target_family {
            let host = llvnr_host_ane_family()
            guard host >= 2, target == host else {
                let expected = manifest.target_arch ?? "family \(target)"
                throw PipelineError.failure("模型目标 ANE \(expected) 与当前主机不匹配（host family \(host)）；请在本机重新导出并验证")
            }
        }
        for (name,hash) in manifest.files {
            guard !name.hasPrefix("/"),!name.split(separator:"/").contains("..") else {throw PipelineError.failure("模型资产路径无效")}
            let data=try Data(contentsOf:directory.appendingPathComponent(name))
            let actual=SHA256.hash(data:data).map {String(format:"%02x",$0)}.joined()
            guard actual==hash else {throw PipelineError.failure("模型校验失败：\(name)")}
        }
        lease=llvnr_acquire_lock()
        guard lease>=0 else {throw PipelineError.failure("ANE / 相机实验锁正被其他进程持有")}
        let key=selected.runtime_manifest_sha256!
        let cache=FileManager.default.urls(for:.cachesDirectory,in:.userDomainMask)[0]
            .appendingPathComponent("com.llvnr.mac.demo/\(key)")
        do {try FileManager.default.createDirectory(at:cache,withIntermediateDirectories:true)}
        catch {llvnr_release_lock(lease);lease = -1;throw error}
        handle=llvnr_compile(directory.path,cache.path)
        guard let model=handle else {
            llvnr_release_lock(lease);lease = -1
            throw PipelineError.failure("模型无法在这台 Mac 上编译或加载，请检查系统兼容范围")
        }
        func wrap(_ spec: ModelManifest.Port, input: Bool) throws -> MTLBuffer {
            var count=0
            let pointer=input ? llvnr_input(model,spec.port,&count) : llvnr_output(model,spec.port,&count)
            let page=Int(getpagesize()), length=(spec.bytes+page-1) & ~(page-1)
            guard let pointer=pointer, count==spec.bytes,
                  Int(bitPattern:pointer)%page==0,
                  let buffer=device.makeBuffer(bytesNoCopy:pointer,length:length,options:.storageModeShared,deallocator:nil) else {
                throw PipelineError.failure("\(spec.name) 缓冲无法安全映射给 Metal")
            }
            buffer.label=spec.name
            return buffer
        }
        do {
            inputs=try Dictionary(uniqueKeysWithValues:manifest.inputs.map{($0.name,try wrap($0,input:true))})
            outputs=try Dictionary(uniqueKeysWithValues:manifest.outputs.map{($0.name,try wrap($0,input:false))})
        } catch {
            llvnr_release(model);handle=nil;llvnr_release_lock(lease);lease = -1;throw error
        }
    }
    func execute() throws {
        guard let handle=handle, llvnr_execute(handle)==0 else {throw PipelineError.failure("ANE 推理执行失败")}
    }
    func warmup() throws {
        for (_,buffer) in inputs {memset(buffer.contents(),0,buffer.length)}
        for key in ["dnr_zoom_map","fusion_zoom_map"] {
            let n=inputs[key]!.length/MemoryLayout<Float16>.size
            inputs[key]!.contents().bindMemory(to:Float16.self,capacity:n).initialize(repeating:1,count:n)
        }
        for _ in 0..<10 {try execute()}
    }
    deinit {
        if let model=handle {llvnr_release(model)}
        llvnr_release_lock(lease)
    }
}
