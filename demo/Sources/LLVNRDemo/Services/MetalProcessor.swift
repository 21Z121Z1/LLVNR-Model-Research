import Foundation
import MetalKit
import CoreVideo

final class MetalProcessor {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    private let packPipeline: MTLComputePipelineState, mapsPipeline: MTLComputePipelineState
    private let historyPipeline: MTLComputePipelineState, unpackPipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache!
    private let poolLock=NSLock()
    private var freeTextures: [OutputTextures] = []
    private let available=DispatchSemaphore(value:6)
    private var previewHistory: MTLBuffer?

    init(device: MTLDevice) throws {
        self.device=device
        guard let q=device.makeCommandQueue(),
              let sourceURL=Bundle.module.url(forResource:"Pipeline",withExtension:"metal",subdirectory:"Resources") else {throw PipelineError.failure("Metal 资源不可用")}
        queue=q
        let lib=try device.makeLibrary(source:String(contentsOf:sourceURL,encoding:.utf8),options:nil)
        library=lib
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f=lib.makeFunction(name:name) else {throw PipelineError.failure("缺少 Metal 内核 \(name)")}
            return try device.makeComputePipelineState(function:f)
        }
        packPipeline=try pipeline("pack420");mapsPipeline=try pipeline("fillMaps")
        historyPipeline=try pipeline("updateHistory");unpackPipeline=try pipeline("unpack420")
        guard CVMetalTextureCacheCreate(nil,nil,device,nil,&textureCache)==kCVReturnSuccess else {
            throw PipelineError.failure("CVMetalTextureCache 创建失败")
        }
        let descriptor=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.bgra8Unorm,width:1920,height:1080,mipmapped:false)
        descriptor.storageMode = .private;descriptor.usage=[.shaderRead,.shaderWrite]
        for _ in 0..<6 {
            guard let texture=device.makeTexture(descriptor:descriptor),let original=device.makeTexture(descriptor:descriptor) else {throw PipelineError.failure("显示缓冲分配失败")}
            freeTextures.append(OutputTextures(processed:texture,original:original))
        }
    }
    func planes(_ pixelBuffer: CVPixelBuffer) throws -> (MTLTexture,MTLTexture,CVMetalTexture,CVMetalTexture) {
        guard CVPixelBufferGetWidth(pixelBuffer)==1920, CVPixelBufferGetHeight(pixelBuffer)==1080,
              CVPixelBufferGetPlaneCount(pixelBuffer)==2 else {throw PipelineError.failure("仅接收原尺寸 1920×1080 双平面 YUV420")}
        var y: CVMetalTexture?, uv: CVMetalTexture?
        let a=CVMetalTextureCacheCreateTextureFromImage(nil,textureCache,pixelBuffer,nil,.r8Unorm,1920,1080,0,&y)
        let b=CVMetalTextureCacheCreateTextureFromImage(nil,textureCache,pixelBuffer,nil,.rg8Unorm,960,540,1,&uv)
        guard a==0,b==0,let yp=y,let up=uv,let yt=CVMetalTextureGetTexture(yp),let ut=CVMetalTextureGetTexture(up) else {throw PipelineError.failure("YUV IOSurface 无法映射为 Metal 纹理")}
        return (yt,ut,yp,up)
    }
    private func dispatch(_ enc: MTLComputeCommandEncoder,_ pipeline: MTLComputePipelineState,_ width: Int,_ height: Int=1) {
        enc.setComputePipelineState(pipeline)
        let w=height==1 ? min(pipeline.maxTotalThreadsPerThreadgroup,256) : 16
        let h=height==1 ? 1 : 16
        enc.dispatchThreads(MTLSize(width:width,height:height,depth:1),threadsPerThreadgroup:MTLSize(width:w,height:h,depth:1))
    }
    private func checkedWait(_ command: MTLCommandBuffer) throws {
        command.commit();command.waitUntilCompleted()
        if let error=command.error {throw error}
    }
    func pack(_ planes: (MTLTexture,MTLTexture,CVMetalTexture,CVMetalTexture), model: ModelRuntime, parameters: MetalParameters) throws {
        guard let command=queue.makeCommandBuffer(),let encoder=command.makeComputeCommandEncoder() else {throw PipelineError.failure("Metal pack command 不可用")}
        var p=parameters
        encoder.setTexture(planes.0,index:0);encoder.setTexture(planes.1,index:1)
        encoder.setBuffer(model.inputs["cur_frame"],offset:0,index:0)
        encoder.setBuffer(model.inputs["his_frame"],offset:0,index:1)
        encoder.setBytes(&p,length:MemoryLayout<MetalParameters>.stride,index:2)
        dispatch(encoder,packPipeline,Int(p.packedWidth),Int(p.packedHeight))
        encoder.setBuffer(model.inputs["dnr_zoom_map"],offset:0,index:0)
        encoder.setBuffer(model.inputs["fusion_zoom_map"],offset:0,index:1)
        dispatch(encoder,mapsPipeline,Int(p.mapWidth*p.mapHeight))
        encoder.endEncoding()
        try checkedWait(command)
    }
    func postprocess(_ planes: (MTLTexture,MTLTexture,CVMetalTexture,CVMetalTexture), model: ModelRuntime, parameters: MetalParameters) throws -> OutputTextures {
        guard let textures=try postprocess(planes,model:model,parameters:parameters,waitForDisplay:true) else {
            throw PipelineError.failure("显示缓冲暂不可用")
        }
        return textures
    }
    func postprocess(_ planes: (MTLTexture,MTLTexture,CVMetalTexture,CVMetalTexture), model: ModelRuntime,
                     parameters: MetalParameters, waitForDisplay: Bool, captureHistory: Bool=false) throws -> OutputTextures? {
        let acquired=available.wait(timeout:waitForDisplay ? .now()+2:.now()) == .success
        if !acquired && waitForDisplay {throw PipelineError.failure("显示缓冲回收超时")}
        var texture: OutputTextures?
        if acquired {
            poolLock.lock();texture=freeTextures.removeLast();poolLock.unlock()
        }
        do {
            guard let command=queue.makeCommandBuffer(),let encoder=command.makeComputeCommandEncoder() else {throw PipelineError.failure("Metal post command 不可用")}
            var p=parameters
            if captureHistory {
                let count=Int(6*p.packedWidth*p.packedHeight)*MemoryLayout<Float16>.size
                if previewHistory?.length != count {
                    previewHistory=device.makeBuffer(length:count,options:.storageModeShared)
                }
                guard previewHistory != nil else {throw PipelineError.failure("暂停历史缓冲分配失败")}
            }
            p.reserved=captureHistory ? 1:0
            if let texture=texture {
                encoder.setBuffer(model.outputs["out_yuv"],offset:0,index:0)
                encoder.setBytes(&p,length:MemoryLayout<MetalParameters>.stride,index:1)
                encoder.setTexture(texture.processed,index:0);encoder.setTexture(planes.0,index:1);encoder.setTexture(planes.1,index:2)
                encoder.setTexture(texture.original,index:3)
                dispatch(encoder,unpackPipeline,1920,1080)
            }
            encoder.setBuffer(model.inputs["his_frame"],offset:0,index:0)
            encoder.setBuffer(model.outputs["state_yuv"],offset:0,index:1)
            encoder.setBytes(&p,length:MemoryLayout<MetalParameters>.stride,index:2)
            encoder.setBuffer(captureHistory ? previewHistory:model.inputs["his_frame"],offset:0,index:3)
            dispatch(encoder,historyPipeline,Int(6*p.packedWidth*p.packedHeight))
            encoder.endEncoding()
            try checkedWait(command)
            return texture
        } catch {if let texture=texture {recycle(texture)};throw error}
    }
    func recycle(_ texture: OutputTextures) {
        poolLock.lock();freeTextures.append(texture);poolLock.unlock();available.signal()
    }
    func copyPreviewHistory() -> Data? {
        guard let buffer=previewHistory else {return nil}
        return Data(bytes:buffer.contents(),count:buffer.length)
    }
}
