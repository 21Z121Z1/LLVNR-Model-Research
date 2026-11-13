import Foundation
import CoreVideo
import Metal

private struct VideoTileParameters {
    var width: UInt32, height: UInt32
    var x: Int32, y: Int32
}

/// Full-resolution tiles overlap over the model's padding. Each tile keeps an
/// independent temporal state and the overlapping outputs are feathered.
final class VideoFrameProcessor {
    let model: ModelRuntime
    private let queue: MTLCommandQueue
    private let pack: MTLComputePipelineState, maps: MTLComputePipelineState
    private let accumulate: MTLComputePipelineState, clear: MTLComputePipelineState
    private let finish: MTLComputePipelineState, history: MTLComputePipelineState
    private let accumulation: MTLTexture
    private var cache: CVMetalTextureCache!
    private let width: Int, height: Int
    private var tiles: [(VideoTileParameters,MTLBuffer)] = []
    private var firstFrame=true

    init(model: ModelRuntime, width: Int, height: Int) throws {
        self.model=model;self.width=width;self.height=height
        let device=model.device
        guard width>0,height>0,width%2==0,height%2==0,let queue=device.makeCommandQueue(),
              let main=Bundle.module.url(forResource:"Pipeline",withExtension:"metal",subdirectory:"Resources"),
              let video=Bundle.module.url(forResource:"VideoProcessing",withExtension:"metal",subdirectory:"Resources") else {
            throw PipelineError.failure("视频尺寸或 Metal 资源不可用")
        }
        self.queue=queue
        let source=try String(contentsOf:main,encoding:.utf8)+"\n"+String(contentsOf:video,encoding:.utf8)
        let library=try device.makeLibrary(source:source,options:nil)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function=library.makeFunction(name:name) else {throw PipelineError.failure("缺少视频内核 \(name)")}
            return try device.makeComputePipelineState(function:function)
        }
        pack=try pipeline("packVideo420");maps=try pipeline("fillMaps")
        accumulate=try pipeline("accumulateVideoTile");clear=try pipeline("clearVideo")
        finish=try pipeline("finishVideo420");history=try pipeline("updateHistory")
        let descriptor=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba32Float,width:width,height:height,mipmapped:false)
        descriptor.storageMode = .private;descriptor.usage=[.shaderRead,.shaderWrite]
        guard let accumulation=device.makeTexture(descriptor:descriptor),
              CVMetalTextureCacheCreate(nil,nil,device,nil,&cache)==kCVReturnSuccess,
              let historyBytes=model.manifest.inputs.first(where:{$0.name=="his_frame"})?.bytes else {
            throw PipelineError.failure("视频处理缓冲分配失败")
        }
        self.accumulation=accumulation
        for y in stride(from:0,to:height,by:model.manifest.height) {
            for x in stride(from:0,to:width,by:model.manifest.width) {
                guard let state=device.makeBuffer(length:historyBytes,options:.storageModeShared) else {
                    throw PipelineError.failure("视频时间状态分配失败")
                }
                tiles.append((VideoTileParameters(width:UInt32(width),height:UInt32(height),x:Int32(x),y:Int32(y)),state))
            }
        }
    }

    func reset() {firstFrame=true}

    struct HistorySnapshot {
        let states: [Data]
        let firstFrame: Bool
        var bytes: Int {states.reduce(0) {$0 + $1.count}}
    }

    func snapshotHistory() -> HistorySnapshot {
        HistorySnapshot(states: tiles.map {Data(bytes: $0.1.contents(), count: $0.1.length)},
                        firstFrame: firstFrame)
    }

    func restoreHistory(_ snapshot: HistorySnapshot) {
        for (tile, state) in zip(tiles, snapshot.states) {
            state.withUnsafeBytes { bytes in
                memcpy(tile.1.contents(), bytes.baseAddress!, state.count)
            }
        }
        firstFrame = snapshot.firstFrame
    }

    private func planes(_ buffer: CVPixelBuffer) throws -> (MTLTexture,MTLTexture,CVMetalTexture,CVMetalTexture) {
        guard CVPixelBufferGetWidth(buffer)==width,CVPixelBufferGetHeight(buffer)==height,
              CVPixelBufferGetPixelFormatType(buffer)==kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange else {
            throw PipelineError.failure("视频解码未提供所需的原尺寸 10-bit YUV420")
        }
        var y: CVMetalTexture?,uv: CVMetalTexture?
        let a=CVMetalTextureCacheCreateTextureFromImage(nil,cache,buffer,nil,.r16Unorm,width,height,0,&y)
        let b=CVMetalTextureCacheCreateTextureFromImage(nil,cache,buffer,nil,.rg16Unorm,width/2,height/2,1,&uv)
        guard a==0,b==0,let y=y,let uv=uv,let yt=CVMetalTextureGetTexture(y),let uvt=CVMetalTextureGetTexture(uv) else {
            throw PipelineError.failure("10-bit 视频无法映射到 Metal")
        }
        return (yt,uvt,y,uv)
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder,_ pipeline: MTLComputePipelineState,_ width: Int,_ height: Int=1) {
        encoder.setComputePipelineState(pipeline)
        encoder.dispatchThreads(MTLSize(width:width,height:height,depth:1),
            threadsPerThreadgroup:MTLSize(width:16,height:height==1 ? 1:8,depth:1))
    }
    private func wait(_ command: MTLCommandBuffer) throws {
        command.commit();command.waitUntilCompleted()
        if let error=command.error {throw error}
    }

    func process(_ source: CVPixelBuffer, into output: CVPixelBuffer, tuning: Tuning) throws {
        let input=try planes(source),destination=try planes(output)
        defer {withExtendedLifetime(input){};withExtendedLifetime(destination){}}
        guard let initial=queue.makeCommandBuffer(),let clearEncoder=initial.makeComputeCommandEncoder() else {
            throw PipelineError.failure("视频命令队列不可用")
        }
        clearEncoder.setTexture(accumulation,index:0)
        dispatch(clearEncoder,clear,width,height);clearEncoder.endEncoding()
        try wait(initial)
        var p=model.manifest.metalParameters()
        p.reset=firstFrame ? 1:0;p.dnr=tuning.dnr;p.fusion=tuning.fusion;p.historyWeight=tuning.historyWeight
        guard let current=model.inputs["cur_frame"],let recurrent=model.inputs["his_frame"],
              let out=model.outputs["out_yuv"],let state=model.outputs["state_yuv"] else {
            throw PipelineError.failure("缺少视频模型端口")
        }
        for (parameters,savedState) in tiles {
            if !firstFrame {memcpy(recurrent.contents(),savedState.contents(),savedState.length)}
            guard let command=queue.makeCommandBuffer(),let encoder=command.makeComputeCommandEncoder() else {
                throw PipelineError.failure("视频打包命令不可用")
            }
            var tile=parameters
            encoder.setTexture(input.0,index:0);encoder.setTexture(input.1,index:1)
            encoder.setBuffer(current,offset:0,index:0);encoder.setBuffer(recurrent,offset:0,index:1)
            encoder.setBytes(&p,length:MemoryLayout<MetalParameters>.stride,index:2)
            encoder.setBytes(&tile,length:MemoryLayout<VideoTileParameters>.stride,index:3)
            dispatch(encoder,pack,Int(p.packedWidth),Int(p.packedHeight))
            encoder.setBuffer(model.inputs["dnr_zoom_map"],offset:0,index:0)
            encoder.setBuffer(model.inputs["fusion_zoom_map"],offset:0,index:1)
            dispatch(encoder,maps,Int(p.mapWidth*p.mapHeight))
            encoder.endEncoding();try wait(command)
            try model.execute()
            guard let post=queue.makeCommandBuffer(),let postEncoder=post.makeComputeCommandEncoder() else {
                throw PipelineError.failure("视频输出命令不可用")
            }
            postEncoder.setBuffer(out,offset:0,index:0)
            postEncoder.setBytes(&p,length:MemoryLayout<MetalParameters>.stride,index:1)
            postEncoder.setBytes(&tile,length:MemoryLayout<VideoTileParameters>.stride,index:2)
            postEncoder.setTexture(accumulation,index:0)
            dispatch(postEncoder,accumulate,Int(p.packedWidth*2),Int(p.packedHeight*2))
            postEncoder.setBuffer(recurrent,offset:0,index:0);postEncoder.setBuffer(state,offset:0,index:1)
            postEncoder.setBytes(&p,length:MemoryLayout<MetalParameters>.stride,index:2)
            postEncoder.setBuffer(recurrent,offset:0,index:3)
            dispatch(postEncoder,history,Int(6*p.packedWidth*p.packedHeight))
            postEncoder.endEncoding();try wait(post)
            memcpy(savedState.contents(),recurrent.contents(),savedState.length)
        }
        guard let final=queue.makeCommandBuffer(),let finalEncoder=final.makeComputeCommandEncoder() else {
            throw PipelineError.failure("视频合成命令不可用")
        }
        finalEncoder.setTexture(accumulation,index:0)
        finalEncoder.setTexture(destination.0,index:1);finalEncoder.setTexture(destination.1,index:2)
        dispatch(finalEncoder,finish,width,height);finalEncoder.endEncoding();try wait(final)
        firstFrame=false
    }
}
