import Foundation
import Metal
import AppKit

enum PipelineValidation {
    static func run(directory: URL,model: ModelRuntime,metal: MetalProcessor) throws -> [String:Any] {
        // The reference fixture points to real native Mac frames, loaded at the
        // original 1080p resolution. No camera or source scheduling in this test.
        let config=try JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("fixture.json"))) as! [String:String]
        let folder=URL(fileURLWithPath:config["frames_directory"]!)
        let previous=try ReplaySource.load(directory:folder,basename:"frame_000000")
        let current=try ReplaySource.load(directory:folder,basename:"frame_000001")
        var p=model.manifest.metalParameters();p.reset=1
        try metal.pack(metal.planes(previous),model:model,parameters:p)
        p.reset=0
        let planes=try metal.planes(current)
        try metal.pack(planes,model:model,parameters:p)
        var errors: [String:Any] = [:]
        var passed=true
        func compare(_ name: String,_ buffer: MTLBuffer,_ count: Int,_ maximum: Double,_ mean: Double) throws {
            let data=try Data(contentsOf:directory.appendingPathComponent(name+".fp16"))
            guard data.count==count*2 else {throw PipelineError.failure("验证 fixture 尺寸不符：\(name)")}
            let got=buffer.contents().bindMemory(to:Float16.self,capacity:count)
            let error: (Double,Double,Bool)=data.withUnsafeBytes {raw in
                let want=raw.bindMemory(to:Float16.self)
                var maxError=0.0,sum=0.0,finite=true
                for i in 0..<count {
                    let delta=Double(abs(Float(got[i])-Float(want[i])))
                    if !delta.isFinite {finite=false;continue}
                    maxError=max(maxError,delta);sum+=delta
                }
                return (maxError,sum/Double(count),finite)
            }
            let ok=error.2 && error.0<=maximum && error.1<=mean
            passed=passed && ok
            errors[name]=["max_abs":error.0,"mean_abs":error.1,"finite":error.2,"pass":ok]
        }
        for spec in model.manifest.inputs {try compare(spec.name,model.inputs[spec.name]!,spec.bytes/2,0.0005,0.00005)}
        try model.execute()
        for spec in model.manifest.outputs {try compare(spec.name,model.outputs[spec.name]!,spec.bytes/2,0.02,0.001)}
        let oldHistory=Data(bytes:model.inputs["his_frame"]!.contents(),count:model.manifest.inputs.first{$0.name=="his_frame"}!.bytes)
        let state=Data(bytes:model.outputs["state_yuv"]!.contents(),count:oldHistory.count)
        let texture=try metal.postprocess(planes,model:model,parameters:p)
        defer{metal.recycle(texture)}
        var expected=Data(count:oldHistory.count)
        expected.withUnsafeMutableBytes {dst in oldHistory.withUnsafeBytes {a in state.withUnsafeBytes {b in
            let x=a.bindMemory(to:Float16.self),y=b.bindMemory(to:Float16.self),z=dst.bindMemory(to:Float16.self)
            for i in 0..<z.count {z[i]=Float16(0.5*Float(x[i])+0.5*Float(y[i]))}
        }}}
        try expected.write(to:directory.appendingPathComponent("history_after_update.fp16"))
        try compare("history_after_update",model.inputs["his_frame"]!,oldHistory.count/2,0,0)
        // Copy once for validation only; the live display path stays on GPU.
        let bytesPerRow=1920*4
        let buffer=metal.device.makeBuffer(length:bytesPerRow*1080,options:.storageModeShared)!
        let command=metal.queue.makeCommandBuffer()!,blit=command.makeBlitCommandEncoder()!
        blit.copy(from:texture.processed,sourceSlice:0,sourceLevel:0,sourceOrigin:MTLOrigin(),sourceSize:MTLSize(width:1920,height:1080,depth:1),to:buffer,destinationOffset:0,destinationBytesPerRow:bytesPerRow,destinationBytesPerImage:bytesPerRow*1080)
        blit.endEncoding();command.commit();command.waitUntilCompleted()
        if let error=command.error {throw error}
        let data=Data(bytes:buffer.contents(),count:buffer.length)
        try data.write(to:directory.appendingPathComponent("metal_output.bgra"))
        let provider=CGDataProvider(data:data as CFData)!
        let image=CGImage(width:1920,height:1080,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:bytesPerRow,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.premultipliedFirst.rawValue|CGBitmapInfo.byteOrder32Little.rawValue),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent)!
        try NSBitmapImageRep(cgImage:image).representation(using:.png,properties:[:])!.write(to:directory.appendingPathComponent("metal_output.png"))
        // Exhaust the actual display pool, then exercise the live path. History
        // must keep updating while no display texture is available, and the next
        // request must succeed as soon as a retained texture is returned.
        var retained: [OutputTextures] = []
        defer {for item in retained {metal.recycle(item)}}
        for _ in 0..<5 {retained.append(try metal.postprocess(planes,model:model,parameters:p))}
        let before=Data(bytes:model.inputs["his_frame"]!.contents(),count:oldHistory.count)
        let start=hostSeconds()
        let unavailable=try metal.postprocess(planes,model:model,parameters:p,waitForDisplay:false)
        let elapsed=hostSeconds()-start
        let after=model.inputs["his_frame"]!.contents().bindMemory(to:Float16.self,capacity:oldHistory.count/2)
        var historyContinued=true
        before.withUnsafeBytes {a in state.withUnsafeBytes {b in
            let x=a.bindMemory(to:Float16.self),y=b.bindMemory(to:Float16.self)
            for i in 0..<x.count where after[i] != Float16(0.5*Float(x[i])+0.5*Float(y[i])) {historyContinued=false;break}
        }}
        metal.recycle(retained.removeLast())
        let resumed=try metal.postprocess(planes,model:model,parameters:p,waitForDisplay:false)
        if let resumed=resumed {metal.recycle(resumed)}
        let recoveryPass=unavailable==nil && elapsed<0.2 && resumed != nil && historyContinued
        if let unavailable=unavailable {metal.recycle(unavailable)}
        passed=passed && recoveryPass
        let result: [String:Any] = ["validation_pass":passed,"errors":errors,
            "display_pool_recovery":["passed":recoveryPass,"exhausted_request_ms":elapsed*1000,
                                     "history_updated_without_display":historyContinued],
            "scope":"Metal pack + ANE full graph + Metal history feedback + exhausted display pool recovery; all outputs checked against Python ANE reference","model_files":model.manifest.files]
        try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:directory.appendingPathComponent("native_validation.json"))
        return result
    }
}
