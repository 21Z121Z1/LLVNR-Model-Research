import Foundation
import Metal
import CoreVideo

final class PipelineEngine {
    let work=DispatchQueue(label:"llvnr.inference.serial",qos:.userInteractive)
    private let lock=NSLock()
    private var fifo: [SourceFrame] = []
    private var draining=false, accepting=true, ended=false, finishing=false
    private var tuning=Tuning()
    private var resetPending=true
    private var model: ModelRuntime?
    private(set) var metal: MetalProcessor?
    private(set) var metrics: RunMetrics?
    var onReady: (()->Void)?
    var onOutput: ((DisplayFrame)->Void)?
    var onError: ((String)->Void)?
    var onFinished: (([String:Any])->Void)?
    private var pendingDisplay=0
    private(set) var isLive=true
    private var previousPTS: Double?
    private var paused=false,previewScheduled=false
    private var previewRevision=0
    private var latestSource: SourceFrame?
    private var latestParameters=MetalParameters()
    private var latestTiming: FrameTiming?
    private var frozenHistory: Data?

    func prepare(options: RunOptions) {
        isLive=options.replayDirectory==nil
        work.async { [self] in
            do {
                guard let device=MTLCreateSystemDefaultDevice() else {throw PipelineError.failure("需要 Apple Silicon / Metal 设备")}
                let processor=try MetalProcessor(device:device)
                let runtime=try ModelRuntime(device:device,modelID:options.modelID)
                try runtime.warmup()
                model=runtime;metal=processor
                if let directory=options.validationDirectory {
                    let result=try PipelineValidation.run(directory:directory,model:runtime,metal:processor)
                    DispatchQueue.main.async{[self] in onFinished?(result)}
                    return
                }
                metrics=try RunMetrics(directory:options.reportDirectory,manifest:runtime.manifest,source:options.replayDirectory==nil ? "camera":"paced_replay")
                metrics?.event("firmware_model_selected",["model_id":options.modelID ?? "default","source_model_sha256":runtime.manifest.source_model_sha256])
                metrics?.event("queue_policy",["policy":isLive ? "latest_pending_frame":"strict_fifo", "capacity":isLive ? 1:8])
                if let directory=options.replayDirectory {metrics?.event("replay_configuration",["directory":directory.path,"frames":options.replayFrames,"content":"saved real YUV frame sequence looped; each release receives a new ID and inference"])}
                DispatchQueue.main.async{[self] in onReady?()}
            } catch {failed(error)}
        }
    }
    func update(_ next: Tuning) {
        lock.lock()
        let changed=next != tuning
        let processingChanged=next.dnr != tuning.dnr || next.fusion != tuning.fusion || next.historyWeight != tuning.historyWeight
        tuning=next
        let preview=paused && processingChanged && !previewScheduled
        if paused && processingChanged {previewRevision+=1;previewScheduled=true}
        lock.unlock()
        if changed,let data=try? JSONEncoder().encode(next),let dict=try? JSONSerialization.jsonObject(with:data) as? [String:Any] {metrics?.event("tuning_requested",dict)}
        if preview {work.async{[self] in reprocessFrozenFrame()}}
    }
    func setPaused(_ next: Bool, completion: @escaping ()->Void) {
        lock.lock();paused=next;lock.unlock()
        work.async { [self] in
            if next {
                frozenHistory=metal?.copyPreviewHistory()
                metrics?.pause()
            } else {
                frozenHistory=nil
                lock.lock();resetPending=true;previousPTS=nil;lock.unlock()
                metrics?.resume()
            }
            metrics?.event(next ? "processing_paused":"processing_resumed")
            DispatchQueue.main.async(execute:completion)
        }
    }
    func resetState() {
        lock.lock();resetPending=true;let preview=paused;lock.unlock()
        metrics?.event("state_reset_requested")
        if preview {
            work.async { [self] in
                lock.lock();let active=paused;previewRevision+=1;previewScheduled=active;lock.unlock()
                guard active else{return}
                latestParameters.reset=1
                reprocessFrozenFrame()
            }
        }
    }
    func enqueue(_ frame: SourceFrame) {
        lock.lock()
        guard accepting && !paused else {lock.unlock();return}
        // Live capture keeps one waiting frame. A temporary stall must not
        // build a quarter-second backlog or stop the camera.
        if isLive, !fifo.isEmpty {
            for stale in fifo {metrics?.drop("input_superseded",values:["id":stale.id,"replacement_id":frame.id])}
            fifo.removeAll(keepingCapacity:true)
            resetPending=true
        }
        let full = !isLive && fifo.count>=8
        if !full {fifo.append(frame)}
        let depth=fifo.count
        metrics?.accepted(frame,queueDepth:depth)
        let launch = !draining && !full
        if launch {draining=true}
        lock.unlock()
        if full {
            metrics?.drop("queue_overflow",values:["id":frame.id,"capacity":8])
            metrics?.recoverableIssue("input_queue",message:"重放输入队列已满，正在重试后续帧")
        } else {
            metrics?.recovered("input_queue")
            if launch {work.async{[self] in drain()}}
        }
    }
    func sourceEnded() {
        lock.lock();accepting=false;ended=true;let idle = !draining;lock.unlock()
        if idle {work.async{[self] in tryFinish()}}
    }
    private func next() -> (SourceFrame,Tuning,Bool)? {
        lock.lock();defer{lock.unlock()}
        guard !fifo.isEmpty else {draining=false;return nil}
        let frame=fifo.removeFirst()
        let discontinuity=previousPTS.map{frame.pts-$0>1.5/30} ?? false
        let reset=resetPending || discontinuity
        previousPTS=frame.pts
        resetPending=false
        metrics?.queueDepth(fifo.count)
        return (frame,tuning,reset)
    }
    private func drain() {
        while let (frame,tuning,reset)=next() {
            autoreleasepool {
                do {
                    guard let runtime=model,let processor=metal else {throw PipelineError.failure("推理运行时尚未就绪")}
                    let begin=hostSeconds()
                    let planes=try processor.planes(frame.pixelBuffer)
                    var p=runtime.manifest.metalParameters()
                    p.reset=reset ? 1:0
                    p.fullRange=CVPixelBufferGetPixelFormatType(frame.pixelBuffer)==kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ? 1:0
                    if let matrix=CVBufferCopyAttachment(frame.pixelBuffer,kCVImageBufferYCbCrMatrixKey,nil) as? String {p.matrix601=matrix == (kCVImageBufferYCbCrMatrix_ITU_R_601_4 as String) ? 1:0}
                    p.dnr=tuning.dnr;p.fusion=tuning.fusion;p.historyWeight=tuning.historyWeight;p.split=tuning.split
                    try processor.pack(planes,model:runtime,parameters:p)
                    let packed=hostSeconds()
                    try runtime.execute()
                    let inferred=hostSeconds()
                    let texture=try processor.postprocess(planes,model:runtime,parameters:p,waitForDisplay:!isLive,captureHistory:true)
                    let timing=FrameTiming(id:frame.id,sourceSequence:frame.sourceSequence,
                        sourcePTS:frame.pts,sourceHostTime:frame.sourceTime,received:frame.received,inferenceStart:begin,
                        packed:packed,inferred:inferred,postprocessed:hostSeconds(),reset:reset,tuning:tuning)
                    metrics?.processedFrame(timing)
                    latestSource=frame;latestParameters=p;latestTiming=timing
                    metrics?.recovered("processing")
                    guard let texture=texture else {
                        metrics?.recoverableIssue("display",message:"显示缓冲暂未回收，正在等待恢复")
                        metrics?.drop("display_superseded",values:["id":frame.id,"reason":"display buffers busy"])
                        return
                    }
                    lock.lock();pendingDisplay+=1;lock.unlock()
                    let display=DisplayFrame(textures:texture,source:frame,timing:timing,recycle:{processor.recycle(texture)})
                    DispatchQueue.main.async{[self] in onOutput?(display)}
                } catch {
                    metrics?.recoverableIssue("processing",message:error.localizedDescription)
                    resetState()
                }
            }
        }
        tryFinish()
    }
    func presented(_ frame: DisplayFrame,time: Double) {
        guard frame.claimCompletion() else {return}
        guard !frame.isPreview else {return}
        var timing=frame.timing;timing.presented=time
        metrics?.presentedFrame(timing)
        lock.lock();pendingDisplay-=1;lock.unlock()
        finishIfIdle()
    }
    func displayFailed(_ frame: DisplayFrame,message: String) {
        guard frame.claimCompletion() else {return}
        metrics?.recoverableIssue("display",message:message)
        if !frame.isPreview {
            metrics?.drop("display_superseded",values:["id":frame.source.id,"reason":message])
            lock.lock();pendingDisplay-=1;lock.unlock()
        }
        frame.recycle()
        finishIfIdle()
        // A display gap fails acceptance, but live capture keeps running so the
        // user can restore visibility without restarting the camera session.
    }
    func displaySuperseded(_ frame: DisplayFrame,reason: String) {
        guard frame.claimCompletion() else {return}
        if !frame.isPreview {
            metrics?.drop("display_superseded",values:["id":frame.source.id,"reason":reason])
            lock.lock();pendingDisplay-=1;lock.unlock()
        }
        frame.recycle()
        finishIfIdle()
    }
    private func finishIfIdle() {
        lock.lock();let idle=ended && !draining;lock.unlock()
        if idle {work.async{[self] in tryFinish()}}
    }
    private func reprocessFrozenFrame() {
        while true {
            lock.lock()
            guard paused else {previewScheduled=false;lock.unlock();return}
            let revision=previewRevision,nextTuning=tuning
            lock.unlock()
            do {
                guard let runtime=model,let processor=metal,let source=latestSource,
                      let baseHistory=frozenHistory, var timing=latestTiming,
                      let history=runtime.inputs["his_frame"] else {
                    lock.lock();previewScheduled=false;lock.unlock();return
                }
                let preserved=Data(bytes:history.contents(),count:baseHistory.count)
                defer {preserved.withUnsafeBytes {memcpy(history.contents(),$0.baseAddress!,preserved.count)}}
                baseHistory.withUnsafeBytes {memcpy(history.contents(),$0.baseAddress!,baseHistory.count)}
                let planes=try processor.planes(source.pixelBuffer)
                var p=latestParameters
                p.dnr=nextTuning.dnr;p.fusion=nextTuning.fusion;p.historyWeight=nextTuning.historyWeight
                try processor.pack(planes,model:runtime,parameters:p)
                try runtime.execute()
                let textures=try processor.postprocess(planes,model:runtime,parameters:p,waitForDisplay:false)
                lock.lock();let current=paused && revision==previewRevision;lock.unlock()
                if let textures=textures {
                    if current {
                        timing.tuning=nextTuning
                        let display=DisplayFrame(textures:textures,source:source,timing:timing,isPreview:true,
                                                 recycle:{processor.recycle(textures)})
                        metrics?.event("paused_frame_reprocessed",["id":source.id,"dnr":p.dnr,"fusion":p.fusion,
                            "history_weight":p.historyWeight,"history_advanced":false])
                        DispatchQueue.main.async{[self] in onOutput?(display)}
                    } else {processor.recycle(textures)}
                } else if current {
                    retryFrozenPreview()
                    return
                }
                metrics?.recovered("paused_preview")
            } catch {
                metrics?.recoverableIssue("paused_preview",message:error.localizedDescription)
                retryFrozenPreview()
                return
            }
            lock.lock()
            let again=paused && revision != previewRevision
            if !again {previewScheduled=false}
            lock.unlock()
            if !again {return}
        }
    }
    private func retryFrozenPreview() {
        work.asyncAfter(deadline:.now()+0.1){[self] in reprocessFrozenFrame()}
    }
    private func tryFinish() {
        lock.lock()
        let done=ended && !draining && fifo.isEmpty && pendingDisplay==0 && !finishing
        if done {finishing=true}
        lock.unlock()
        guard done else {return}
        do {
            let result=try metrics?.finish() ?? [:]
            model=nil
            DispatchQueue.main.async{[self] in onFinished?(result)}
        } catch {failed(error)}
    }
    private func failed(_ error: Error) {DispatchQueue.main.async{[self] in onError?(error.localizedDescription)}}
}
