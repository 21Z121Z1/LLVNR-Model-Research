import AVFoundation
import Metal

struct VideoProgress {
    var fraction=0.0
    var pass=0
    var frames=0
}

struct VideoProcessingResult {
    let input: URL, output: URL
    let frames: Int, passes: Int
    let multiPassRequested: Bool, multiPassSupported: Bool
    let width: Int, height: Int, bitDepth: Int
    let sourceBitRate: Float, outputBitRate: Float
    let frameRate: Float
    let colors: [String:String]
    let duration: Double
    let transform: CGAffineTransform
    let isHDR: Bool

    var report: [String:Any] {
        ["passed":true,"frames":frames,"passes":passes,"hardware_encoding_required":true,
         "multi_pass_requested":multiPassRequested,"multi_pass_supported":multiPassSupported,
         "width":width,"height":height,"bit_depth":bitDepth,"source_bit_rate":sourceBitRate,
         "output_bit_rate":outputBitRate,"frame_rate":frameRate,"colors":colors,
         "duration_seconds":duration,"hdr":isHDR,
         "model_input_domain":"normalized source YUV code values; original HLG/PQ transfer preserved",
         "source_path":input.path,"output_path":output.path]
    }
}

final class VideoProcessor {
    private let work=DispatchQueue(label:"llvnr.video.processing",qos:.userInitiated)
    private let lock=NSLock()
    private var cancelled=false
    private var failure: Error?

    func cancel() {lock.lock();cancelled=true;lock.unlock()}

    func start(input: URL, output: URL, modelID: String, tuning: Tuning, multiPass: Bool,
               progress: @escaping (VideoProgress)->Void,
               completion: @escaping (Result<VideoProcessingResult,Error>)->Void) {
        Task {
            do {
                let info=try await VideoMediaInfo.load(input)
                work.async { [self] in
                    do {
                        let result=try encode(info:info,input:input,output:output,modelID:modelID,
                                              tuning:tuning,multiPass:multiPass,progress:progress)
                        DispatchQueue.main.async{completion(.success(result))}
                    } catch {DispatchQueue.main.async{completion(.failure(error))}}
                }
            } catch {DispatchQueue.main.async{completion(.failure(error))}}
        }
    }

    private func check(_ writer: AVAssetWriter?=nil) throws {
        lock.lock();let cancelled=cancelled,error=failure;lock.unlock()
        if let error=error {throw error}
        if cancelled {throw CancellationError()}
        if let writer=writer,writer.status == .failed || writer.status == .cancelled {
            throw writer.error ?? PipelineError.failure("视频编码器已停止")
        }
    }
    private func fail(_ error: Error) {lock.lock();if failure==nil {failure=error};lock.unlock()}

    private func encode(info: VideoMediaInfo, input: URL, output: URL, modelID: String,
                        tuning: Tuning, multiPass: Bool,
                        progress: @escaping (VideoProgress)->Void) throws -> VideoProcessingResult {
        guard input.resolvingSymlinksInPath().standardizedFileURL != output.resolvingSymlinksInPath().standardizedFileURL else {
            throw PipelineError.failure("请为处理结果选择一个新的文件名")
        }
        try check()
        guard info.duration.isNumeric,info.duration.seconds>0,let device=MTLCreateSystemDefaultDevice() else {
            throw PipelineError.failure("视频时长或 Metal 设备不可用")
        }
        let model=try ModelRuntime(device:device,modelID:modelID)
        try model.warmup()
        let processor=try VideoFrameProcessor(model:model,width:info.width,height:info.height)
        let temporary=output.deletingLastPathComponent().appendingPathComponent(".llvnr-"+UUID().uuidString+"."+output.pathExtension)
        let type: AVFileType=output.pathExtension.lowercased()=="mp4" ? .mp4:.mov
        let writer=try AVAssetWriter(outputURL:temporary,fileType:type)
        var completed=false
        defer {
            if !completed {writer.cancelWriting();try? FileManager.default.removeItem(at:temporary)}
        }
        let settings=info.writerSettings()
        guard writer.canApply(outputSettings:settings,forMediaType:.video) else {
            throw PipelineError.failure("硬件编码器不支持源视频的编码／颜色规格")
        }
        let video=AVAssetWriterInput(mediaType:.video,outputSettings:settings,sourceFormatHint:info.format)
        video.expectsMediaDataInRealTime=false
        video.performsMultiPassEncodingIfSupported=multiPass
        video.transform=info.transform;video.mediaTimeScale=info.timeScale
        guard writer.canAdd(video) else {throw PipelineError.failure("无法创建视频输出轨道")}
        writer.add(video);writer.metadata=info.metadata

        var copyInputs: [(AVAssetTrack,AVAssetWriterInput)] = []
        for (track,format) in info.passthroughTracks {
            let target=AVAssetWriterInput(mediaType:track.mediaType,outputSettings:nil,sourceFormatHint:format)
            target.expectsMediaDataInRealTime=false
            guard writer.canAdd(target) else {throw PipelineError.failure("无法原样保留源文件的 \(track.mediaType.rawValue) 轨道")}
            writer.add(target);copyInputs.append((track,target))
        }
        guard writer.startWriting() else {throw writer.error ?? PipelineError.failure("硬件编码器启动失败")}
        writer.startSession(atSourceTime:.zero)
        let supportsMultiPass=video.canPerformMultiplePasses
        let copies=DispatchGroup()
        for (track,target) in copyInputs {
            copies.enter()
            DispatchQueue(label:"llvnr.video.copy.\(track.trackID)",qos:.userInitiated).async { [self] in
                defer {copies.leave()}
                do {
                    let reader=try AVAssetReader(asset:info.asset)
                    let source=AVAssetReaderTrackOutput(track:track,outputSettings:nil)
                    source.alwaysCopiesSampleData=false
                    guard reader.canAdd(source) else {throw PipelineError.failure("源辅助轨道读取失败")}
                    reader.add(source)
                    guard reader.startReading() else {throw reader.error ?? PipelineError.failure("源辅助轨道启动失败")}
                    defer {reader.cancelReading()}
                    while true {
                        try check(writer)
                        if !target.isReadyForMoreMediaData {Thread.sleep(forTimeInterval:0.002);continue}
                        let next=source.copyNextSampleBuffer()
                        guard let sample=next else {break}
                        guard target.append(sample) else {throw writer.error ?? PipelineError.failure("辅助轨道写入失败")}
                    }
                    if reader.status == .failed {throw reader.error ?? PipelineError.failure("辅助轨道解码失败")}
                    target.markAsFinished()
                } catch {fail(error);writer.cancelWriting()}
            }
        }

        var pool: CVPixelBufferPool?
        let attributes: [String:Any] = [kCVPixelBufferWidthKey as String:info.width,kCVPixelBufferHeightKey as String:info.height,
            kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            kCVPixelBufferIOSurfacePropertiesKey as String:[:],kCVPixelBufferMetalCompatibilityKey as String:true]
        guard CVPixelBufferPoolCreate(nil,nil,attributes as CFDictionary,&pool)==kCVReturnSuccess,let pool=pool else {
            throw PipelineError.failure("视频输出像素池不可用")
        }
        let passSignal=DispatchSemaphore(value:0),passLock=NSLock()
        var nextPass: AVAssetWriterInputPassDescription?
        video.respondToEachPassDescription(on:DispatchQueue(label:"llvnr.video.passes")) { [weak video] in
            passLock.lock();nextPass=video?.currentPassDescription;passLock.unlock();passSignal.signal()
        }
        var passes=0,firstPassFrames=0
        while true {
            while passSignal.wait(timeout:.now()+0.05) != .success {try check(writer)}
            passLock.lock();let description=nextPass;passLock.unlock()
            guard let description=description else {video.markAsFinished();break}
            passes+=1
            let ranges=description.sourceTimeRanges.map(\.timeRangeValue)
            let reader=try AVAssetReader(asset:info.asset)
            let source=AVAssetReaderTrackOutput(track:info.track,outputSettings:[
                kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
                kCVPixelBufferIOSurfacePropertiesKey as String:[:],kCVPixelBufferMetalCompatibilityKey as String:true,
                AVVideoAllowWideColorKey:true,AVVideoColorPropertiesKey:info.colors])
            source.alwaysCopiesSampleData=false
            guard reader.canAdd(source) else {throw PipelineError.failure("无法创建 10-bit 视频解码输出")}
            reader.add(source)
            guard reader.startReading() else {throw reader.error ?? PipelineError.failure("视频解码启动失败")}
            processor.reset()
            var frames=0,lastProgress=0.0
            do {
                while let sample=source.copyNextSampleBuffer() {
                    try autoreleasepool {
                        try check(writer)
                        guard let image=CMSampleBufferGetImageBuffer(sample) else {throw PipelineError.failure("视频帧没有图像缓冲")}
                        var outputBuffer: CVPixelBuffer?
                        guard CVPixelBufferPoolCreatePixelBuffer(nil,pool,&outputBuffer)==kCVReturnSuccess,let outputBuffer=outputBuffer else {
                            throw PipelineError.failure("视频输出像素分配失败")
                        }
                        try processor.process(image,into:outputBuffer,tuning:tuning)
                        info.attachColors(to:outputBuffer)
                        let pts=CMSampleBufferGetPresentationTimeStamp(sample)
                        if ranges.contains(where:{CMTimeRangeContainsTime($0,time:pts)}) {
                            while !video.isReadyForMoreMediaData {try check(writer);Thread.sleep(forTimeInterval:0.002)}
                            var format: CMVideoFormatDescription?
                            let formatStatus=CMVideoFormatDescriptionCreateForImageBuffer(allocator:kCFAllocatorDefault,imageBuffer:outputBuffer,formatDescriptionOut:&format)
                            var timing=CMSampleTimingInfo(duration:CMSampleBufferGetDuration(sample),presentationTimeStamp:pts,decodeTimeStamp:.invalid)
                            var encoded: CMSampleBuffer?
                            guard formatStatus==noErr,let format=format,
                                  CMSampleBufferCreateReadyWithImageBuffer(allocator:kCFAllocatorDefault,imageBuffer:outputBuffer,
                                    formatDescription:format,sampleTiming:&timing,sampleBufferOut:&encoded)==noErr,let encoded=encoded,
                                  video.append(encoded) else {throw writer.error ?? PipelineError.failure("视频帧编码失败")}
                        }
                        frames+=1
                        if hostSeconds()-lastProgress>0.1 {
                            let value=VideoProgress(fraction:min(1,max(0,pts.seconds/info.duration.seconds)),pass:passes,frames:frames)
                            DispatchQueue.main.async{progress(value)};lastProgress=hostSeconds()
                        }
                    }
                }
                if reader.status == .failed {throw reader.error ?? PipelineError.failure("视频读取失败")}
            } catch {reader.cancelReading();throw error}
            reader.cancelReading()
            if passes==1 {firstPassFrames=frames}
            video.markCurrentPassAsFinished()
        }
        while copies.wait(timeout:.now()+0.05) != .success {try check(writer)}
        try check(writer)
        writer.endSession(atSourceTime:info.duration)
        let finished=DispatchSemaphore(value:0)
        writer.finishWriting{finished.signal()}
        while finished.wait(timeout:.now()+0.05) != .success {try check(writer)}
        guard writer.status == .completed else {throw writer.error ?? PipelineError.failure("视频封装失败")}
        if FileManager.default.fileExists(atPath:output.path) {
            _=try FileManager.default.replaceItemAt(output,withItemAt:temporary)
        } else {try FileManager.default.moveItem(at:temporary,to:output)}
        completed=true
        let finalAsset=AVURLAsset(url:output)
        let actualRate=finalAsset.tracks(withMediaType:.video).first?.estimatedDataRate ?? 0
        return VideoProcessingResult(input:input,output:output,frames:firstPassFrames,passes:passes,
            multiPassRequested:multiPass,multiPassSupported:supportsMultiPass,width:info.width,height:info.height,
            bitDepth:info.bitDepth,sourceBitRate:info.bitRate,outputBitRate:actualRate,frameRate:info.frameRate,
            colors:info.colors,duration:info.duration.seconds,transform:info.transform,isHDR:info.isHDR)
    }
}
