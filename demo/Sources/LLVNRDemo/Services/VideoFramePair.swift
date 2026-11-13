import AVFoundation
import CoreImage

final class VideoFramePair {
    let id=UUID()
    let original: CVPixelBuffer
    let processed: CVPixelBuffer
    let transform: CGAffineTransform
    let time: CMTime
    let duration: CMTime
    let isHDR: Bool
    let originalSample: CMSampleBuffer?
    let processedSample: CMSampleBuffer?

    var displaySize: CGSize {
        CGRect(x:0,y:0,width:CVPixelBufferGetWidth(original),height:CVPixelBufferGetHeight(original))
            .applying(transform).size
    }

    init(original: CVPixelBuffer, processed: CVPixelBuffer, transform: CGAffineTransform, time: CMTime, isHDR: Bool,
         duration: CMTime = .invalid, originalSample: CMSampleBuffer? = nil, processedSample: CMSampleBuffer? = nil) {
        self.original=original;self.processed=processed;self.transform=transform;self.time=time;self.isHDR=isHDR
        self.duration=duration
        self.originalSample=originalSample;self.processedSample=processedSample
    }

    func image(processed: Bool) -> CIImage {
        let source=CIImage(cvPixelBuffer:processed ? self.processed:original)
        let oriented=source.transformed(by:transform)
        return oriented.transformed(by:CGAffineTransform(translationX:-oriented.extent.minX,y:-oriented.extent.minY))
    }

    static func load(result: VideoProcessingResult, seconds: Double) async throws -> VideoFramePair {
        let sourceInfo=try await VideoMediaInfo.load(result.input)
        let outputInfo=try await VideoMediaInfo.load(result.output)
        let end=max(0,result.duration-1/Double(max(1,result.frameRate)))
        let requested=CMTime(seconds:min(end,max(0,seconds)),preferredTimescale:sourceInfo.timeScale)
        let source=try readFrame(info:sourceInfo,time:requested)
        let output=try readFrame(info:outputInfo,time:source.1)
        guard abs(source.1.seconds-output.1.seconds)<0.0001 else {
            throw PipelineError.failure("原始与处理后视频的对比帧时间不一致")
        }
        return VideoFramePair(original:source.0,processed:output.0,transform:result.transform,time:source.1,isHDR:result.isHDR)
    }

    private static func readFrame(info: VideoMediaInfo, time: CMTime) throws -> (CVPixelBuffer,CMTime) {
        let reader=try AVAssetReader(asset:info.asset)
        let output=AVAssetReaderTrackOutput(track:info.track,outputSettings:[
            kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            kCVPixelBufferIOSurfacePropertiesKey as String:[:],kCVPixelBufferMetalCompatibilityKey as String:true,
            AVVideoAllowWideColorKey:true,AVVideoColorPropertiesKey:info.colors])
        output.alwaysCopiesSampleData=false
        guard reader.canAdd(output) else {throw PipelineError.failure("对比帧解码不可用")}
        reader.add(output);reader.timeRange=CMTimeRange(start:time,duration:.positiveInfinity)
        guard reader.startReading() else {throw reader.error ?? PipelineError.failure("对比帧读取失败")}
        defer {reader.cancelReading()}
        guard let sample=output.copyNextSampleBuffer(),let buffer=CMSampleBufferGetImageBuffer(sample) else {
            throw reader.error ?? PipelineError.failure("所选位置没有视频帧")
        }
        info.attachColors(to:buffer)
        return (buffer,CMSampleBufferGetPresentationTimeStamp(sample))
    }
}
