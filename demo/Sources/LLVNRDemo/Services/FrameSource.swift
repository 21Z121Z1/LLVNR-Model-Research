import Foundation
import AVFoundation
import CoreVideo

protocol FrameSource: AnyObject {
    func start() throws
    func stop()
    func setPaused(_ paused: Bool, completion: @escaping (Error?)->Void)
}

struct CameraInput: Identifiable, Equatable {
    let id: String
    let name: String
}

final class CameraSource: NSObject, FrameSource, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let session=AVCaptureSession()
    private let delegateQueue=DispatchQueue(label:"llvnr.capture",qos:.userInteractive)
    private let sessionQueue=DispatchQueue(label:"llvnr.camera.session",qos:.userInitiated)
    private var nextID=0
    private let receive: (SourceFrame)->Void
    private let dropped: ([String:Any])->Void
    private let ended: ()->Void
    private var stopped=false
    private var paused=false
    private let deviceID: String?

    static func devices() -> [AVCaptureDevice] {
        AVCaptureDevice.DiscoverySession(deviceTypes:[.builtInWideAngleCamera,.external,.continuityCamera,.deskViewCamera],
            mediaType:.video,position:.unspecified).devices.filter(\.isConnected)
    }
    static func inputs() -> [CameraInput] {
        devices().map{CameraInput(id:$0.uniqueID,name:$0.localizedName)}
    }

    init(deviceID: String?=nil, receive: @escaping (SourceFrame)->Void,dropped: @escaping ([String:Any])->Void,ended: @escaping ()->Void) {
        self.deviceID=deviceID;self.receive=receive;self.dropped=dropped;self.ended=ended
    }
    func start() throws {
        guard AVCaptureDevice.authorizationStatus(for:.video) == .authorized else {throw PipelineError.failure("需要系统相机权限；请在系统弹窗或隐私与安全性中允许 LLVNR Demo")}
        let devices=Self.devices()
        let selected=deviceID.flatMap{id in devices.first{$0.uniqueID==id}}
        guard let device=selected ?? (deviceID==nil ? devices.first:nil) else {throw PipelineError.failure("所选摄像头不可用，请重新选择摄像头")}
        guard let format=device.formats.first(where:{f in
            let s=CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            return s.width==1920 && s.height==1080 && f.videoSupportedFrameRateRanges.contains{$0.minFrameRate<=30 && $0.maxFrameRate>=30}
        }) else {throw PipelineError.failure("所选摄像头不支持 1920×1080 @30，请选择其他摄像头")}
        session.beginConfiguration()
        var committed=false
        session.sessionPreset = .hd1920x1080
        do {
            let input=try AVCaptureDeviceInput(device:device)
            guard session.canAddInput(input) else {throw PipelineError.failure("无法添加相机输入")}
            session.addInput(input)
            try device.lockForConfiguration()
            device.activeFormat=format
            device.activeVideoMinFrameDuration=CMTime(value:1,timescale:30)
            device.activeVideoMaxFrameDuration=CMTime(value:1,timescale:30)
            device.unlockForConfiguration()
            let output=AVCaptureVideoDataOutput()
            // Capture callbacks must stay current during transient system load.
            // The delegate records every dropped-frame callback in RunMetrics.
            output.alwaysDiscardsLateVideoFrames=true
            let pixelFormat=output.availableVideoPixelFormatTypes.contains(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
                ? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            guard output.availableVideoPixelFormatTypes.contains(pixelFormat) else {throw PipelineError.failure("所选摄像头不能输出双平面 YUV420")}
            output.videoSettings=[kCVPixelBufferPixelFormatTypeKey as String:pixelFormat,
                kCVPixelBufferMetalCompatibilityKey as String:true]
            output.setSampleBufferDelegate(self,queue:delegateQueue)
            guard session.canAddOutput(output) else {throw PipelineError.failure("无法添加相机 YUV 输出")}
            session.addOutput(output)
            session.commitConfiguration();committed=true
            session.startRunning()
            guard session.isRunning else {throw PipelineError.failure("相机启动失败")}
        } catch {
            if !committed {session.commitConfiguration()}
            throw error
        }
    }
    func captureOutput(_ output: AVCaptureOutput,didOutput sampleBuffer: CMSampleBuffer,from connection: AVCaptureConnection) {
        guard !stopped,!paused,let buffer=CMSampleBufferGetImageBuffer(sampleBuffer) else {return}
        let arrival=hostSeconds(),pts=CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let time=session.synchronizationClock.map{CMSyncConvertTime(pts,from:$0,to:CMClockGetHostTimeClock()).seconds} ?? pts.seconds
        let seq=(CMGetAttachment(sampleBuffer,key:"com.apple.cmio.buffer_attachment.sequence_number" as CFString,attachmentModeOut:nil) as? NSNumber)?.int64Value
        var metadata: [String:Any] = [:]
        if let m=CMGetAttachment(sampleBuffer,key:"MetadataDictionary" as CFString,attachmentModeOut:nil) as? [String:Any] {
            for name in ["LuxLevel","NormalizedSNR","AEAverage","AGC","ispDGain","sensorDGain","ReadNoise_1x","ReadNoise_8x"] {
                if let value=m[name] as? NSNumber {metadata[name]=value}
            }
        }
        let frame=SourceFrame(buffer:buffer,id:nextID,sequence:seq,pts:pts.seconds,received:arrival,sourceTime:time,metadata:metadata)
        nextID+=1;receive(frame)
    }
    func captureOutput(_ output: AVCaptureOutput,didDrop sampleBuffer: CMSampleBuffer,from connection: AVCaptureConnection) {
        guard !stopped,!paused else{return}
        let reason=CMGetAttachment(sampleBuffer,key:kCMSampleBufferAttachmentKey_DroppedFrameReason,attachmentModeOut:nil)
        dropped(["pts":CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds,"reason":String(describing:reason)])
    }
    func stop() {
        sessionQueue.async { [self] in
            if session.isRunning {session.stopRunning()}
            delegateQueue.async { [self] in
                guard !stopped else {return};stopped=true;ended()
            }
        }
    }
    func setPaused(_ next: Bool, completion: @escaping (Error?)->Void) {
        sessionQueue.async { [self] in
            let ended=delegateQueue.sync{stopped}
            guard !ended else {DispatchQueue.main.async{completion(nil)};return}
            delegateQueue.sync{paused=next}
            if next {if session.isRunning {session.stopRunning()}}
            else {if !session.isRunning {session.startRunning()}}
            let error: Error? = !next && !session.isRunning ? PipelineError.failure("摄像头恢复失败，正在重试"):nil
            DispatchQueue.main.async{completion(error)}
        }
    }
}

final class ReplaySource: FrameSource {
    private let queue=DispatchQueue(label:"llvnr.replay.source",qos:.userInteractive)
    private var timer: DispatchSourceTimer?
    private var frames: [CVPixelBuffer] = []
    private var nextID=0
    private let target: Int
    private let receive: (SourceFrame)->Void
    private let ended: ()->Void
    private var epoch=0.0
    private var pausedAt: Double?
    init(directory: URL,count: Int,receive: @escaping (SourceFrame)->Void,ended: @escaping ()->Void) throws {
        self.receive=receive;self.ended=ended;target=count>0 ? count : 1800
        let folder=directory.appendingPathComponent("frames")
        let names=try FileManager.default.contentsOfDirectory(atPath:folder.path).filter{$0.hasSuffix("_p0.420v.bin")}.sorted()
        guard !names.isEmpty else {throw PipelineError.failure("重放目录没有 1080p 420v 帧")}
        for name in names {frames.append(try Self.load(directory:folder,basename:String(name.dropLast("_p0.420v.bin".count))))}
    }
    static func load(directory: URL,basename: String) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [String:Any] = [kCVPixelBufferIOSurfacePropertiesKey as String:[:],kCVPixelBufferMetalCompatibilityKey as String:true]
        guard CVPixelBufferCreate(nil,1920,1080,kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,attrs as CFDictionary,&pb)==kCVReturnSuccess,let buffer=pb else {throw PipelineError.failure("重放 YUV 缓冲分配失败")}
        CVPixelBufferLockBaseAddress(buffer,[])
        defer {CVPixelBufferUnlockBaseAddress(buffer,[])}
        for plane in 0..<2 {
            let data=try Data(contentsOf:directory.appendingPathComponent("\(basename)_p\(plane).420v.bin"))
            let height=plane==0 ? 1080 : 540
            guard data.count==1920*height,let base=CVPixelBufferGetBaseAddressOfPlane(buffer,plane) else {throw PipelineError.failure("重放帧尺寸不符")}
            let stride=CVPixelBufferGetBytesPerRowOfPlane(buffer,plane)
            data.withUnsafeBytes { bytes in
                for y in 0..<height {memcpy(base.advanced(by:y*stride),bytes.baseAddress!.advanced(by:y*1920),1920)}
            }
        }
        CVBufferSetAttachment(buffer,kCVImageBufferYCbCrMatrixKey,kCVImageBufferYCbCrMatrix_ITU_R_709_2,.shouldPropagate)
        return buffer
    }
    func start() throws {
        let source=DispatchSource.makeTimerSource(queue:queue);timer=source
        epoch=hostSeconds()+0.2
        source.schedule(deadline:.now() + .milliseconds(200),repeating:.nanoseconds(33_333_333),leeway:.microseconds(100))
        source.setEventHandler { [self] in
            guard pausedAt==nil else{return}
            guard nextID<target else {finish();return}
            let now=hostSeconds(),pts=Double(nextID)/30
            receive(SourceFrame(buffer:frames[nextID%frames.count],id:nextID,sequence:Int64(nextID),pts:pts,received:now,sourceTime:epoch+pts))
            nextID+=1
            if nextID==target {finish()}
        }
        source.resume()
    }
    private func finish() {guard let t=timer else {return};t.cancel();timer=nil;ended()}
    func stop() {queue.async{[self] in finish()}}
    func setPaused(_ next: Bool, completion: @escaping (Error?)->Void) {
        queue.async { [self] in
            if next {if pausedAt==nil {pausedAt=hostSeconds()}}
            else if let start=pausedAt {epoch+=hostSeconds()-start;pausedAt=nil}
            DispatchQueue.main.async{completion(nil)}
        }
    }
}
