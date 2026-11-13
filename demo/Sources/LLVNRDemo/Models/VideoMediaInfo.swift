import AVFoundation
import VideoToolbox

struct VideoMediaInfo {
    let asset: AVURLAsset
    let track: AVAssetTrack
    let passthroughTracks: [(AVAssetTrack,CMFormatDescription)]
    let format: CMFormatDescription
    let width: Int, height: Int, bitDepth: Int
    let frameRate: Float, bitRate: Float
    let duration: CMTime, timeRange: CMTimeRange
    let transform: CGAffineTransform
    let timeScale: CMTimeScale
    let codec: AVVideoCodecType
    let colors: [String:String]
    let metadata: [AVMetadataItem]
    let isHDR: Bool

    static func load(_ url: URL) async throws -> VideoMediaInfo {
        let asset=AVURLAsset(url:url,options:[AVURLAssetPreferPreciseDurationAndTimingKey:true])
        guard let track=try await asset.loadTracks(withMediaType:.video).first,
              let format=try await track.load(.formatDescriptions).first else {
            throw PipelineError.failure("文件没有可读取的视频轨道")
        }
        let size=CMVideoFormatDescriptionGetDimensions(format)
        let subtype=CMFormatDescriptionGetMediaSubType(format)
        let codec: AVVideoCodecType
        switch subtype {
        case kCMVideoCodecType_HEVC,0x68657631,0x64766831,0x64766865:codec = .hevc
        case kCMVideoCodecType_H264:codec = .h264
        default:throw PipelineError.failure("当前视频处理支持 H.264 和 HEVC 编码的 MOV／MP4")
        }
        let atoms=CMFormatDescriptionGetExtension(format,extensionKey:kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms) as? [String:Data] ?? [:]
        let hevc=atoms["hvcC"]
        let inferredDepth=(hevc?.count ?? 0)>17 ? 8+Int(hevc![17]&7):8
        let depth=(CMFormatDescriptionGetExtension(format,extensionKey:kCMFormatDescriptionExtension_BitsPerComponent) as? NSNumber)?.intValue ?? inferredDepth
        guard depth<=10 else {throw PipelineError.failure("当前硬件导出路径支持 8-bit／10-bit 视频")}
        if let dovi=atoms.first(where:{$0.key.lowercased()=="dvcc" || $0.key.lowercased()=="dvvc"})?.value,
           dovi.count>2,(dovi[2]>>1)==5 {
            throw PipelineError.failure("当前支持带 HLG／PQ 兼容图像层的杜比视界片源；此文件为 Profile 5")
        }
        var colors: [String:String] = [:]
        for (key,extensionKey,fallback) in [
            (AVVideoColorPrimariesKey,kCMFormatDescriptionExtension_ColorPrimaries,AVVideoColorPrimaries_ITU_R_709_2),
            (AVVideoTransferFunctionKey,kCMFormatDescriptionExtension_TransferFunction,AVVideoTransferFunction_ITU_R_709_2),
            (AVVideoYCbCrMatrixKey,kCMFormatDescriptionExtension_YCbCrMatrix,AVVideoYCbCrMatrix_ITU_R_709_2)] {
            colors[key]=(CMFormatDescriptionGetExtension(format,extensionKey:extensionKey) as? String) ?? fallback
        }
        let transfer=colors[AVVideoTransferFunctionKey]
        let hdr=transfer==AVVideoTransferFunction_ITU_R_2100_HLG || transfer==AVVideoTransferFunction_SMPTE_ST_2084_PQ
        if track.hasMediaCharacteristic(.containsHDRVideo) && !hdr {
            throw PipelineError.failure("无法确认此 HDR 轨道的 HLG／PQ 色彩描述")
        }
        var copied: [(AVAssetTrack,CMFormatDescription)] = []
        for other in try await asset.load(.tracks) where other.trackID != track.trackID {
            guard other.mediaType != .video else {throw PipelineError.failure("当前视频处理支持单个视频轨道")}
            if let description=try await other.load(.formatDescriptions).first {copied.append((other,description))}
        }
        return VideoMediaInfo(asset:asset,track:track,passthroughTracks:copied,format:format,
            width:Int(size.width),height:Int(size.height),bitDepth:hdr ? max(10,depth):depth,
            frameRate:try await track.load(.nominalFrameRate),bitRate:try await track.load(.estimatedDataRate),
            duration:try await asset.load(.duration),timeRange:try await track.load(.timeRange),
            transform:try await track.load(.preferredTransform),timeScale:try await track.load(.naturalTimeScale),
            codec:codec,colors:colors,metadata:try await asset.load(.metadata),isHDR:hdr)
    }

    func writerSettings() -> [String:Any] {
        var compression: [String:Any] = [AVVideoExpectedSourceFrameRateKey:max(1,frameRate)]
        if bitRate>0 {compression[AVVideoAverageBitRateKey]=Int(bitRate.rounded())}
        if codec == .hevc {
            compression[AVVideoProfileLevelKey]=bitDepth>8 ? kVTProfileLevel_HEVC_Main10_AutoLevel:kVTProfileLevel_HEVC_Main_AutoLevel
        } else {compression[AVVideoProfileLevelKey]=AVVideoProfileLevelH264HighAutoLevel}
        if isHDR {
            compression[kVTCompressionPropertyKey_HDRMetadataInsertionMode as String]=kVTHDRMetadataInsertionMode_Auto
            compression[kVTCompressionPropertyKey_PreserveDynamicHDRMetadata as String]=false
        }
        return [AVVideoCodecKey:codec,AVVideoWidthKey:width,AVVideoHeightKey:height,
            AVVideoColorPropertiesKey:colors,AVVideoCompressionPropertiesKey:compression,
            AVVideoEncoderSpecificationKey:[
                kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder as String:true]]
    }

    func attachColors(to buffer: CVPixelBuffer) {
        for (key,value) in [
            (kCVImageBufferColorPrimariesKey,colors[AVVideoColorPrimariesKey]!),
            (kCVImageBufferTransferFunctionKey,colors[AVVideoTransferFunctionKey]!),
            (kCVImageBufferYCbCrMatrixKey,colors[AVVideoYCbCrMatrixKey]!)] {
            CVBufferSetAttachment(buffer,key,value as CFString,.shouldPropagate)
        }
    }
}
