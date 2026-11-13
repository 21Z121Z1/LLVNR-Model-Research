import AVFoundation

/// Comparison uses one HDR display interpretation for both images. Inference
/// and frame export continue to own their original pixel buffers separately.
final class VideoDisplaySamples {
    private var pool: CVPixelBufferPool?
    private var width = 0
    private var height = 0
    private var pixelFormat: OSType = 0

    func make(image: CVPixelBuffer, colorsFrom reference: CVPixelBuffer, time: CMTime) throws -> CMSampleBuffer {
        let imageWidth = CVPixelBufferGetWidth(image)
        let imageHeight = CVPixelBufferGetHeight(image)
        let format = CVPixelBufferGetPixelFormatType(image)
        if pool == nil || width != imageWidth || height != imageHeight || pixelFormat != format {
            let attributes: [String: Any] = [
                kCVPixelBufferWidthKey as String: imageWidth,
                kCVPixelBufferHeightKey as String: imageHeight,
                kCVPixelBufferPixelFormatTypeKey as String: format,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]
            ]
            guard CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool) == kCVReturnSuccess else {
                throw PipelineError.failure("HDR 显示像素池分配失败")
            }
            width = imageWidth
            height = imageHeight
            pixelFormat = format
        }
        var copy: CVPixelBuffer?
        let limit = [kCVPixelBufferPoolAllocationThresholdKey as String: 6] as CFDictionary
        guard let pool = pool,
              CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(nil, pool, limit, &copy) == kCVReturnSuccess,
              let copy = copy else { throw PipelineError.failure("HDR 显示缓冲暂不可用") }
        guard CVPixelBufferLockBaseAddress(image, .readOnly) == kCVReturnSuccess else {
            throw PipelineError.failure("原始显示像素无法读取")
        }
        defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
        guard CVPixelBufferLockBaseAddress(copy, []) == kCVReturnSuccess else {
            throw PipelineError.failure("显示像素无法写入")
        }
        defer { CVPixelBufferUnlockBaseAddress(copy, []) }
        for plane in 0..<CVPixelBufferGetPlaneCount(image) {
            guard let source = CVPixelBufferGetBaseAddressOfPlane(image, plane),
                  let destination = CVPixelBufferGetBaseAddressOfPlane(copy, plane) else {
                throw PipelineError.failure("显示像素平面不可用")
            }
            let sourceStride = CVPixelBufferGetBytesPerRowOfPlane(image, plane)
            let destinationStride = CVPixelBufferGetBytesPerRowOfPlane(copy, plane)
            let rowBytes = CVPixelBufferGetWidthOfPlane(image, plane) * (plane == 0 ? 2 : 4)
            for row in 0..<CVPixelBufferGetHeightOfPlane(image, plane) {
                memcpy(destination.advanced(by: row * destinationStride),
                       source.advanced(by: row * sourceStride), rowBytes)
            }
        }
        CVBufferRemoveAllAttachments(copy)
        // Shared source colorimetry and mastering conditions keep the two sides
        // comparable. Dolby per-frame metadata refers to the unedited pixels
        // and is deliberately excluded from this editing-preview copy.
        for key in [kCVImageBufferColorPrimariesKey, kCVImageBufferTransferFunctionKey,
                    kCVImageBufferYCbCrMatrixKey, kCVImageBufferChromaLocationTopFieldKey,
                    kCVImageBufferChromaLocationBottomFieldKey, kCVImageBufferMasteringDisplayColorVolumeKey,
                    kCVImageBufferContentLightLevelInfoKey, kCVImageBufferAmbientViewingEnvironmentKey] {
            if let value = CVBufferCopyAttachment(reference, key, nil) {
                CVBufferSetAttachment(copy, key, value, .shouldPropagate)
            }
        }
        var description: CMVideoFormatDescription?
        var sample: CMSampleBuffer?
        var timing = CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: time, decodeTimeStamp: .invalid)
        guard CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: copy,
                  formatDescriptionOut: &description) == noErr, let description = description,
              CMSampleBufferCreateReadyWithImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: copy,
                  formatDescription: description, sampleTiming: &timing, sampleBufferOut: &sample) == noErr,
              let sample = sample else { throw PipelineError.failure("HDR 显示 sample buffer 创建失败") }
        if let attachments = CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: true) as? [NSMutableDictionary] {
            attachments.first?[kCMSampleAttachmentKey_DisplayImmediately] = true
        }
        return sample
    }
}
