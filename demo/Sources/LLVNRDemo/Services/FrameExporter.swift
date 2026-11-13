import Foundation
import Metal
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import CoreImage

enum FrameImageFormat: String, CaseIterable, Identifiable {
    case heif, jpeg, png
    var id: String {rawValue}
    var title: String {rawValue.uppercased()}
    var fileExtension: String {self == .heif ? "heic":(self == .jpeg ? "jpg":"png")}
    var type: UTType {self == .heif ? .heic:(self == .jpeg ? .jpeg:.png)}
}

enum FrameExporter {
    private static let work=DispatchQueue(label:"llvnr.frame.export",qos:.userInitiated)

    static func save(_ frame: DisplayFrame, in parent: URL, format: FrameImageFormat,
                     quality: Double, completion: @escaping (Result<URL,Error>)->Void) {
        // The selected pair stays alive even if playback replaces the preview
        // while the folder picker or PNG encoder is running.
        work.async {
            defer {frame.endRead()}
            do {
                guard let queue=frame.texture.device.makeCommandQueue() else {
                    throw PipelineError.failure("帧保存队列不可用")
                }
                let original=try encode(frame.originalTexture,queue:queue,format:format,quality:quality)
                let processed=try encode(frame.texture,queue:queue,format:format,quality:quality)
                let destination=try writePair(original:original,processed:processed,parent:parent,format:format)
                DispatchQueue.main.async{completion(.success(destination))}
            } catch {DispatchQueue.main.async{completion(.failure(error))}}
        }
    }

    static func save(_ pair: VideoFramePair, in parent: URL, format: FrameImageFormat,
                     quality: Double, completion: @escaping (Result<URL,Error>)->Void) {
        work.async {
            do {
                let context=CIContext(options:[.workingFormat:CIFormat.RGBAh,
                    .workingColorSpace:CGColorSpace(name:CGColorSpace.extendedLinearITUR_2020)!])
                let original=try encode(pair.image(processed:false),context:context,hdr:pair.isHDR,format:format,quality:quality)
                let processed=try encode(pair.image(processed:true),context:context,hdr:pair.isHDR,format:format,quality:quality)
                let destination=try writePair(original:original,processed:processed,parent:parent,format:format)
                DispatchQueue.main.async{completion(.success(destination))}
            } catch {DispatchQueue.main.async{completion(.failure(error))}}
        }
    }

    private static func writePair(original: Data, processed: Data, parent: URL, format: FrameImageFormat) throws -> URL {
        let formatter=DateFormatter();formatter.dateFormat="yyyyMMdd-HHmmss-SSS"
        let name="LLVNR-"+formatter.string(from:Date())+"-"+UUID().uuidString.prefix(6)
        let destination=parent.appendingPathComponent(name,isDirectory:true)
        let staging=parent.appendingPathComponent(".llvnr-export-"+UUID().uuidString,isDirectory:true)
        try FileManager.default.createDirectory(at:staging,withIntermediateDirectories:false)
        do {
            try original.write(to:staging.appendingPathComponent("原始画面."+format.fileExtension),options:.atomic)
            try processed.write(to:staging.appendingPathComponent("处理后."+format.fileExtension),options:.atomic)
            try FileManager.default.moveItem(at:staging,to:destination)
        } catch {try? FileManager.default.removeItem(at:staging);throw error}
        return destination
    }

    private static func encode(_ image: CIImage, context: CIContext, hdr: Bool,
                               format: FrameImageFormat, quality: Double) throws -> Data {
        let key=CIImageRepresentationOption(rawValue:kCGImageDestinationLossyCompressionQuality as String)
        let options: [CIImageRepresentationOption:Any] = [key:min(1,max(0,quality))]
        let sRGB=CGColorSpace(name:CGColorSpace.sRGB)!
        let pq=CGColorSpace(name:CGColorSpace.itur_2100_PQ)!
        if format == .heif && hdr {
            let temporary=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".heic")
            defer {try? FileManager.default.removeItem(at:temporary)}
            try context.writeHEIF10Representation(of:image,to:temporary,colorSpace:pq,options:options)
            return try Data(contentsOf:temporary)
        }
        if format == .heif,let data=context.heifRepresentation(of:image,format:.RGBA8,colorSpace:sRGB,options:options) {return data}
        if format == .png,let data=context.pngRepresentation(of:image,format:hdr ? .RGBA16:.RGBA8,colorSpace:hdr ? pq:sRGB,options:[:]) {return data}
        if format == .jpeg {
            var output=image
            if hdr {
                guard #available(macOS 15.0,*),let filter=CIFilter(name:"CIToneMapHeadroom") else {
                    throw PipelineError.failure("HDR 帧的 JPEG 色调映射需要 macOS 15 或更新版本")
                }
                filter.setValue(image,forKey:kCIInputImageKey)
                filter.setValue(1,forKey:"inputTargetHeadroom")
                guard let mapped=filter.outputImage else {throw PipelineError.failure("HDR 色调映射失败")}
                output=mapped
            }
            if let data=context.jpegRepresentation(of:output,colorSpace:sRGB,options:options) {return data}
        }
        throw PipelineError.failure("\(format.title) 编码失败")
    }

    private static func encode(_ texture: MTLTexture, queue: MTLCommandQueue,
                               format: FrameImageFormat, quality: Double) throws -> Data {
        let stride=texture.width*4
        guard let buffer=texture.device.makeBuffer(length:stride*texture.height,options:.storageModeShared),
              let command=queue.makeCommandBuffer(),let encoder=command.makeBlitCommandEncoder() else {
            throw PipelineError.failure("图像导出缓冲不可用")
        }
        encoder.copy(from:texture,sourceSlice:0,sourceLevel:0,sourceOrigin:MTLOrigin(),
            sourceSize:MTLSize(width:texture.width,height:texture.height,depth:1),to:buffer,
            destinationOffset:0,destinationBytesPerRow:stride,destinationBytesPerImage:buffer.length)
        encoder.endEncoding();command.commit();command.waitUntilCompleted()
        if let error=command.error {throw error}
        let data=Data(bytes:buffer.contents(),count:buffer.length)
        guard let provider=CGDataProvider(data:data as CFData),let colorSpace=CGColorSpace(name:CGColorSpace.sRGB),
              let image=CGImage(width:texture.width,height:texture.height,bitsPerComponent:8,bitsPerPixel:32,
                bytesPerRow:stride,space:colorSpace,
                bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.premultipliedFirst.rawValue|CGBitmapInfo.byteOrder32Little.rawValue),
                provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent) else {
            throw PipelineError.failure("图像编码准备失败")
        }
        let encoded=NSMutableData()
        guard let destination=CGImageDestinationCreateWithData(encoded,format.type.identifier as CFString,1,nil) else {
            throw PipelineError.failure("\(format.title) 编码器不可用")
        }
        let properties: [CFString:Any] = format == .png ? [:]:[kCGImageDestinationLossyCompressionQuality:min(1,max(0,quality))]
        CGImageDestinationAddImage(destination,image,properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {throw PipelineError.failure("\(format.title) 保存失败")}
        return encoded as Data
    }
}
