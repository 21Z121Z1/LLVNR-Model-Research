import Foundation
import CoreMedia
import CoreVideo
import Metal

enum ComparisonMode: Int, Codable, CaseIterable, Identifiable {
    // Keep persisted raw values while presenting the original image first.
    case original = 1, processed = 0, wipe = 2, sideBySide = 3
    var id: Int {rawValue}
    var title: String {
        switch self {
        case .processed:return "处理后"
        case .original:return "原始画面"
        case .wipe:return "滑动对比"
        case .sideBySide:return "并排对比"
        }
    }
}
struct Tuning: Codable, Equatable {
    var dnr: Float = 1.05208
    var fusion: Float = 1
    /// Contribution of the new state in H(next)=(1-w)H+w*state_yuv.
    var historyWeight: Float = 0.5
    var split: Float = 0.5
    var comparison: ComparisonMode = .processed

    init() {}

    func hasSameProcessing(as other: Tuning) -> Bool {
        dnr == other.dnr && fusion == other.fusion && historyWeight == other.historyWeight
    }

    private enum CodingKeys: String, CodingKey {
        case dnr, fusion, historyWeight, split, comparison
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        dnr = try values.decodeIfPresent(Float.self, forKey: .dnr) ?? 1.05208
        fusion = try values.decodeIfPresent(Float.self, forKey: .fusion) ?? 1
        historyWeight = try values.decodeIfPresent(Float.self, forKey: .historyWeight) ?? 0.5
        split = try values.decodeIfPresent(Float.self, forKey: .split) ?? 0.5
        comparison = try values.decodeIfPresent(ComparisonMode.self, forKey: .comparison) ?? .processed
    }
}

struct ComparisonViewport: Equatable {
    var zoom: Float = 1
    var centerX: Float = 0.5
    var centerY: Float = 0.5

    mutating func clamp() {
        zoom = min(8, max(1, zoom))
        centerX = min(1 - 0.25 / zoom, max(0.25 / zoom, centerX))
        centerY = min(1 - 0.5 / zoom, max(0.5 / zoom, centerY))
    }
}

struct DisplayParameters: Equatable {
    var mode: UInt32 = 0
    var split: Float = 0.5
    var zoom: Float = 1
    var centerX: Float = 0.5
    var centerY: Float = 0.5
}

struct MetalParameters {
    var width: UInt32 = 1920, height: UInt32 = 1080
    var packedWidth: UInt32 = 1152, packedHeight: UInt32 = 648
    var mapWidth: UInt32 = 576, mapHeight: UInt32 = 324
    var reset: UInt32 = 0, fullRange: UInt32 = 0
    var originX: UInt32 = 192, originY: UInt32 = 112, matrix601: UInt32 = 0, reserved: UInt32 = 0
    var dnr: Float = 1.05208, fusion: Float = 1, historyWeight: Float = 0.5, split: Float = 0
}

struct ModelManifest: Decodable {
    struct Port: Decodable { var name: String; var port: String; var shape: [Int]; var bytes: Int }
    var version: Int
    var width: Int, height: Int, padded_height: Int, padded_width: Int, origin_x: Int, origin_y: Int, fps: Int
    var node_count: Int, coefficient_group_count: Int
    var source_model_sha256: String
    var source_graph_sha256: String
    var execution_profile: String?
    var files: [String: String]
    var inputs: [Port], outputs: [Port]
    var ane_inputs: [Port]?, ane_outputs: [Port]?
    func metalParameters() -> MetalParameters {
        var p=MetalParameters()
        p.width=UInt32(width);p.height=UInt32(height)
        p.packedWidth=UInt32(padded_width/2);p.packedHeight=UInt32(padded_height/2)
        p.mapWidth=UInt32(padded_width/4);p.mapHeight=UInt32(padded_height/4)
        p.originX=UInt32(origin_x);p.originY=UInt32(origin_y)
        return p
    }
}

enum PipelineError: LocalizedError {
    case failure(String)
    var errorDescription: String? { if case let .failure(message) = self { return message }; return nil }
}

func hostSeconds() -> Double { CMClockGetTime(CMClockGetHostTimeClock()).seconds }

final class SourceFrame {
    let pixelBuffer: CVPixelBuffer
    let id: Int
    let sourceSequence: Int64?
    let pts: Double
    let received: Double
    let sourceTime: Double
    let metadata: [String: Any]
    init(buffer: CVPixelBuffer, id: Int, sequence: Int64?, pts: Double,
         received: Double, sourceTime: Double, metadata: [String: Any] = [:]) {
        pixelBuffer=buffer;self.id=id;sourceSequence=sequence;self.pts=pts
        self.received=received;self.sourceTime=sourceTime;self.metadata=metadata
    }
}

struct FrameTiming: Codable {
    var id: Int, sourceSequence: Int64?
    var sourcePTS: Double, sourceHostTime: Double, received: Double, inferenceStart: Double
    var packed: Double, inferred: Double, postprocessed: Double
    var reset: Bool, tuning: Tuning
    var presented: Double?
    var queueMS: Double { (inferenceStart-received)*1000 }
    var inferenceMS: Double { (inferred-packed)*1000 }
    var arrivalToDisplayMS: Double? { presented.map { ($0-received)*1000 } }
    var sourceToDisplayMS: Double? { presented.map { ($0-sourceHostTime)*1000 } }
}

final class DisplayFrame {
    let presentationID=UUID()
    let isPreview: Bool
    let texture: MTLTexture
    let originalTexture: MTLTexture
    let source: SourceFrame
    var timing: FrameTiming
    private let releaseTexture: () -> Void
    private let lock=NSLock()
    private var retired=false,recycled=false,completed=false
    private var readers=0
    init(textures: OutputTextures, source: SourceFrame, timing: FrameTiming, isPreview: Bool=false, recycle: @escaping () -> Void) {
        self.isPreview=isPreview
        self.texture=textures.processed;self.originalTexture=textures.original;self.source=source;self.timing=timing;self.releaseTexture=recycle
    }
    func beginRead() {
        lock.lock();readers+=1;lock.unlock()
    }
    func endRead() {
        lock.lock();readers-=1
        let release=retired && readers==0 && !recycled
        if release {recycled=true}
        lock.unlock()
        if release {releaseTexture()}
    }
    func recycle() {
        lock.lock();retired=true
        let release=readers==0 && !recycled
        if release {recycled=true}
        lock.unlock()
        if release {releaseTexture()}
    }
    func claimCompletion() -> Bool {lock.lock();defer{lock.unlock()};if completed{return false};completed=true;return true}
}

struct OutputTextures {let processed: MTLTexture;let original: MTLTexture}

struct MetricsSnapshot {
    var source = 0, processed = 0, presented = 0, sourceGaps = 0, captureDrops = 0
    var overflow = 0, failures = 0, queueDepth = 0, maxQueueDepth = 0
    var supersededInputs = 0, supersededDisplays = 0
    var recoveries = 0
    var lastError = ""
    var processedFPS = 0.0, presentedFPS = 0.0
    var inferenceP95 = 0.0, arrivalLatencyP50 = 0.0, arrivalLatencyP95 = 0.0
    var sourceLatencyP95 = 0.0
    var runDirectory = ""
    var verdict = "等待测量"
}

struct RunOptions {
    var modelID: String?
    var cameraID: String?
    var videoInput: URL?
    var videoOutput: URL?
    var videoFrameDirectory: URL?
    var multiPass=false
    var replayDirectory: URL?
    var replayFrames = 0
    var reportDirectory: URL?
    var validationDirectory: URL?
    var autoStart = false
    var quitAfterRun = false
    var cameraSeconds = 0.0
    var useDefaultTuning = false
    static func parse() -> RunOptions {
        var o=RunOptions(); var it=CommandLine.arguments.dropFirst().makeIterator()
        while let a=it.next() {
            switch a {
            case "--model": o.modelID=it.next()
            case "--camera": o.cameraID=it.next()
            case "--video": o.videoInput=it.next().map{URL(fileURLWithPath:$0)}
            case "--video-output": o.videoOutput=it.next().map{URL(fileURLWithPath:$0)}
            case "--save-video-frame": o.videoFrameDirectory=it.next().map{URL(fileURLWithPath:$0)}
            case "--multi-pass": o.multiPass=true
            case "--replay": o.replayDirectory=URL(fileURLWithPath:it.next() ?? "")
            case "--frames": o.replayFrames=Int(it.next() ?? "0") ?? 0
            case "--report": o.reportDirectory=URL(fileURLWithPath:it.next() ?? "")
            case "--validate": o.validationDirectory=URL(fileURLWithPath:it.next() ?? "")
            case "--autostart": o.autoStart=true
            case "--quit-after-run": o.quitAfterRun=true
            case "--camera-seconds": o.cameraSeconds=Double(it.next() ?? "0") ?? 0
            case "--default-tuning": o.useDefaultTuning=true
            default: break
            }
        }
        return o
    }
}
