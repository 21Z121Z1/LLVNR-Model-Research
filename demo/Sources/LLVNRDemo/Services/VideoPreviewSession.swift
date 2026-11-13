import AVFoundation
import Metal

/// A single inference queue serves the latest seek before advancing the
/// playback look-ahead. Checkpoints belong to one model and one set of inputs.
final class VideoPreviewSession {
    let modelID: String
    var onFrame: ((VideoFramePair) -> Void)?
    var onProgress: ((Double, Double) -> Void)?
    var onError: ((Error) -> Void)?

    private struct Request {
        let revision: Int
        let seconds: Double
        let tuning: Tuning
    }
    private struct Checkpoint {
        let seconds: Double
        let history: VideoFrameProcessor.HistorySnapshot
    }
    private let work = DispatchQueue(label: "llvnr.video.preview", qos: .userInitiated)
    private let lock = NSLock()
    private let exportedModelID: String
    private let exportedTuning: Tuning
    private var pending: Request?
    private var revision = 0
    private var stopped = false
    private var running = false
    private var ready = false
    private var sourceInfo: VideoMediaInfo?
    private var outputInfo: VideoMediaInfo?
    private var sourceReader: AVAssetReader?
    private var sourceOutput: AVAssetReaderTrackOutput?
    private var resultReader: AVAssetReader?
    private var resultOutput: AVAssetReaderTrackOutput?
    private var processor: VideoFrameProcessor?
    private var activeTuning: Tuning?
    private var usesExport = true
    private var frames: [VideoFramePair] = []
    private var checkpoints: [Checkpoint] = []
    private var lastTime = -Double.infinity
    private var requestedTime = 0.0
    private var lastProgress = 0.0
    private var reachedEnd = false
    private let frameBudget = 256 * 1024 * 1024
    private let historyBudget = 128 * 1024 * 1024

    init(result: VideoProcessingResult, modelID: String,
         exportedModelID: String, exportedTuning: Tuning) {
        self.modelID = modelID
        self.exportedModelID = exportedModelID
        self.exportedTuning = exportedTuning
        Task { [weak self] in
            do {
                let source = try await VideoMediaInfo.load(result.input)
                let output = try await VideoMediaInfo.load(result.output)
                self?.work.async { [weak self] in
                    guard let self = self else { return }
                    self.sourceInfo = source
                    self.outputInfo = output
                    self.lock.lock()
                    self.ready = true
                    let start = !self.stopped && !self.running && self.pending != nil
                    if start { self.running = true }
                    self.lock.unlock()
                    if start { self.drain() }
                }
            } catch { self?.publish(error) }
        }
    }

    func request(seconds: Double, tuning: Tuning) {
        lock.lock()
        guard !stopped else { lock.unlock(); return }
        revision += 1
        pending = Request(revision: revision, seconds: seconds, tuning: tuning)
        let start = ready && !running
        if start { running = true }
        lock.unlock()
        if start { work.async { [self] in drain() } }
    }

    func stop(completion: @escaping () -> Void = {}) {
        lock.lock()
        stopped = true
        revision += 1
        pending = nil
        lock.unlock()
        work.async { [self] in
            sourceReader?.cancelReading()
            resultReader?.cancelReading()
            sourceReader = nil
            resultReader = nil
            sourceOutput = nil
            resultOutput = nil
            frames.removeAll()
            checkpoints.removeAll()
            processor = nil
            DispatchQueue.main.async(execute: completion)
        }
    }

    private func isCurrent(_ request: Request) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return !stopped && request.revision == revision
    }

    private func drain() {
        while true {
            lock.lock()
            let request = pending
            pending = nil
            let shouldStop = stopped
            lock.unlock()
            if shouldStop { break }
            do {
                if let request = request {
                    if let pair = try render(request), isCurrent(request) {
                        DispatchQueue.main.async { [weak self] in
                            guard let self = self, self.isCurrent(request) else { return }
                            self.onFrame?(pair)
                        }
                    }
                } else if canPrefetch {
                    _ = try nextFrame()
                } else {
                    lock.lock()
                    if pending == nil { running = false; lock.unlock(); return }
                    lock.unlock()
                }
            } catch {
                reachedEnd = true
                publish(error)
            }
        }
        lock.lock(); running = false; lock.unlock()
    }

    private func render(_ request: Request) throws -> VideoFramePair? {
        guard let info = sourceInfo else { return nil }
        let end = CMTimeRangeGetEnd(info.timeRange).seconds
        let target = min(max(0, end - 0.000001), max(0, request.seconds))
        requestedTime = target
        if activeTuning?.hasSameProcessing(as: request.tuning) != true {
            activeTuning = request.tuning
            frames.removeAll()
            checkpoints.removeAll()
            usesExport = modelID == exportedModelID && request.tuning.hasSameProcessing(as: exportedTuning)
            if !usesExport && processor == nil {
                guard let device = MTLCreateSystemDefaultDevice() else {
                    throw PipelineError.failure("Metal 设备不可用")
                }
                let model = try ModelRuntime(device: device, modelID: modelID)
                try model.warmup()
                processor = try VideoFrameProcessor(model: model, width: info.width, height: info.height)
            }
            try restart(at: 0, history: nil)
        }
        if let cached = frames.first(where: { contains($0, seconds: target) }) { return cached }
        if target < lastTime || reachedEnd {
            let checkpoint = checkpoints.last(where: { $0.seconds <= target })
            try restart(at: usesExport ? target : (checkpoint?.seconds ?? 0), history: checkpoint?.history)
        } else if usesExport && target - lastTime > 0.5 {
            try restart(at: target, history: nil)
        }
        while isCurrent(request) {
            guard let pair = try nextFrame() else { return frames.last }
            if contains(pair, seconds: target) || pair.time.seconds >= target { return pair }
            if hostSeconds() - lastProgress > 0.1 {
                lastProgress = hostSeconds()
                DispatchQueue.main.async { [weak self] in
                    guard let self = self, self.isCurrent(request) else { return }
                    self.onProgress?(pair.time.seconds, target)
                }
            }
        }
        return nil
    }

    private func contains(_ pair: VideoFramePair, seconds: Double) -> Bool {
        pair.time.seconds <= seconds + 0.000001 && seconds < pair.time.seconds + frameDuration(pair) - 0.000001
    }

    private func frameDuration(_ pair: VideoFramePair) -> Double {
        pair.duration.isNumeric && pair.duration.seconds > 0
            ? pair.duration.seconds : 1 / Double(max(1, sourceInfo?.frameRate ?? 30))
    }

    private var canPrefetch: Bool {
        guard !reachedEnd, sourceReader != nil, let info=sourceInfo else { return false }
        let capacity=max(2,frameBudget / max(1,info.width * info.height * 6))
        let lookAhead=min(0.25,Double(max(1,capacity / 2-1)) / Double(max(1,info.frameRate)))
        return lastTime < requestedTime + lookAhead
    }

    private func reader(_ info: VideoMediaInfo, start: Double) throws -> (AVAssetReader, AVAssetReaderTrackOutput) {
        let reader = try AVAssetReader(asset: info.asset)
        let output = AVAssetReaderTrackOutput(track: info.track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferMetalCompatibilityKey as String: true,
            AVVideoAllowWideColorKey: true,
            AVVideoColorPropertiesKey: info.colors
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw PipelineError.failure("视频预览解码不可用") }
        reader.add(output)
        reader.timeRange = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 1_000_000),
                                       duration: .positiveInfinity)
        guard reader.startReading() else { throw reader.error ?? PipelineError.failure("视频预览读取失败") }
        return (reader, output)
    }

    private func restart(at seconds: Double, history: VideoFrameProcessor.HistorySnapshot?) throws {
        sourceReader?.cancelReading()
        resultReader?.cancelReading()
        guard let sourceInfo = sourceInfo else { return }
        (sourceReader, sourceOutput) = try reader(sourceInfo, start: seconds)
        if usesExport, let outputInfo = outputInfo {
            (resultReader, resultOutput) = try reader(outputInfo, start: seconds)
        } else {
            resultReader = nil
            resultOutput = nil
            if let history = history { processor?.restoreHistory(history) }
            else { processor?.reset() }
        }
        lastTime = seconds - 1
        reachedEnd = false
    }

    private func nextFrame() throws -> VideoFramePair? {
        try autoreleasepool {
            guard let info = sourceInfo, let tuning = activeTuning else { return nil }
            guard let sample = sourceOutput?.copyNextSampleBuffer(),
                  let original = CMSampleBufferGetImageBuffer(sample) else {
                if sourceReader?.status == .failed {
                    throw sourceReader?.error ?? PipelineError.failure("视频预览解码失败")
                }
                reachedEnd = true
                return nil
            }
            let time = CMSampleBufferGetPresentationTimeStamp(sample)
            let processed: CVPixelBuffer
            var processedSample: CMSampleBuffer?
            if usesExport {
                guard let result = resultOutput?.copyNextSampleBuffer(),
                      abs(CMSampleBufferGetPresentationTimeStamp(result).seconds - time.seconds) < 0.0001,
                      let image = CMSampleBufferGetImageBuffer(result) else {
                    throw PipelineError.failure("原始与处理后视频的对比帧时间不一致")
                }
                processed = image
                processedSample = result
            } else {
                guard let processor = processor else { throw PipelineError.failure("视频预览模型尚未就绪") }
                if time.seconds>0.2 && !checkpoints.contains(where:{abs($0.seconds-time.seconds)<0.24}) {
                    checkpoints.append(Checkpoint(seconds: time.seconds, history: processor.snapshotHistory()))
                    checkpoints.sort {$0.seconds<$1.seconds}
                    while checkpoints.reduce(0, { $0 + $1.history.bytes }) > historyBudget {
                        let distant=checkpoints.indices.max {abs(checkpoints[$0].seconds-requestedTime)<abs(checkpoints[$1].seconds-requestedTime)}!
                        checkpoints.remove(at:distant)
                    }
                }
                var buffer: CVPixelBuffer?
                let attributes: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                                               kCVPixelBufferMetalCompatibilityKey as String: true]
                guard CVPixelBufferCreate(nil, info.width, info.height,
                        kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, attributes as CFDictionary, &buffer) == kCVReturnSuccess,
                      let buffer = buffer else { throw PipelineError.failure("视频预览缓冲分配失败") }
                try processor.process(original, into: buffer, tuning: tuning)
                info.attachColors(to: buffer)
                processed = buffer
            }
            let pair = VideoFramePair(original: original, processed: processed, transform: info.transform,
                                      time: time, isHDR: info.isHDR, duration: CMSampleBufferGetDuration(sample),
                                      originalSample: sample, processedSample: processedSample)
            frames.removeAll {CMTimeCompare($0.time,time)==0}
            frames.append(pair)
            let bytesPerPair = max(1, CVPixelBufferGetDataSize(original) + CVPixelBufferGetDataSize(processed))
            while frames.count > max(2, frameBudget / bytesPerPair) {
                let distant=frames.indices.max {abs(frames[$0].time.seconds-requestedTime)<abs(frames[$1].time.seconds-requestedTime)}!
                frames.remove(at:distant)
            }
            lastTime = time.seconds
            return pair
        }
    }

    private func publish(_ error: Error) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.lock.lock(); let active = !self.stopped; self.lock.unlock()
            if active { self.onError?(error) }
        }
    }
}
