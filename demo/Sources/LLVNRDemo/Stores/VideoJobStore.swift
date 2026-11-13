import AVFoundation
import Observation

@Observable final class VideoJobStore {
    var processing = false
    var isPresented = false
    var multiPass = false
    var progress = VideoProgress()
    var result: VideoProcessingResult?
    var error: String?
    var framePair: VideoFramePair?
    var loadingFrame = false
    var isPlaying = false
    var previewProgress = ""
    var previewDiffersFromExport = false
    var frameTime = 0.0 { didSet { requestFrame() } }
    @ObservationIgnored private var processor: VideoProcessor?
    @ObservationIgnored private var preview: VideoPreviewSession?
    @ObservationIgnored private var playbackTick: DispatchWorkItem?
    @ObservationIgnored private var audio: AVPlayer?
    @ObservationIgnored private var requestedTuning = Tuning()
    @ObservationIgnored private var exportedTuning = Tuning()
    @ObservationIgnored private var exportedModelID = ""
    @ObservationIgnored private var previewRevision = UUID()
    @ObservationIgnored private var scrubbing = false

    func start(input: URL, output: URL, modelID: String, tuning: Tuning,
               completion: ((Result<VideoProcessingResult, Error>) -> Void)? = nil) {
        processing = true
        isPresented = true
        progress = VideoProgress()
        error = nil
        closePreview { [weak self] in
            guard let self = self else { return }
            self.exportedTuning = tuning
            self.requestedTuning = tuning
            self.exportedModelID = modelID
            let processor = VideoProcessor()
            self.processor = processor
            processor.start(input: input, output: output, modelID: modelID, tuning: tuning, multiPass: self.multiPass,
                progress: { [weak self] in self?.progress = $0 },
                completion: { [weak self] outcome in
                    guard let self = self else { return }
                    self.processing = false
                    self.processor = nil
                    switch outcome {
                    case .success(let result):
                        self.result = result
                        self.audio = AVPlayer(url: result.input)
                        self.updatePreview(modelID: modelID, tuning: tuning)
                    case .failure(let error):
                        self.error = error is CancellationError ? "已取消处理" : error.localizedDescription
                    }
                    completion?(outcome)
                })
        }
    }

    func cancel() { processor?.cancel() }

    func closePreview(completion: @escaping () -> Void = {}) {
        playbackTick?.cancel()
        isPlaying = false
        audio?.pause()
        audio = nil
        framePair = nil
        loadingFrame = false
        result = nil
        frameTime = 0
        previewRevision = UUID()
        let old = preview
        preview = nil
        if let old = old { old.stop(completion: completion) }
        else { completion() }
    }

    func updatePreview(modelID: String, tuning: Tuning) {
        guard let result = result, !processing else { return }
        previewDiffersFromExport = modelID != exportedModelID || !tuning.hasSameProcessing(as: exportedTuning)
        let changed = !tuning.hasSameProcessing(as: requestedTuning)
        requestedTuning = tuning
        if preview?.modelID != modelID {
            let revision = UUID()
            previewRevision = revision
            let old = preview
            preview = nil
            let launch = { [weak self] in
                guard let self = self, self.previewRevision == revision else { return }
                let session = VideoPreviewSession(result: result, modelID: modelID,
                    exportedModelID: self.exportedModelID, exportedTuning: self.exportedTuning)
                self.preview = session
                session.onFrame = { [weak self, weak session] pair in
                    guard let self = self, self.preview === session else { return }
                    self.framePair = pair
                    self.loadingFrame = false
                    self.previewProgress = ""
                    self.schedulePlayback()
                }
                session.onProgress = { [weak self, weak session] current, target in
                    guard let self = self, self.preview === session else { return }
                    self.previewProgress = String(format: "重算历史 %.2f / %.2f s", current, target)
                }
                session.onError = { [weak self, weak session] error in
                    guard let self = self, self.preview === session else { return }
                    self.error = error.localizedDescription
                    self.loadingFrame = false
                    self.isPlaying = false
                    self.audio?.pause()
                }
                self.requestFrame()
            }
            if let old = old { old.stop(completion: launch) }
            else { launch() }
        } else if changed { requestFrame() }
    }

    func togglePlayback() {
        guard result != nil, !processing else { return }
        isPlaying.toggle()
        if isPlaying {
            if let result = result, frameTime >= result.duration - 0.05 { frameTime = 0 }
            else if loadingFrame { requestFrame() }
            else { schedulePlayback() }
        } else {
            playbackTick?.cancel()
            audio?.pause()
        }
    }

    func setScrubbing(_ active: Bool) {
        scrubbing = active
        if active {
            playbackTick?.cancel()
            audio?.pause()
        } else if !loadingFrame { schedulePlayback() }
    }

    private func requestFrame() {
        guard result != nil, let preview = preview else { return }
        playbackTick?.cancel()
        loadingFrame = true
        error = nil
        preview.request(seconds: frameTime, tuning: requestedTuning)
        // Hold audio when a seek needs more than one frame of processing.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            if self?.loadingFrame == true { self?.audio?.pause() }
        }
    }

    private func schedulePlayback() {
        playbackTick?.cancel()
        guard isPlaying, !scrubbing, let pair = framePair, let result = result else { return }
        let duration = pair.duration.isNumeric && pair.duration.seconds > 0
            ? pair.duration.seconds : 1 / Double(max(1, result.frameRate))
        let next = pair.time.seconds + duration
        guard next < result.duration - 0.00001 else {
            isPlaying = false
            audio?.pause()
            return
        }
        if let audio = audio {
            if abs(audio.currentTime().seconds - pair.time.seconds) > 0.1 {
                audio.seek(to: pair.time, toleranceBefore: .zero, toleranceAfter: .zero)
            }
            audio.play()
        }
        let tick = DispatchWorkItem { [weak self] in
            guard let self = self, self.isPlaying else { return }
            self.frameTime = next
        }
        playbackTick = tick
        DispatchQueue.main.asyncAfter(deadline: .now() + duration, execute: tick)
    }
}
