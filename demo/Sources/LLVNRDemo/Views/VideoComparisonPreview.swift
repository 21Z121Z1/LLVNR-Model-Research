import SwiftUI
import AVFoundation
import AppKit

struct VideoComparisonPreview: NSViewRepresentable {
    let pair: VideoFramePair
    let mode: ComparisonMode
    let split: Float
    @Binding var viewport: ComparisonViewport

    func makeNSView(context: Context) -> VideoComparisonView { VideoComparisonView() }

    func updateNSView(_ view: VideoComparisonView, context: Context) {
        view.comparisonInteraction = mode == .sideBySide
        view.onPan = { x, y in
            var next = viewport
            next.centerX -= Float(x) / next.zoom
            next.centerY += Float(y) / next.zoom
            next.clamp()
            viewport = next
        }
        view.onZoom = { factor in
            var next = viewport
            next.zoom *= factor
            next.clamp()
            viewport = next
        }
        view.onReset = { viewport = ComparisonViewport() }
        view.update(pair: pair, mode: mode, split: split, viewport: viewport)
    }
}

/// System video layers retain YUV transfer functions and perform the display's
/// HDR tone mapping directly, including on a paused video frame.
final class VideoComparisonView: NSView {
    var comparisonInteraction = false
    var onPan: ((Double, Double) -> Void)?
    var onZoom: ((Float) -> Void)?
    var onReset: (() -> Void)?
    private let originalClip = CALayer()
    private let processedClip = CALayer()
    private let originalLayer = AVSampleBufferDisplayLayer()
    private let processedLayer = AVSampleBufferDisplayLayer()
    private let divider = CALayer()
    private let displaySamples = VideoDisplaySamples()
    private var pair: VideoFramePair?
    private var lastTime: CMTime?
    private var mode = ComparisonMode.processed
    private var split: Float = 0.5
    private var viewport = ComparisonViewport()
    private var lastPoint = NSPoint.zero

    init() {
        super.init(frame: .zero)
        wantsLayer = true
        layer?.backgroundColor = NSColor.black.cgColor
        layer?.masksToBounds = true
        for clip in [processedClip, originalClip] {
            clip.masksToBounds = true
            layer?.addSublayer(clip)
        }
        originalClip.addSublayer(originalLayer)
        processedClip.addSublayer(processedLayer)
        divider.backgroundColor = NSColor.white.cgColor
        layer?.addSublayer(divider)
        for video in [originalLayer, processedLayer] { video.videoGravity = .resizeAspect }
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    func update(pair: VideoFramePair, mode: ComparisonMode, split: Float, viewport: ComparisonViewport) {
        let changed = self.pair?.id != pair.id
        let modeChanged = self.mode != mode
        self.pair = pair
        self.mode = mode
        self.split = split
        self.viewport = viewport
        originalClip.isHidden = mode == .processed
        processedClip.isHidden = mode == .original
        if changed || modeChanged {
            for video in [originalLayer, processedLayer] {
                if #available(macOS 26.0, *) { video.preferredDynamicRange = pair.isHDR ? .high : .standard }
                else { video.wantsExtendedDynamicRangeContent = pair.isHDR }
            }
            let flush=modeChanged || (lastTime.map {CMTimeCompare(pair.time,$0)<=0} ?? false)
            display(pair:pair,flush:flush)
            lastTime=pair.time
        }
        needsLayout = true
    }

    private func display(pair: VideoFramePair, flush: Bool) {
        do {
            if mode != .processed {
                let sample=try displaySamples.make(image:pair.original,colorsFrom:pair.original,time:pair.time)
                enqueue(sample,in:originalLayer,flush:flush)
            } else {flushLayer(originalLayer)}
            if mode != .original {
                let sample=try displaySamples.make(image:pair.processed,colorsFrom:pair.original,time:pair.time)
                enqueue(sample,in:processedLayer,flush:flush)
            } else {flushLayer(processedLayer)}
        } catch {
            flushLayer(originalLayer)
            flushLayer(processedLayer)
            DispatchQueue.main.asyncAfter(deadline:.now()+0.02) { [weak self] in
                guard let self=self,self.pair?.id==pair.id else{return}
                self.display(pair:pair,flush:true)
            }
        }
    }

    private func flushLayer(_ layer: AVSampleBufferDisplayLayer) {
        if #available(macOS 15.0, *) {layer.sampleBufferRenderer.flush()}
        else {layer.flush()}
    }

    private func enqueue(_ sample: CMSampleBuffer,in layer: AVSampleBufferDisplayLayer,flush: Bool) {
        if #available(macOS 15.0, *) {
            if flush || layer.sampleBufferRenderer.requiresFlushToResumeDecoding || !layer.sampleBufferRenderer.isReadyForMoreMediaData {
                layer.sampleBufferRenderer.flush()
            }
            layer.sampleBufferRenderer.enqueue(sample)
        } else {
            if flush || layer.requiresFlushToResumeDecoding || !layer.isReadyForMoreMediaData {layer.flush()}
            layer.enqueue(sample)
        }
    }

    override func layout() {
        super.layout()
        guard let pair = pair else { return }
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        originalClip.isHidden = mode == .processed
        processedClip.isHidden = mode == .original
        divider.isHidden = mode != .wipe && mode != .sideBySide
        originalClip.frame = bounds
        processedClip.frame = bounds
        if mode == .wipe {
            originalClip.frame.size.width = bounds.width * CGFloat(split)
            divider.frame = CGRect(x: bounds.width * CGFloat(split), y: 0, width: 1, height: bounds.height)
        }
        if mode == .sideBySide {
            originalClip.frame.size.width = bounds.width / 2
            processedClip.frame = CGRect(x: bounds.width / 2, y: 0, width: bounds.width / 2, height: bounds.height)
            divider.frame = CGRect(x: bounds.width / 2, y: 0, width: 1, height: bounds.height)
        }
        let width = CGFloat(CVPixelBufferGetWidth(pair.original))
        let height = CGFloat(CVPixelBufferGetHeight(pair.original))
        let transform = pair.transform
        let displayHeight = abs(transform.b * width) + abs(transform.d * height)
        let zoom = mode == .sideBySide ? CGFloat(viewport.zoom) : 1
        let scale = bounds.height / max(1, displayHeight) * zoom
        let centerX = mode == .sideBySide ? CGFloat(viewport.centerX) : 0.5
        let centerY = mode == .sideBySide ? CGFloat(viewport.centerY) : 0.5
        let paneWidth = mode == .sideBySide ? bounds.width / 2 : bounds.width
        for video in [originalLayer, processedLayer] {
            video.bounds = CGRect(x: 0, y: 0, width: width, height: height)
            video.position = CGPoint(x: paneWidth / 2 + (0.5 - centerX) * bounds.width * zoom,
                                     y: bounds.height / 2 + (centerY - 0.5) * bounds.height * zoom)
            // Convert the track's image-coordinate transform to AppKit's
            // bottom-left coordinate system, rotating around the layer center.
            video.setAffineTransform(CGAffineTransform(a: transform.a * scale, b: -transform.b * scale,
                                                       c: -transform.c * scale, d: transform.d * scale, tx: 0, ty: 0))
        }
        CATransaction.commit()
    }

    override func mouseDown(with event: NSEvent) {
        guard comparisonInteraction else { super.mouseDown(with: event); return }
        lastPoint = convert(event.locationInWindow, from: nil)
        if event.clickCount == 2 { onReset?() }
    }
    override func mouseDragged(with event: NSEvent) {
        guard comparisonInteraction else { super.mouseDragged(with: event); return }
        let point = convert(event.locationInWindow, from: nil)
        onPan?(Double((point.x - lastPoint.x) / max(1, bounds.width)),
               Double((point.y - lastPoint.y) / max(1, bounds.height)))
        lastPoint = point
    }
    override func scrollWheel(with event: NSEvent) {
        guard comparisonInteraction else { super.scrollWheel(with: event); return }
        onZoom?(exp(Float(event.scrollingDeltaY) * 0.015))
    }
    override func magnify(with event: NSEvent) {
        guard comparisonInteraction else { super.magnify(with: event); return }
        onZoom?(max(0.1, 1 + Float(event.magnification)))
    }
}
