import Foundation
import MetalKit
import QuartzCore

final class DisplayRenderer: NSObject, MTKViewDelegate {
    private var pending: [DisplayFrame] = []
    private var activeID: UUID?
    private var activeFrame: DisplayFrame?
    private var retries: [UUID:Int] = [:]
    private let engine: PipelineEngine
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private weak var view: MTKView?
    private var primed=false,primeInFlight=false
    init(view: MTKView,engine: PipelineEngine,processor: MetalProcessor) throws {
        self.engine=engine
        guard let queue=processor.device.makeCommandQueue() else {throw PipelineError.failure("Metal 显示队列不可用")}
        self.queue=queue
        let descriptor=MTLRenderPipelineDescriptor()
        descriptor.vertexFunction=processor.library.makeFunction(name:"displayVertex")
        descriptor.fragmentFunction=processor.library.makeFunction(name:"displayFragment")
        descriptor.colorAttachments[0].pixelFormat=view.colorPixelFormat
        pipeline=try processor.device.makeRenderPipelineState(descriptor:descriptor)
        super.init()
        self.view=view
        view.delegate=self
    }
    func prime() {
        DispatchQueue.main.async { [weak self] in self?.view?.draw() }
    }
    private var lastFrame: DisplayFrame?
    private var displayParameters = DisplayParameters()

    func currentFrame() -> DisplayFrame? {lastFrame}

    func update(comparison: ComparisonMode, split: Float, viewport: ComparisonViewport) {
        let next = DisplayParameters(mode: UInt32(comparison.rawValue), split: split,
                                              zoom: viewport.zoom, centerX: viewport.centerX,
                                              centerY: viewport.centerY)
        guard next != displayParameters else{return}
        displayParameters=next
        if activeID == nil { view?.draw() }
    }

    func enqueue(_ frame: DisplayFrame) {
        dispatchPrecondition(condition: .onQueue(.main))
        if engine.isLive {
            for stale in pending {
                engine.displaySuperseded(stale, reason: "newer output ready")
            }
            pending.removeAll(keepingCapacity: true)
        }
        pending.append(frame)
        if activeID == nil { view?.draw() }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        DispatchQueue.main.async { [weak self] in self?.view?.draw() }
    }

    func draw(in view: MTKView) {
        if !primed { primeDisplay(view); return }
        guard activeID == nil else { return }
        let isNew = !pending.isEmpty
        guard let frame = pending.first ?? lastFrame else { return }
        guard let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
              let command = queue.makeCommandBuffer(),
              let encoder = command.makeRenderCommandEncoder(descriptor: pass) else {
            if isNew {
                pending.removeFirst()
                engine.displayFailed(frame, message: "显示 drawable 暂不可用")
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { [weak self] in self?.view?.draw() }
            return
        }
        if isNew {
            pending.removeFirst()
            activeID = frame.presentationID
            activeFrame = frame
        }
        frame.beginRead()
        encoder.setRenderPipelineState(pipeline)
        let w = Double(view.drawableSize.width), h = Double(view.drawableSize.height)
        let scale = min(w / 1920, h / 1080)
        encoder.setViewport(MTLViewport(originX: (w - 1920 * scale) / 2,
                                        originY: (h - 1080 * scale) / 2,
                                        width: 1920 * scale, height: 1080 * scale, znear: 0, zfar: 1))
        encoder.setFragmentTexture(frame.texture, index: 0)
        encoder.setFragmentTexture(frame.originalTexture, index: 1)
        var parameters = displayParameters
        encoder.setFragmentBytes(&parameters, length: MemoryLayout<DisplayParameters>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        if isNew {
            drawable.addPresentedHandler { [weak self] drawable in
                let time = drawable.presentedTime
                DispatchQueue.main.async {
                    guard let self = self, self.activeID == frame.presentationID else { return }
                    if time > 0 {
                        self.engine.presented(frame, time: time)
                        self.engine.metrics?.recovered("display",stableFor:1)
                        self.lastFrame?.recycle()
                        self.lastFrame = frame
                        self.retries[frame.presentationID] = nil
                        self.advance(frame.presentationID)
                    } else {
                        self.retry(frame)
                    }
                }
            }
        }
        command.addCompletedHandler { [weak self] command in
            self?.engine.metrics?.event("display_prime_completed",["status":command.status.rawValue])
            frame.endRead()
            guard isNew else { return }
            DispatchQueue.main.async {
                guard let self = self, self.activeID == frame.presentationID else { return }
                if let error = command.error {
                    self.engine.displayFailed(frame, message: error.localizedDescription)
                    self.advance(frame.presentationID)
                } else {
                    // Occluded windows may never issue a presented callback.
                    // Release the stalled output and let new live frames retry.
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { [weak self] in
                        guard let self = self, self.activeID == frame.presentationID else { return }
                        self.engine.displayFailed(frame, message: "等待 WindowServer 呈现超时，正在重试")
                        self.advance(frame.presentationID)
                    }
                }
            }
        }
        command.present(drawable)
        command.commit()
    }

    private func primeDisplay(_ view: MTKView) {
        guard !primeInFlight else{return}
        guard view.window != nil,view.drawableSize.width>0,view.drawableSize.height>0,
              let pass=view.currentRenderPassDescriptor,let drawable=view.currentDrawable,
              let command=queue.makeCommandBuffer(),let encoder=command.makeRenderCommandEncoder(descriptor:pass) else {retryPrime();return}
        primeInFlight=true;encoder.endEncoding()
        engine.metrics?.event("display_prime_submitted",["width":view.drawableSize.width,"height":view.drawableSize.height,
            "window_visible":view.window?.isVisible ?? false,"hidden":view.isHiddenOrHasHiddenAncestor])
        drawable.addPresentedHandler { [weak self] drawable in
            self?.engine.metrics?.event("display_prime_presented", ["time": drawable.presentedTime])
        }
        command.addCompletedHandler { [weak self] command in
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.primeInFlight = false
                if let error = command.error {
                    self.engine.metrics?.recoverableIssue("display", message: error.localizedDescription)
                    self.retryPrime()
                } else {
                    self.primed = true
                    self.view?.draw()
                }
            }
        }
        command.present(drawable);command.commit()
    }
    private func retryPrime() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { [weak self] in self?.view?.draw() }
    }
    deinit {
        for frame in pending { engine.displaySuperseded(frame, reason: "preview detached") }
        if let frame=activeFrame {engine.displaySuperseded(frame,reason:"preview detached")}
        lastFrame?.recycle()
    }

    private func advance(_ id: UUID) {
        guard activeID==id else{return}
        activeID=nil
        activeFrame=nil
        if !pending.isEmpty {view?.draw()}
    }
    private func retry(_ frame: DisplayFrame) {
        guard activeID==frame.presentationID else{return}
        let attempt=(retries[frame.presentationID] ?? 0)+1;retries[frame.presentationID]=attempt
        engine.metrics?.event("presentation_retry",["id":frame.source.id,"attempt":attempt,"reason":"drawable timestamp zero; not counted as displayed"])
        guard attempt<=10 else {
            retries[frame.presentationID]=nil;engine.displayFailed(frame,message:"画面未被 WindowServer 呈现，已记录显示缺帧")
            advance(frame.presentationID);return
        }
        activeID=nil;activeFrame=nil;pending.insert(frame,at:0)
        DispatchQueue.main.asyncAfter(deadline:.now()+1.0/60){[weak self] in self?.view?.draw()}
    }
}
