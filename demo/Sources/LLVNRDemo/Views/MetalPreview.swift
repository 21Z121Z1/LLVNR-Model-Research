import SwiftUI
import MetalKit

struct MetalPreview: NSViewRepresentable {
    let store: DemoStore
    func makeNSView(context: Context) -> MTKView {
        let view=ComparisonMetalView(frame:.zero,device:store.engine?.metal?.device)
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor=MTLClearColor(red:0,green:0,blue:0,alpha:1)
        view.framebufferOnly=true;view.isPaused=true;view.enableSetNeedsDisplay=false
        view.autoResizeDrawable=true
        if let layer=view.layer as? CAMetalLayer {
            layer.colorspace=CGColorSpace(name:CGColorSpace.sRGB)
            layer.maximumDrawableCount=3
            layer.presentsWithTransaction=false
            layer.isOpaque=true
        }
        store.attach(view:view)
        view.onPan = { [weak store] x, y in
            guard let store=store else {return}
            var viewport=store.comparisonViewport
            viewport.centerX-=Float(x)/viewport.zoom
            viewport.centerY+=Float(y)/viewport.zoom
            viewport.clamp()
            store.comparisonViewport=viewport
        }
        view.onZoom = { [weak store] factor in
            guard let store=store else {return}
            var viewport=store.comparisonViewport
            viewport.zoom*=factor;viewport.clamp()
            store.comparisonViewport=viewport
        }
        view.onReset = { [weak store] in store?.comparisonViewport=ComparisonViewport() }
        return view
    }
    func updateNSView(_ view: MTKView,context: Context) {
        (view as? ComparisonMetalView)?.comparisonInteraction=store.tuning.comparison == .sideBySide
        store.updateDisplay()
    }
}

final class ComparisonMetalView: MTKView {
    var comparisonInteraction=false
    var onPan: ((Double,Double)->Void)?
    var onZoom: ((Float)->Void)?
    var onReset: (()->Void)?
    private var lastPoint=NSPoint.zero

    override func mouseDown(with event: NSEvent) {
        guard comparisonInteraction else {super.mouseDown(with:event);return}
        lastPoint=convert(event.locationInWindow,from:nil)
        if event.clickCount==2 {onReset?()}
    }
    override func mouseDragged(with event: NSEvent) {
        guard comparisonInteraction else {super.mouseDragged(with:event);return}
        let point=convert(event.locationInWindow,from:nil)
        onPan?(Double((point.x-lastPoint.x)/max(1,bounds.width)),
               Double((point.y-lastPoint.y)/max(1,bounds.height)))
        lastPoint=point
    }
    override func scrollWheel(with event: NSEvent) {
        guard comparisonInteraction else {super.scrollWheel(with:event);return}
        onZoom?(exp(Float(event.scrollingDeltaY)*0.015))
    }
    override func magnify(with event: NSEvent) {
        guard comparisonInteraction else {super.magnify(with:event);return}
        onZoom?(max(0.1,1+Float(event.magnification)))
    }
}
