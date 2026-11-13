import Foundation
import Observation
import AppKit
import UniformTypeIdentifiers
import AVFoundation
import MetalKit

@Observable final class DemoStore {
    var catalog: ModelCatalog?
    var selectedModelID="" {didSet{
        if oldValue != selectedModelID {
            selectedPresetID=availablePresets.first?.id ?? ""
            prepared=false
            UserDefaults.standard.set(selectedModelID,forKey:"selectedFirmwareModel")
            videoJob.updatePreview(modelID:selectedModelID,tuning:tuning)
        }
    }}
    var selectedPresetID=""
    var cameras: [CameraInput] = []
    var selectedCameraID="" {didSet{UserDefaults.standard.set(selectedCameraID,forKey:"selectedCameraID")}}
    var showModelInspector=false
    var videoJob=VideoJobStore()
    var presetGain: Double=100
    var presetZoom: Double=1
    var presetTripod=false
    var selectedModel: ModelCatalog.Entry? {try? catalog?.entry(selectedModelID)}
    var availablePresets: [ModelCatalog.Preset] {
        guard let model=selectedModel else {return []}
        return catalog?.presets.filter{model.preset_ids.contains($0.id)} ?? []
    }
    var selectedPreset: ModelCatalog.Preset? {availablePresets.first{$0.id==selectedPresetID}}
    var selectedPresetText: String {
        guard let preset=selectedPreset,let catalog=catalog else {return "没有绑定的 tuning"}
        do {return try catalog.presetText(preset)} catch {return error.localizedDescription}
    }
    var canStart: Bool {selectedModel?.canRun == true}
    var status="就绪 · 1080p 输入 / 输出"
    var busy=false
    var isPaused=false
    var pauseTransition=false
    var hasFrame=false
    var savingFrames=false
    var savedFramesPath=""
    var showSavedFrames=false
    var frameImageFormat=FrameImageFormat.heif
    var frameImageQuality=0.95
    var prepared=false
    var snapshot=MetricsSnapshot()
    var errorMessage: String?
    var comparisonViewport=ComparisonViewport() {didSet{updateDisplay()}}
    var tuning=Tuning() {didSet{
        engine?.update(tuning)
        updateDisplay()
        videoJob.updatePreview(modelID:selectedModelID,tuning:tuning)
        if let data=try? JSONEncoder().encode(tuning) {UserDefaults.standard.set(data,forKey:"modelTuning")}
    }}
    var reportPath=""
    var validationPassed: Bool?
    @ObservationIgnored var engine: PipelineEngine?
    @ObservationIgnored private var source: FrameSource?
    @ObservationIgnored private var renderer: DisplayRenderer?
    @ObservationIgnored private var timer: Timer?
    @ObservationIgnored private var sourceStarted=false
    @ObservationIgnored private var options=RunOptions()
    @ObservationIgnored private var earlyFrames: [DisplayFrame] = []
    @ObservationIgnored private var activity: NSObjectProtocol?
    @ObservationIgnored private var cameraObservers: [NSObjectProtocol] = []
    @ObservationIgnored private var stopping=false
    @ObservationIgnored private var afterStop: (()->Void)?

    init() {
        do {
            let loaded=try ModelCatalog.load()
            catalog=loaded
            let saved=UserDefaults.standard.string(forKey:"selectedFirmwareModel")
            selectedModelID=(try? loaded.entry(saved))?.id ?? loaded.default_model_id
            selectedPresetID=availablePresets.first?.id ?? ""
        } catch {
            catalog=nil
            status="未安装模型资产包"
        }
        if let data=UserDefaults.standard.data(forKey:"modelTuning"),let saved=try? JSONDecoder().decode(Tuning.self,from:data) {tuning=saved}
        selectedCameraID=UserDefaults.standard.string(forKey:"selectedCameraID") ?? ""
        refreshCameras()
        for notification in [AVCaptureDevice.wasConnectedNotification,AVCaptureDevice.wasDisconnectedNotification] {
            cameraObservers.append(NotificationCenter.default.addObserver(forName:notification,object:nil,queue:.main){[weak self] _ in self?.refreshCameras()})
        }
    }

    func importModelKit() {
        guard !busy, !videoJob.processing else { return }
        let panel=NSOpenPanel()
        panel.title="导入 LLVNR 模型资产包"
        panel.prompt="导入"
        panel.message="选择 build_kit.py 生成的 .llvnrkit 文件夹或 .zip"
        panel.canChooseFiles=true
        panel.canChooseDirectories=true
        panel.canCreateDirectories=false
        panel.allowsMultipleSelection=false
        var types:[UTType]=[.zip]
        if let kitType=UTType(filenameExtension:"llvnrkit") {types.append(kitType)}
        panel.allowedContentTypes=types
        panel.begin { [weak self] response in
            guard response == .OK, let source=panel.url, let self=self else { return }
            do {
                let loaded=try ModelStore.install(from:source)
                self.catalog=loaded
                let saved=UserDefaults.standard.string(forKey:"selectedFirmwareModel")
                self.selectedModelID=(try? loaded.entry(saved))?.id ?? loaded.default_model_id
                self.selectedPresetID=self.availablePresets.first?.id ?? ""
                self.prepared=false
                self.errorMessage=nil
                self.status="模型资产包已安装"
            } catch {
                self.errorMessage="导入模型资产包失败：\(error.localizedDescription)"
            }
        }
    }

    func refreshCameras() {
        cameras=CameraSource.inputs()
        if !busy && !cameras.contains(where:{$0.id==selectedCameraID}) {selectedCameraID=cameras.first?.id ?? ""}
    }

    func start(_ options: RunOptions=RunOptions()) {
        guard !busy,!videoJob.processing else {return}
        var options=options
        if options.useDefaultTuning {tuning=Tuning()}
        if let id=options.modelID {selectedModelID=id}
        if let input=options.videoInput,let output=options.videoOutput {
            videoJob.multiPass=options.multiPass
            beginVideo(input:input,output:output,options:options)
            return
        }
        if videoJob.result != nil {
            videoJob.closePreview { [weak self] in self?.start(options) }
            return
        }
        guard canStart else {errorMessage=selectedModel?.status_detail ?? "所选模型不可用";return}
        options.modelID=selectedModelID
        options.cameraID=options.cameraID ?? (selectedCameraID.isEmpty ? nil:selectedCameraID)
        self.options=options;busy=true;prepared=false;sourceStarted=false;errorMessage=nil
        isPaused=false;pauseTransition=false;stopping=false;hasFrame=false
        status="正在准备所选模型，首次编译需要稍候…"
        activity=ProcessInfo.processInfo.beginActivity(options:[.userInitiated,.latencyCritical],reason:"Continuous local camera LLVNR processing")
        renderer=nil;engine=nil;source=nil;snapshot=MetricsSnapshot()
        let e=PipelineEngine();engine=e;e.update(tuning)
        e.onReady={ [weak self,weak e] in
            guard let self=self else{return}
            guard !self.stopping else {e?.sourceEnded();return}
            self.prepared=true;self.status="ANE 就绪 · 正在连接显示"
        }
        e.onOutput={ [weak self] frame in
            guard let self=self else {frame.recycle();return}
            if frame.isPreview && !self.isPaused {frame.recycle();return}
            self.hasFrame=true
            if let renderer=self.renderer {renderer.enqueue(frame)}
            else {self.earlyFrames.append(frame)}
        }
        e.onError={ [weak self] message in
            guard let self=self else{return}
            self.errorMessage=message;self.status="运行失败：\(message)"
            if self.sourceStarted {self.stop()}
            else {self.busy=false}
        }
        e.onFinished={ [weak self] result in
            guard let self=self else{return}
            self.timer?.invalidate();self.timer=nil
            if let metrics=self.engine?.metrics {self.snapshot=metrics.snapshot();self.reportPath=metrics.directory.path}
            self.busy=false;self.source=nil
            self.isPaused=false;self.pauseTransition=false;self.stopping=false
            if let activity=self.activity {ProcessInfo.processInfo.endActivity(activity);self.activity=nil}
            if let pass=result["validation_pass"] as? Bool {
                self.validationPassed=pass;self.status=pass ? "原生 Metal / ANE 数值验证通过":"原生数值验证失败"
            } else {
                let pass=result["acceptance_pass"] as? Bool ?? false
                self.status=options.replayDirectory==nil ? "相机已停止" : (pass ? "1080p30 验收通过":"测量结束 · 详情已记录")
                self.snapshot.verdict=self.status
            }
            if options.quitAfterRun {DispatchQueue.main.asyncAfter(deadline:.now()+0.3){NSApp.terminate(nil)}}
            let next=self.afterStop;self.afterStop=nil;next?()
        }
        e.prepare(options:options)
        timer=Timer.scheduledTimer(withTimeInterval:0.25,repeats:true){[weak self] _ in
            guard let self=self else{return}
            if let metrics=self.engine?.metrics {self.snapshot=metrics.snapshot();self.reportPath=metrics.directory.path}
        }
    }
    func attach(view: MTKView) {
        guard let engine=engine,let processor=engine.metal else{return}
        do {
            renderer=try DisplayRenderer(view:view,engine:engine,processor:processor)
            updateDisplay()
            for frame in earlyFrames {renderer?.enqueue(frame)}
            earlyFrames.removeAll()
            renderer?.prime()
            // Capture and inference recover independently of WindowServer.
            startSource()
        } catch {errorMessage=error.localizedDescription;busy=false}
    }
    private func startSource() {
        guard !sourceStarted,!stopping,let e=engine else{return}
        sourceStarted=true
        e.metrics?.event("source_start_requested",["authorization":AVCaptureDevice.authorizationStatus(for:.video).rawValue])
        let launch={ [weak self] in
            guard let self=self else{return}
            guard self.busy,!self.stopping,self.engine === e else{return}
            e.work.async {
                do {
                    let s: FrameSource
                    if let directory=self.options.replayDirectory {
                        s=try ReplaySource(directory:directory,count:self.options.replayFrames,receive:{[weak e] in e?.enqueue($0)},ended:{[weak e] in e?.sourceEnded()})
                    } else {
                        s=CameraSource(deviceID:self.options.cameraID,receive:{[weak e] in e?.enqueue($0)},dropped:{[weak e] in e?.metrics?.drop("capture_drop",values:$0)},ended:{[weak e] in e?.sourceEnded()})
                    }
                    DispatchQueue.main.sync{self.source=s}
                    try s.start()
                    e.metrics?.event("source_started",["camera_id":self.options.cameraID ?? "default"])
                    DispatchQueue.main.async {
                        self.status=self.options.replayDirectory==nil ? "正在处理 · 本地相机 1080p30":"正在处理 · 1080p30 真实 YUV 定速重放"
                        if self.options.cameraSeconds>0 && self.options.replayDirectory==nil {
                            DispatchQueue.main.asyncAfter(deadline:.now()+self.options.cameraSeconds){[weak self] in self?.stop()}
                        }
                    }
                } catch {DispatchQueue.main.async{self.errorMessage=error.localizedDescription;self.status=error.localizedDescription;self.busy=false;e.sourceEnded()}}
            }
        }
        if options.replayDirectory==nil && AVCaptureDevice.authorizationStatus(for:.video) == .notDetermined {
            AVCaptureDevice.requestAccess(for:.video){allowed in DispatchQueue.main.async {
                if allowed {launch()} else {self.errorMessage="系统未授权相机";self.busy=false;e.sourceEnded()}
            }}
        } else {launch()}
    }
    func stop() {
        guard busy else{return};stopping=true;status="正在停止相机…"
        if let source=source {source.stop()} else {engine?.sourceEnded()}
    }
    func togglePause() {
        guard busy,!pauseTransition,!stopping,let source=source,let engine=engine else{return}
        pauseTransition=true
        if isPaused {
            engine.setPaused(false){[weak self] in self?.resumeSource(source,engine:engine)}
        } else {
            source.setPaused(true){[weak self] error in
                guard let self=self,!self.stopping,self.busy else{return}
                if let error=error {engine.metrics?.recoverableIssue("camera",message:error.localizedDescription)}
                engine.setPaused(true){[weak self] in
                    guard let self=self,!self.stopping,self.busy else{return}
                    self.isPaused=true;self.pauseTransition=false;self.status="已暂停"
                }
            }
        }
    }
    private func resumeSource(_ source: FrameSource, engine: PipelineEngine) {
        guard busy,!stopping,self.engine === engine else{return}
        source.setPaused(false){[weak self] error in
            guard let self=self,self.busy,!self.stopping,self.engine === engine else{return}
            if let error=error {
                engine.metrics?.recoverableIssue("camera",message:error.localizedDescription)
                DispatchQueue.main.asyncAfter(deadline:.now()+0.5){[weak self] in self?.resumeSource(source,engine:engine)}
            } else {
                engine.metrics?.recovered("camera")
                self.isPaused=false;self.pauseTransition=false;self.status="正在处理"
            }
        }
    }
    func resetState() {engine?.resetState()}
    func updateDisplay() {
        renderer?.update(comparison:tuning.comparison,split:tuning.split,viewport:comparisonViewport)
    }
    func saveFrames() {
        let pair=videoJob.framePair
        let frame=pair==nil ? renderer?.currentFrame():nil
        guard !savingFrames,!videoJob.loadingFrame,frame != nil || pair != nil else{return}
        frame?.beginRead()
        let panel=NSOpenPanel()
        panel.title="保存处理前后帧"
        panel.prompt="保存到此文件夹"
        panel.canChooseFiles=false;panel.canChooseDirectories=true
        panel.canCreateDirectories=true;panel.allowsMultipleSelection=false
        panel.directoryURL=FileManager.default.urls(for:.picturesDirectory,in:.userDomainMask).first
        let selected: (NSApplication.ModalResponse)->Void = { [weak self] response in
            guard response == .OK,let folder=panel.url else {frame?.endRead();return}
            guard let self=self else {frame?.endRead();return}
            self.savingFrames=true
            let finished: (Result<URL,Error>)->Void = { [weak self] result in
                guard let self=self else{return}
                self.savingFrames=false
                switch result {
                case .success(let url):self.savedFramesPath=url.path;self.showSavedFrames=true
                case .failure(let error):self.errorMessage="帧保存失败：\(error.localizedDescription)"
                }
            }
            if let pair=pair {FrameExporter.save(pair,in:folder,format:self.frameImageFormat,quality:self.frameImageQuality,completion:finished)}
            else if let frame=frame {FrameExporter.save(frame,in:folder,format:self.frameImageFormat,quality:self.frameImageQuality,completion:finished)}
        }
        if let window=NSApp.keyWindow {panel.beginSheetModal(for:window,completionHandler:selected)}
        else {panel.begin(completionHandler:selected)}
    }
    func revealSavedFrames() {
        guard !savedFramesPath.isEmpty else{return}
        NSWorkspace.shared.open(URL(fileURLWithPath:savedFramesPath))
    }
    func openVideo() {
        guard !videoJob.processing,canStart else{return}
        let panel=NSOpenPanel()
        panel.title="选择要处理的视频";panel.allowedContentTypes=[.movie];panel.allowsMultipleSelection=false
        panel.begin { [weak self] response in
            guard response == .OK,let input=panel.url,let self=self else{return}
            self.chooseVideoOutput(input:input)
        }
    }

    func exportCurrentVideo() {
        guard let input=videoJob.result?.input,!videoJob.processing else{return}
        chooseVideoOutput(input:input)
    }

    private func chooseVideoOutput(input: URL) {
        let save=NSSavePanel()
        let mp4=input.pathExtension.lowercased()=="mp4"
        save.title="导出处理后的视频";save.allowedContentTypes=mp4 ? [.mpeg4Movie]:[.quickTimeMovie]
        save.nameFieldStringValue=input.deletingPathExtension().lastPathComponent+"-LLVNR."+(mp4 ? "mp4":"mov")
        save.directoryURL=input.deletingLastPathComponent()
        save.begin { [weak self] response in
            guard response == .OK,let output=save.url,let self=self else{return}
            let launch: ()->Void = { [weak self] in self?.beginVideo(input:input,output:output) }
            if self.busy {self.afterStop=launch;self.stop()} else {launch()}
        }
    }

    private func beginVideo(input: URL, output: URL, options: RunOptions=RunOptions()) {
        prepared=false;renderer=nil;engine=nil;source=nil;hasFrame=false
        let modelID=selectedModelID
        videoJob.start(input:input,output:output,modelID:modelID,tuning:tuning){[weak self] outcome in
            guard let self=self else{return}
            if let folder=options.videoFrameDirectory,case .success(let result)=outcome {
                Task {
                    do {
                        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
                        let pair=try await VideoFramePair.load(result:result,seconds:0)
                        FrameExporter.save(pair,in:folder,format:self.frameImageFormat,quality:self.frameImageQuality){ saved in
                            var report=result.report
                            switch saved {
                            case .success(let url):report["saved_frame_pair"]=url.path
                            case .failure(let error):report["passed"]=false;report["error"]=error.localizedDescription
                            }
                            self.finishVideoCommand(report,options:options)
                        }
                    } catch {DispatchQueue.main.async{self.finishVideoCommand(["passed":false,"error":error.localizedDescription],options:options)}}
                }
            } else {
                let report: [String:Any]
                switch outcome {
                case .success(let result):report=result.report
                case .failure(let error):report=["passed":false,"error":error.localizedDescription]
                }
                self.finishVideoCommand(report,options:options)
            }
        }
    }

    private func finishVideoCommand(_ report: [String:Any], options: RunOptions) {
        if let directory=options.reportDirectory {
            do {
                try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
                try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys])
                    .write(to:directory.appendingPathComponent("video_validation.json"),options:.atomic)
            } catch {errorMessage=error.localizedDescription}
        }
        if options.quitAfterRun {
            videoJob.isPresented=false
            DispatchQueue.main.asyncAfter(deadline:.now()+0.2){NSApp.terminate(nil)}
        }
    }
    func revealReport() {if !reportPath.isEmpty {NSWorkspace.shared.open(URL(fileURLWithPath:reportPath))}}
    func revealSelectedModel() {
        guard let model=selectedModel,let (directory,_)=try? catalog?.runtimeManifest(model) else {return}
        NSWorkspace.shared.activateFileViewerSelecting([directory.appendingPathComponent("manifest.json")])
    }
    func revealSelectedPreset() {
        guard let base=ModelCatalog.directory,let preset=selectedPreset else {return}
        NSWorkspace.shared.activateFileViewerSelecting([base.appendingPathComponent(preset.file)])
    }
    func applyInputPreset() {
        guard (!busy || isPaused),let preset=selectedPreset,let model=selectedModel,model.canRun else {return}
        do {
            let suffix=presetTripod ? "OnTripodLUT":"LUT"
            func mapped(_ prefix: String) throws -> Float {
                guard let scale=preset.input_modulation[prefix+"Scale"+suffix],
                      let offset=preset.input_modulation[prefix+"Offset"+suffix],presetZoom.isFinite else {
                    throw PipelineError.failure("缺少对应 tuning LUT")
                }
                let value=try presetZoom*ModelCatalog.interpolate(scale,at:presetGain)+ModelCatalog.interpolate(offset,at:presetGain)
                guard value.isFinite,(0...3).contains(value) else {throw PipelineError.failure("试算结果超出当前输入 map 的 0…3 范围")}
                return Float(value)
            }
            var next=tuning
            next.dnr=try mapped("DNRZoomFactor");next.fusion=try mapped("FusionZoomFactor")
            tuning=next;status="已应用 \(preset.name) 的降噪与融合输入"
        } catch {errorMessage=error.localizedDescription}
    }
    deinit {for observer in cameraObservers {NotificationCenter.default.removeObserver(observer)}}
}
