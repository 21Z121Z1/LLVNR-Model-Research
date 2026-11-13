import SwiftUI

struct ContentView: View {
    @Bindable var store: DemoStore
    @AppStorage("showParameterInspector") private var showInspector = true

    var body: some View {
        GeometryReader { geometry in
            let handleHeight: CGFloat = showsWipeHandle ? 18 : 0
            let videoHeight = min(geometry.size.width / previewAspect,
                                  max(1, geometry.size.height - footerMinimumHeight - handleHeight))
            let videoWidth = videoHeight * previewAspect
            VStack(spacing: 0) {
                preview.frame(width: videoWidth, height: videoHeight)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                if showsWipeHandle {
                    WipeHandle(value: $store.tuning.split)
                        .frame(width: videoWidth, height: handleHeight)
                }
                comparisonFooter
                    .frame(maxWidth: .infinity)
                    .frame(height: footerMinimumHeight, alignment: .center)
            }
            .frame(width: geometry.size.width, height: geometry.size.height, alignment: .top)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .frame(minWidth: 620, maxWidth: .infinity, maxHeight: .infinity)
        .inspector(isPresented: $showInspector) {
            ParameterInspector(store: store)
                .inspectorColumnWidth(min: 300, ideal: 330, max: 380)
        }
        .frame(minWidth: 960, minHeight: 640)
        .toolbar { previewToolbar }
        .sheet(isPresented:$store.videoJob.isPresented) {VideoJobView(job:store.videoJob)}
    }

    private var showsComparison: Bool {store.prepared || store.videoJob.framePair != nil}
    private var showsWipeHandle: Bool {showsComparison && store.tuning.comparison == .wipe}

    private var footerMinimumHeight: CGFloat {
        var height: CGFloat = 54
        if store.videoJob.framePair != nil {height += 24}
        if showsComparison && store.tuning.comparison == .sideBySide {height += 22}
        if store.errorMessage != nil {height += 32}
        if store.videoJob.error != nil {height += 24}
        return height
    }

    private var comparisonFooter: some View {
        VStack(spacing: 4) {
            if let result=store.videoJob.result,store.videoJob.framePair != nil {
                HStack(spacing:10) {
                    Button(store.videoJob.isPlaying ? "暂停视频":"播放视频",
                           systemImage:store.videoJob.isPlaying ? "pause.fill":"play.fill") {
                        store.videoJob.togglePlayback()
                    }.labelStyle(.iconOnly).buttonStyle(.plain)
                    Text(String(format:"%.2f s",store.videoJob.frameTime)).monospacedDigit()
                    Slider(value:$store.videoJob.frameTime,in:0...max(0.001,result.duration),
                           onEditingChanged:store.videoJob.setScrubbing)
                        .labelsHidden().accessibilityLabel("视频对比帧位置")
                    Text(String(format:"%.2f s",result.duration)).monospacedDigit().foregroundStyle(.secondary)
                    if store.videoJob.loadingFrame {
                        ProgressView().controlSize(.mini).help(store.videoJob.previewProgress)
                    }
                    if store.videoJob.previewDiffersFromExport {
                        Button("导出当前参数…") {store.exportCurrentVideo()}.controlSize(.small)
                    }
                }.font(.caption).frame(maxWidth:720)
            }
            if showsComparison {
                ComparisonControls(selection: $store.tuning.comparison)
                if store.tuning.comparison == .sideBySide {comparisonZoom}
            }
            metrics
            if let error = store.errorMessage {
                Label(error, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                    .font(.callout)
                    .textSelection(.enabled)
            }
            if let error=store.videoJob.error, store.videoJob.result != nil {
                Text(error).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var preview: some View {
        ZStack {
            Color.black
            if let pair=store.videoJob.framePair {
                VideoComparisonPreview(pair:pair,mode:store.tuning.comparison,split:store.tuning.split,
                    viewport:$store.comparisonViewport)
                    .allowedDynamicRange(.high)
            } else if store.prepared {
                MetalPreview(store: store)
            } else {
                ContentUnavailableView {
                    Label(store.videoJob.processing ? "正在处理视频":(store.busy ? "正在准备模型" : "相机预览"), systemImage: "camera.aperture")
                } description: {
                    Text(store.busy ? store.status : "选择模型后启动相机，比较降噪前后的画面。")
                } actions: {
                    if store.busy || store.videoJob.processing {
                        ProgressView().controlSize(.small)
                    } else {
                        if #available(macOS 26.0, *) {
                            cameraButton.buttonStyle(.glassProminent)
                        } else {
                            cameraButton.buttonStyle(.borderedProminent)
                        }
                    }
                }
                .environment(\.colorScheme, .dark)
            }
        }
        .aspectRatio(previewAspect, contentMode: .fit)
        .background(.black)
    }
    private var previewAspect: CGFloat {
        guard let pair=store.videoJob.framePair else{return 16/9}
        return pair.displaySize.width/max(1,pair.displaySize.height)
    }

    private var comparisonZoom: some View {
        HStack(spacing: 12) {
            Image(systemName: "minus.magnifyingglass").foregroundStyle(.secondary)
            Slider(value: $store.comparisonViewport.zoom, in: 1...8)
                .frame(width: 160)
                .accessibilityLabel("并排同步缩放")
                .onChange(of: store.comparisonViewport.zoom) { _, _ in store.comparisonViewport.clamp() }
            Text(String(format: "%.1f×", store.comparisonViewport.zoom)).monospacedDigit()
            Button("复位") { store.comparisonViewport = ComparisonViewport() }
        }
        .controlSize(.small)
        .font(.caption)
        .frame(maxWidth: .infinity)
    }

    private var metrics: some View {
        HStack(spacing: 24) {
            if let result=store.videoJob.result,store.videoJob.framePair != nil {
                metric("视频帧率",String(format:"%.2f fps",result.frameRate))
                metric("源码率",String(format:"%.2f Mbps",result.sourceBitRate/1e6))
                metric("导出码率",String(format:"%.2f Mbps",result.outputBitRate/1e6))
            } else {
                metric("实际处理", String(format: "%.2f fps", store.snapshot.processedFPS))
                metric("显示帧率", String(format: "%.2f fps", store.snapshot.presentedFPS))
                metric("显示延迟 P95", String(format: "%.1f ms", store.snapshot.sourceLatencyP95))
            }
        }
        .font(.callout)
        .lineLimit(1)
        .minimumScaleFactor(0.8)
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.vertical, 2)
    }

    private func metric(_ name: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(name).foregroundStyle(.secondary)
            Text(value).monospacedDigit()
        }
    }

    private var cameraButton: some View {
        Button(store.busy ? "停止相机" : "启动相机", systemImage: store.busy ? "stop.fill" : "play.fill") {
            if store.busy { store.stop() } else { store.start() }
        }
        .disabled(store.videoJob.processing || (!store.busy && !store.canStart))
    }

    @ToolbarContentBuilder private var previewToolbar: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            cameraButton
                .labelStyle(.titleAndIcon)
                .keyboardShortcut(.space, modifiers: [])
        }
        ToolbarItemGroup(placement: .primaryAction) {
            Menu {
                Toggle("多次编码",isOn:$store.videoJob.multiPass)
                Divider()
                Button("选择视频并导出…") {store.openVideo()}
            } label: {
                Label("处理视频",systemImage:"film")
            } primaryAction: {store.openVideo()}
                .disabled(store.videoJob.processing)
                .help("导入并处理视频")
            Button(store.isPaused ? "继续处理":"暂停处理", systemImage: store.isPaused ? "play.fill":"pause.fill") {
                store.togglePause()
            }
            .disabled(!store.busy || !store.hasFrame || store.pauseTransition)
            .help(store.isPaused ? "继续处理":"暂停处理")
            .keyboardShortcut("p",modifiers:[.command,.shift])
            Menu {
                Picker("格式",selection:$store.frameImageFormat) {
                    ForEach(FrameImageFormat.allCases) {format in
                        Text(format.title+(format == .jpeg && store.videoJob.framePair?.isHDR==true ? "（SDR）":"")).tag(format)
                    }
                }
                Picker("质量",selection:$store.frameImageQuality) {
                    ForEach([0.5,0.75,0.85,0.9,0.95,1.0],id:\.self) {quality in Text("\(Int((quality*100).rounded()))%").tag(quality)}
                }.disabled(store.frameImageFormat == .png)
                Divider()
                Button("保存处理前后帧…") {store.saveFrames()}
            } label: {
                Label("保存处理前后帧",systemImage:"square.and.arrow.down")
            } primaryAction: {
                store.saveFrames()
            }
                .disabled((!store.hasFrame && store.videoJob.framePair==nil) || store.savingFrames || store.pauseTransition || store.videoJob.loadingFrame)
                .help("保存原始画面与处理后画面")
                .popover(isPresented:$store.showSavedFrames) {
                    VStack(alignment:.leading,spacing:12) {
                        Label("已保存处理前后两张帧",systemImage:"checkmark.circle")
                        Button("在 Finder 中打开") {store.revealSavedFrames();store.showSavedFrames=false}
                    }.padding(18)
                }
        }
        if #available(macOS 26.0, *) {
            ToolbarSpacer(.fixed, placement: .primaryAction)
        }
        ToolbarItemGroup(placement: .primaryAction) {
            Button("重置时间状态", systemImage: "arrow.counterclockwise") { store.resetState() }
                .disabled(!store.busy)
                .help("清空历史帧，从下一帧重新融合")
                .keyboardShortcut("r", modifiers: [.command, .shift])
            Button("参数检查器", systemImage: "sidebar.right") {
                withAnimation { showInspector.toggle() }
            }
            .help(showInspector ? "隐藏参数检查器" : "显示参数检查器")
            .keyboardShortcut("i", modifiers: [.command, .option])
        }
    }
}

private struct ComparisonControls: View {
    @Binding var selection: ComparisonMode

    var body: some View {
        if #available(macOS 26.0, *) {
            GlassEffectContainer(spacing: 8) {
                HStack(spacing: 8) {
                    ForEach(ComparisonMode.allCases) { mode in
                        if mode == selection {
                            button(mode).buttonStyle(.glassProminent)
                        } else {
                            button(mode).buttonStyle(.glass)
                        }
                    }
                }
                .buttonBorderShape(.roundedRectangle(radius: 8))
                .controlSize(.regular)
            }
        } else {
            Picker("画面对比", selection: $selection) {
                ForEach(ComparisonMode.allCases) { mode in Text(mode.title).tag(mode) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(8)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private func button(_ mode: ComparisonMode) -> some View {
        Button { selection = mode } label: {
            Label(mode.title, systemImage: icon(mode))
                .font(.callout)
                .padding(.horizontal, 4)
        }
        .help(mode.title)
        .accessibilityAddTraits(mode == selection ? .isSelected : [])
    }

    private func icon(_ mode: ComparisonMode) -> String {
        switch mode {
        case .processed: return "sparkles"
        case .original: return "camera"
        case .wipe: return "rectangle.lefthalf.filled"
        case .sideBySide: return "rectangle.split.2x1"
        }
    }
}

private struct ParameterInspector: View {
    @Bindable var store: DemoStore

    var body: some View {
        Form {
            Section {
                Picker("输入设备",selection:$store.selectedCameraID) {
                    if store.cameras.isEmpty {Text("未检测到摄像头").tag("")}
                    ForEach(store.cameras) {camera in Text(camera.name).tag(camera.id)}
                }
                .disabled(store.busy || store.videoJob.processing)
            } header: {
                Label("摄像头输入",systemImage:"video")
            }

            Section {
                ModelSelectionView(store: store)
            } header: {
                Label("模型", systemImage: "square.stack.3d.up")
            }

            Section {
                TuningView(store: store)
                    .disabled(store.videoJob.processing)
            } header: {
                Label("处理参数", systemImage: "slider.horizontal.3")
            }

            Section {
                FirmwareInputsView(store: store)
            } header: {
                Label("固件输入试算", systemImage: "function")
            }

            Section {
                MetricsView(snapshot: store.snapshot).padding(.vertical, 6)
                if !store.reportPath.isEmpty {
                    Button("查看逐帧记录", systemImage: "folder") { store.revealReport() }
                }
            } header: {
                Label("运行详情", systemImage: "waveform.path.ecg")
            }
        }
        .formStyle(.grouped)
        .controlSize(.small)
    }
}

struct TuningView: View {
    @Bindable var store: DemoStore

    var body: some View {
        knob("DNR zoom map · 空域降噪强度", value: $store.tuning.dnr, range: 0...3)
        knob("Fusion zoom map · 时域降噪强度", value: $store.tuning.fusion, range: 0...3)
        knob("History update weight · 时域历史更新强度", value: $store.tuning.historyWeight, range: 0...1)
        Button("恢复默认参数") { store.tuning = Tuning() }
    }

    private func knob(_ title: String, value: Binding<Float>, range: ClosedRange<Float>) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(title)
                    .font(.caption)
                Spacer()
                Text(String(format: "%.3f", value.wrappedValue))
                    .monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: value, in: range)
                .labelsHidden()
                .frame(maxWidth: .infinity)
                .accessibilityLabel(title)
        }
        .padding(.vertical, 3)
    }
}

struct MetricsView: View {
    let snapshot: MetricsSnapshot

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 7) {
                row("源帧 / 已处理", "\(snapshot.source) / \(snapshot.processed)")
                row("唯一呈现帧", "\(snapshot.presented)")
                row("源帧序号缺口", "\(snapshot.sourceGaps)")
                row("相机丢帧回调", "\(snapshot.captureDrops)")
                row("FIFO 溢出", "\(snapshot.overflow)")
                row("实时取帧更新", "\(snapshot.supersededInputs)")
                row("未呈现帧", "\(snapshot.supersededDisplays)")
                row("错误", "\(snapshot.failures)")
                row("自动恢复", "\(snapshot.recoveries)")
                row("队列 / 峰值", "\(snapshot.queueDepth) / \(snapshot.maxQueueDepth)")
                row("模型耗时 P95", String(format: "%.2f ms", snapshot.inferenceP95))
                row("到达 → 显示 P50", String(format: "%.1f ms", snapshot.arrivalLatencyP50))
                row("到达 → 显示 P95", String(format: "%.1f ms", snapshot.arrivalLatencyP95))
            }
            .font(.caption.monospacedDigit())
            if !snapshot.lastError.isEmpty {
                Text(snapshot.lastError).font(.caption).foregroundStyle(.secondary)
            }
            Text("按独立源帧计数，以实际呈现回调计时；窗口需保持可见。")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func row(_ title: String, _ value: String) -> some View {
        GridRow {
            Text(title).foregroundStyle(.secondary)
            Text(value)
        }
    }
}
