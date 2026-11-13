import SwiftUI

struct ModelSelectionView: View {
    @Bindable var store: DemoStore

    var body: some View {
        Group {
            if let catalog = store.catalog {
                Picker("镜头版本", selection: variantSelection) {
                    ForEach(variants(catalog), id: \.name) { model in
                        Text(variantTitle(model)).tag(model.name)
                    }
                }
                .disabled(store.busy || store.videoJob.processing)
                .help("选择对应镜头的降噪模型；不会切换或缩放 Mac 相机")
                Picker("模型尺寸", selection: canvasSelection) {
                    Text("2304 × 1296").tag(2304)
                    Text("2464 × 1388").tag(2464)
                }
                .disabled(store.busy || store.videoJob.processing)
                Button("模型与 tuning 详情", systemImage: "doc.text.magnifyingglass") {
                    store.showModelInspector = true
                }
            } else {
                Text("未安装模型资产包").foregroundStyle(.secondary)
                Button("导入模型资产包…", systemImage:"shippingbox") {
                    store.importModelKit()
                }
            }
            Button("导入其他模型资产包…", systemImage:"shippingbox.and.arrow.backward") {
                store.importModelKit()
            }
            .disabled(store.busy || store.videoJob.processing)
        }
        .sheet(isPresented: $store.showModelInspector) { ModelInspectorView(store: store) }
    }

    private func variants(_ catalog: ModelCatalog) -> [ModelCatalog.Entry] {
        catalog.models.filter { $0.width == 2304 }
    }

    private func variantTitle(_ entry: ModelCatalog.Entry) -> String {
        let family = ["SuperWide": "超广角", "Wide": "广角", "Tele": "长焦"][entry.family] ?? entry.family
        return family + " " + entry.zoom.replacingOccurrences(of: "x", with: "×")
    }

    private var variantSelection: Binding<String> {
        Binding(get: { store.selectedModel?.name ?? "" }, set: { name in
            if let entry = store.catalog?.models.first(where: {
                $0.name == name && $0.width == (store.selectedModel?.width ?? 2304)
            }) {
                store.selectedModelID = entry.id
            }
        })
    }

    private var canvasSelection: Binding<Int> {
        Binding(get: { store.selectedModel?.width ?? 2304 }, set: { width in
            if let entry = store.catalog?.models.first(where: {
                $0.width == width && $0.name == store.selectedModel?.name
            }) {
                store.selectedModelID = entry.id
            }
        })
    }
}

struct FirmwareInputsView: View {
    @Bindable var store: DemoStore

    var body: some View {
        Group {
                if store.availablePresets.count > 1 {
                    Picker("预设", selection: $store.selectedPresetID) {
                        ForEach(store.availablePresets) { preset in Text(preset.name).tag(preset.id) }
                    }
                }
                LabeledContent("原机增益") {
                    TextField("增益", value: $store.presetGain, format: .number)
                        .frame(width: 90)
                }
                LabeledContent("变焦输入") {
                    TextField("变焦", value: $store.presetZoom, format: .number)
                        .frame(width: 90)
                }
                Toggle("三脚架分支", isOn: $store.presetTripod)
                Button("应用到处理参数") { store.applyInputPreset() }
                    .disabled(!store.canStart)
        }
        .textFieldStyle(.roundedBorder)
        .disabled((store.busy && !store.isPaused) || store.videoJob.processing)
    }
}

private struct ModelInspectorView: View {
    @Bindable var store: DemoStore

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 12) {
                Image(systemName: "square.stack.3d.up")
                    .font(.title).foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 4) {
                    Text(store.selectedModel?.title ?? "模型目录").font(.title2.weight(.semibold))
                }
            }
            if let model = store.selectedModel {
                Text(model.status_detail).font(.callout).foregroundStyle(.secondary)
                LabeledContent("SHA-256") {
                    Text(model.sha256).font(.caption.monospaced()).textSelection(.enabled)
                }
                .font(.caption)
            }
            GroupBox {
                ScrollView {
                    Text(store.selectedPresetText).font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(8)
                }
            } label: {
                Text(store.selectedPreset?.name ?? "固件 tuning")
            }
            HStack {
                Button("显示模型文件", systemImage: "folder") { store.revealSelectedModel() }
                Button("显示 tuning", systemImage: "doc.text") { store.revealSelectedPreset() }
                Spacer()
                doneButton.keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 800, height: 640)
    }

    @ViewBuilder private var doneButton: some View {
        if #available(macOS 26.0, *) {
            Button("完成") { store.showModelInspector = false }.buttonStyle(.glassProminent)
        } else {
            Button("完成") { store.showModelInspector = false }.buttonStyle(.borderedProminent)
        }
    }
}
