import SwiftUI
import AppKit

struct VideoJobView: View {
    @Bindable var job: VideoJobStore

    var body: some View {
        VStack(alignment:.leading,spacing:18) {
            Label(job.processing ? "正在处理视频":"视频处理",systemImage:"film")
                .font(.title2.weight(.semibold))
            if job.processing {
                ProgressView(value:job.progress.fraction)
                Text(job.progress.pass==0 ? "正在准备模型与编码器":"第 \(job.progress.pass) 遍 · \(job.progress.frames) 帧")
                    .font(.callout).monospacedDigit().foregroundStyle(.secondary)
                Button("取消处理",role:.cancel){job.cancel()}
            } else if let result=job.result {
                Text(result.output.lastPathComponent).lineLimit(2)
                Text("\(result.width) × \(result.height) · \(result.bitDepth)-bit · \(result.frames) 帧 · \(result.passes) 次硬件编码")
                    .font(.callout).foregroundStyle(.secondary)
                if result.multiPassRequested && !result.multiPassSupported {
                    Text("所选硬件编码器当前提供单次编码。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                    Button("播放视频",systemImage:"play.rectangle") {NSWorkspace.shared.open(result.output)}
                    Button("在 Finder 中显示",systemImage:"folder") {NSWorkspace.shared.activateFileViewerSelecting([result.output])}
                    Spacer()
                    Button("对比画面") {job.isPresented=false}.keyboardShortcut(.defaultAction)
                }
            } else if let error=job.error {
                Text(error).textSelection(.enabled)
                Button("完成") {job.isPresented=false}.keyboardShortcut(.defaultAction)
            }
        }
        .padding(24).frame(width:520)
        .interactiveDismissDisabled(job.processing)
    }
}
