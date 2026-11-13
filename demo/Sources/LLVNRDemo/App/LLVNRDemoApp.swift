import SwiftUI
import AppKit

final class DemoAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular);NSApp.activate(ignoringOtherApps:true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication)->Bool {true}
}

@main struct LLVNRDemoApp: App {
    @NSApplicationDelegateAdaptor(DemoAppDelegate.self) var delegate
    @State private var store=DemoStore()
    var body: some Scene {
        WindowGroup("LLVNR Demo") {
            ContentView(store:store).onAppear {
                if let index=CommandLine.arguments.firstIndex(of:"--install-kit"),index+1<CommandLine.arguments.count {
                    let kit=URL(fileURLWithPath:CommandLine.arguments[index+1])
                    var result:[String:Any]
                    do {
                        let loaded=try ModelStore.install(from:kit)
                        result=["passed":true,"models":loaded.models.count,"default_model_id":loaded.default_model_id,
                                "installed_root":ModelStore.installedRoot.path]
                    } catch {
                        result=["passed":false,"error":error.localizedDescription]
                    }
                    if let reportIndex=CommandLine.arguments.firstIndex(of:"--install-report"),
                       reportIndex+1<CommandLine.arguments.count {
                        let report=URL(fileURLWithPath:CommandLine.arguments[reportIndex+1])
                        try? JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:report)
                    } else {
                        let text=(try? JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]))
                            .flatMap{String(data:$0,encoding:.utf8)} ?? "{}"
                        FileHandle.standardOutput.write(Data((text+"\n").utf8))
                    }
                    NSApp.terminate(nil);return
                }
                if let index=CommandLine.arguments.firstIndex(of:"--check-catalog"),index+1<CommandLine.arguments.count {
                    let output=URL(fileURLWithPath:CommandLine.arguments[index+1])
                    do {
                        let result=try ModelCatalog.load().validationReport()
                        try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:output)
                    } catch {
                        let result:[String:Any]=["passed":false,"error":error.localizedDescription]
                        try? JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted]).write(to:output)
                    }
                    NSApp.terminate(nil);return
                }
                let options=RunOptions.parse()
                if options.autoStart || options.validationDirectory != nil || options.videoInput != nil {store.start(options)}
            }
        }.defaultSize(width:1180,height:740)
        .commands {
            CommandGroup(after:.newItem) {
                Button("启动本地相机"){store.start()}.disabled(store.busy || !store.canStart)
                Button("停止并完成测量"){store.stop()}.disabled(!store.busy)
                Divider()
                Button("处理视频…"){store.openVideo()}.disabled(store.videoJob.processing)
                    .keyboardShortcut("o",modifiers:[.command])
                Button("导入模型资产包…"){store.importModelKit()}
                    .disabled(store.busy || store.videoJob.processing)
            }
        }
    }
}
