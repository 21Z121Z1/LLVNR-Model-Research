import Foundation

private struct LatencyHistogram {
    // Fixed memory, 0.05ms buckets, including an overflow bucket at 1000ms.
    private var bins=[UInt64](repeating:0,count:20_001)
    private var count: UInt64=0
    private(set) var maximum=0.0
    mutating func add(_ value: Double) {
        guard value.isFinite,value>=0 else{return}
        bins[min(20_000,Int((value*20).rounded(.up)))]+=1;count+=1
        maximum=max(maximum,value)
    }
    func percentile(_ fraction: Double)->Double {
        guard count>0 else{return 0}
        let target=UInt64(Double(count-1)*fraction)+1
        var sum: UInt64=0
        for (i,n) in bins.enumerated() {sum+=n;if sum>=target{return Double(i)/20}}
        return maximum
    }
}

final class RunMetrics {
    let directory: URL
    private let lock=NSLock()
    private let writer=DispatchQueue(label:"llvnr.metrics.write",qos:.utility)
    private var file: FileHandle?
    private var state=MetricsSnapshot()
    private var firstPTS: Double?,firstInference: Double?,lastInference: Double?,firstPresentation: Double?,lastPresentation: Double?
    private var lastReceivedID = -1,lastProcessedID = -1,lastDisplayedID = -1
    private var missingProcess: [Int] = [],missingDisplay: [Int] = []
    private var inference=LatencyHistogram(),packing=LatencyHistogram(),post=LatencyHistogram()
    private var queued=LatencyHistogram(),arrival=LatencyHistogram(),capture=LatencyHistogram()
    private var previousSequence: Int64?, previousPTS: Double?
    private var closed=false
    private var details: [String: Any] = [:]
    private var activeIssues: Set<String> = []
    private var recoveryStarted: [String:Double] = [:]
    private var pausedAt: Double?
    private var pausedSeconds=0.0
    private var pauseCount=0
    private var lastPTS: Double?

    init(directory: URL?, manifest: ModelManifest, source: String) throws {
        let base=FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0]
            .appendingPathComponent("LLVNRDemo/Runs")
        self.directory=directory ?? base.appendingPathComponent(ISO8601DateFormatter().string(from:Date()).replacingOccurrences(of:":",with:"-"))
        try FileManager.default.createDirectory(at:self.directory,withIntermediateDirectories:true)
        let url=self.directory.appendingPathComponent("frames.jsonl")
        guard !FileManager.default.fileExists(atPath:url.path) else {throw PipelineError.failure("测量目录已存在；请使用新目录：\(self.directory.path)")}
        FileManager.default.createFile(atPath:url.path,contents:nil)
        file=try FileHandle(forWritingTo:url)
        state.runDirectory=self.directory.path
        details=["source":source,"width":1920,"height":1080,"requested_fps":30,
                 "node_count":manifest.node_count,"coefficient_group_count":manifest.coefficient_group_count,
                 "model_files":manifest.files,"source_model_sha256":manifest.source_model_sha256,
                 "os":ProcessInfo.processInfo.operatingSystemVersionString,
                 "host":ProcessInfo.processInfo.hostName,"start_utc":ISO8601DateFormatter().string(from:Date()),
                 "clock":"CM host clock; source PTS converted from capture synchronizationClock; CAMetalDrawable.presentedTime",
                 "source_clock_scope":source=="camera" ? "capture timestamp to drawable presentation; sensor exposure start not measured" : "paced replay release to drawable presentation; not physical camera latency",
                 "display_scope":"WindowServer drawable presentation timestamp; optical panel scan-out not measured",
                 "statistics":"whole-run counters and timing histograms; fixed 0.05ms buckets, bounded memory; all raw frame records retained in JSONL"]
        event("run_started",details)
    }
    func event(_ type: String,_ values: [String: Any] = [:]) {
        var row=values;row["event"]=type;row["host_time"]=hostSeconds()
        guard let data=try? JSONSerialization.data(withJSONObject:row,options:[.sortedKeys]) else {return}
        writer.async { [self] in
            do {try file?.write(contentsOf:data);try file?.write(contentsOf:Data([10]))}
            catch {fputs("LLVNR metrics write failed: \(error)\n",stderr)}
        }
    }
    func accepted(_ frame: SourceFrame,queueDepth: Int) {
        lock.lock()
        if let seq=frame.sourceSequence,let last=previousSequence,seq>last+1 {state.sourceGaps+=Int(seq-last-1)}
        else if frame.sourceSequence==nil,let pts=previousPTS {
            let delta=frame.pts-pts
            if delta>1.5/30 {state.sourceGaps+=max(0,Int((delta*30).rounded())-1)}
        }
        if let pts=previousPTS,frame.pts<=pts {state.failures+=1}
        previousSequence=frame.sourceSequence;previousPTS=frame.pts
        lastPTS=frame.pts
        if frame.id != lastReceivedID+1 {state.failures+=1}
        lastReceivedID=frame.id
        if firstPTS==nil {firstPTS=frame.pts}
        state.source+=1;state.queueDepth=queueDepth
        state.maxQueueDepth=max(state.maxQueueDepth,queueDepth)
        lock.unlock()
        var row: [String:Any] = ["id":frame.id,"pts":frame.pts,"received":frame.received,"source_host_time":frame.sourceTime,"queue_depth":queueDepth]
        if let seq=frame.sourceSequence {row["source_sequence"]=seq}
        if !frame.metadata.isEmpty {row["metadata"]=frame.metadata}
        event("received",row)
    }
    func queueDepth(_ depth: Int) {lock.lock();state.queueDepth=depth;lock.unlock()}
    func pause() {
        lock.lock()
        if pausedAt==nil {pausedAt=hostSeconds();pauseCount+=1}
        lock.unlock()
    }
    func resume() {
        lock.lock()
        if let start=pausedAt {pausedSeconds+=hostSeconds()-start;pausedAt=nil}
        previousSequence=nil;previousPTS=nil
        lock.unlock()
    }
    func processedFrame(_ timing: FrameTiming) {
        lock.lock()
        if timing.id<=lastProcessedID {state.failures+=1}
        if timing.id>lastProcessedID+1 {
            for id in (lastProcessedID+1)..<timing.id where missingProcess.count<128 {missingProcess.append(id)}
        }
        lastProcessedID=timing.id;state.processed+=1
        let activeTime=timing.inferred-pausedSeconds
        if firstInference==nil {firstInference=activeTime};lastInference=activeTime
        inference.add(timing.inferenceMS);packing.add((timing.packed-timing.inferenceStart)*1000)
        post.add((timing.postprocessed-timing.inferred)*1000);queued.add(timing.queueMS)
        lock.unlock()
        logTiming("processed",timing)
    }
    func presentedFrame(_ timing: FrameTiming) {
        lock.lock()
        if timing.id<=lastDisplayedID {state.failures+=1}
        if timing.id>lastDisplayedID+1 {
            for id in (lastDisplayedID+1)..<timing.id where missingDisplay.count<128 {missingDisplay.append(id)}
        }
        lastDisplayedID=timing.id;state.presented+=1
        let activeTime=timing.presented.map{$0-pausedSeconds}
        if firstPresentation==nil {firstPresentation=activeTime};lastPresentation=activeTime
        if let value=timing.arrivalToDisplayMS {arrival.add(value)}
        if let value=timing.sourceToDisplayMS {capture.add(value)}
        lock.unlock()
        logTiming("presented",timing)
    }
    private func logTiming(_ eventName: String,_ timing: FrameTiming) {
        if let data=try? JSONEncoder().encode(timing),let obj=try? JSONSerialization.jsonObject(with:data) as? [String:Any] {event(eventName,obj)}
    }
    func drop(_ type: String,values: [String:Any] = [:]) {
        lock.lock()
        if type=="capture_drop" {state.captureDrops+=1}
        else if type=="queue_overflow" {state.overflow+=1}
        else if type=="input_superseded" {state.supersededInputs+=1}
        else if type=="display_superseded" {state.supersededDisplays+=1}
        else {state.failures+=1}
        lock.unlock();event(type,values)
    }
    func recoverableIssue(_ key: String, message: String) {
        lock.lock()
        recoveryStarted[key]=nil
        let first=activeIssues.insert(key).inserted
        if first {state.failures+=1;state.lastError=message}
        lock.unlock()
        if first {event("recoverable_error",["stage":key,"error":message])}
    }
    func recovered(_ key: String, stableFor interval: Double=0) {
        lock.lock()
        if interval>0,activeIssues.contains(key) {
            let now=hostSeconds()
            let start=recoveryStarted[key] ?? now
            recoveryStarted[key]=start
            if now-start<interval {lock.unlock();return}
        }
        let wasActive=activeIssues.remove(key) != nil
        recoveryStarted[key]=nil
        if wasActive {state.recoveries+=1}
        lock.unlock()
        if wasActive {event("recovered",["stage":key])}
    }
    func snapshot() -> MetricsSnapshot {
        lock.lock();defer{lock.unlock()}
        var s=state
        if let first=firstInference,let last=lastInference,last>first {s.processedFPS=Double(s.processed-1)/(last-first)}
        if let first=firstPresentation,let last=lastPresentation,last>first {s.presentedFPS=Double(s.presented-1)/(last-first)}
        s.inferenceP95=inference.percentile(0.95)
        s.arrivalLatencyP50=arrival.percentile(0.5);s.arrivalLatencyP95=arrival.percentile(0.95)
        s.sourceLatencyP95=capture.percentile(0.95)
        s.verdict=s.failures>0 ? "存在运行错误" : "测量中"
        return s
    }
    func finish() throws -> [String:Any] {
        lock.lock()
        if closed {lock.unlock();return [:]}
        closed=true
        if lastProcessedID<lastReceivedID {for id in (lastProcessedID+1)...lastReceivedID where missingProcess.count<128 {missingProcess.append(id)}}
        if lastDisplayedID<lastReceivedID {for id in (lastDisplayedID+1)...lastReceivedID where missingDisplay.count<128 {missingDisplay.append(id)}}
        let sourcePause=(details["source"] as? String)=="camera" ? pausedSeconds:0
        let sourceDuration=max(0,(lastPTS ?? 0)-(firstPTS ?? 0)-sourcePause)
        lock.unlock()
        let s=snapshot()
        let sourceFPS=sourceDuration>0 ? Double(max(0,s.source-1))/sourceDuration : 0
        let captureMax=capture.maximum
        let pass=sourceDuration>=29.9 && sourceFPS>=29.9 && s.processedFPS>=29.9 && s.presentedFPS>=29.9 && s.source==s.processed && s.source==s.presented && s.overflow==0 && s.sourceGaps==0 && s.captureDrops==0 && s.supersededInputs==0 && s.supersededDisplays==0 && s.failures==0 && s.queueDepth==0 && s.sourceLatencyP95<200 && s.arrivalLatencyP95<66.7 && captureMax<300
        var report=details
        report.merge(["source_frames":s.source,"processed_frames":s.processed,"presented_unique_frames":s.presented,
            "source_sequence_gaps":s.sourceGaps,"capture_drop_callbacks":s.captureDrops,"queue_overflow":s.overflow,
            "failures":s.failures,"max_queue_depth":s.maxQueueDepth,"queue_at_end":s.queueDepth,
            "superseded_input_frames":s.supersededInputs,"superseded_display_frames":s.supersededDisplays,
            "recovery_count":s.recoveries,"last_error":s.lastError,
            "pause_count":pauseCount,"paused_seconds":pausedSeconds,
            "source_duration_seconds":sourceDuration,"source_fps":sourceFPS,
            "processed_completion_fps":s.processedFPS,"unique_presentation_fps":s.presentedFPS,
            "missing_processed_ids":missingProcess,"missing_presented_ids":missingDisplay,
            "inference_p50_ms":inference.percentile(0.5),"inference_p95_ms":s.inferenceP95,
            "pack_p95_ms":packing.percentile(0.95),
            "post_p95_ms":post.percentile(0.95),
            "queue_p95_ms":queued.percentile(0.95),
            "arrival_to_display_p50_ms":s.arrivalLatencyP50,"arrival_to_display_p95_ms":s.arrivalLatencyP95,
            "source_to_display_p95_ms":s.sourceLatencyP95,"source_to_display_max_ms":captureMax,
            "acceptance_pass":pass,
            "acceptance_rule":">=29.9s at >=29.9 fps, every source ID processed and presented once, zero source gaps/drop callbacks/overflow/errors, empty final queue; arrival-to-present p95<66.7ms; source-to-present p95<200ms max<300ms (application-set latency guardrails)",
            "complete_utc":ISO8601DateFormatter().string(from:Date())],uniquingKeysWith:{$1})
        event("run_finished",report)
        writer.sync {try? file?.synchronize()}
        try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:directory.appendingPathComponent("summary.json"))
        return report
    }
    deinit {try? file?.close()}
}
