import SwiftUI

struct WipeHandle: View {
    @Binding var value: Float

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Capsule().fill(.secondary.opacity(0.25)).frame(height: 3)
                Image(systemName: "arrow.left.and.right")
                    .font(.caption.weight(.semibold))
                    .frame(width: 36, height: 26)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 7))
                    .position(x: min(geometry.size.width - 20,
                                     max(20, geometry.size.width * CGFloat(value))), y: geometry.size.height / 2)
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { drag in
                value = Float(min(1, max(0, drag.location.x / max(1, geometry.size.width))))
            })
            .accessibilityElement()
            .accessibilityLabel("滑动对比分界")
            .accessibilityValue("\(Int(value * 100))%")
            .accessibilityAdjustableAction { direction in
                let delta: Float = direction == .increment ? 0.05 : -0.05
                value = min(1, max(0, value + delta))
            }
        }
    }
}
