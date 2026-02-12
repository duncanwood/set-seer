// ContentView.swift
import SwiftUI

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()

    var body: some View {
        ZStack {
            CameraPreviewView(session: cameraManager.getCaptureSession())
                .ignoresSafeArea()

            // Main Overlay
            GeometryReader { geometry in
                ForEach(cameraManager.predictions) { prediction in
                    let rect = transformBoundingBox(prediction.boundingBox, in: geometry.size)

                    // 1. If Showing Sets, draw the set borders
                    if cameraManager.showSets {
                        let participatingSets = cameraManager.validSets.filter { set in
                            set.cards.contains { $0.id == prediction.id }
                        }.sorted { $0.colorIndex < $1.colorIndex }
                        
                        if !participatingSets.isEmpty {
                            // "Progressively larger boundaries"
                            // We can use ZStack to layer them, or just separate Rectangles with different negative padding (stat frame growth)
                            // Let's draw them from outside in, or inside out?
                            // Inside out means largest last? No, largest first if filled, but these are strokes.
                            // If strokes, order doesn't matter much unless alpha.
                            // We want "boundaries" so valid sets are distinct.
                            // Let's iterate `enumerated()` so we can expand the frame.
                            
                            ForEach(Array(participatingSets.enumerated()), id: \.element.id) { index, set in
                                let expansion = CGFloat(index * 6)
                                let color = getSetColor(set.colorIndex)
                                
                                Rectangle()
                                    .stroke(color, lineWidth: 3)
                                    .frame(width: rect.width + expansion * 2, height: rect.height + expansion * 2)
                                    .position(x: rect.midX, y: rect.midY)
                            }
                        } else {
                            // Card not in any set - maybe gray it out or just show thin border?
                            // User didn't specify, but "show each valid set" implies others are less important.
                            // Let's keep a thin faint box so we know it's detected.
                            Rectangle()
                                .stroke(Color.gray.opacity(0.5), lineWidth: 1)
                                .frame(width: rect.width, height: rect.height)
                                .position(x: rect.midX, y: rect.midY)
                        }
                    } else {
                        // 2. Default View (just detection)
                        Rectangle()
                            .stroke(Color.green, lineWidth: 3)
                            .frame(width: rect.width, height: rect.height)
                            .position(x: rect.midX, y: rect.midY)
                    }

                    // Label (Optional in set mode? Maybe keep it small)
                    if !cameraManager.showSets {
                        Text("\(prediction.label) (\(String(format: "%.2f", prediction.confidence)))")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundColor(.black)
                            .padding(4)
                            .background(Color.green.opacity(0.8))
                            .cornerRadius(4)
                            .position(
                                x: rect.midX,
                                y: rect.minY > 20 ? rect.minY - 15 : rect.maxY + 15
                            )
                    }
                }
            }
            
            // HUD Layer (Top)
            VStack {
                // Top Bar
                HStack {
                    Text("Sets: \(cameraManager.validSets.count)")
                        .font(.title2.bold())
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(10)
                    
                    Spacer()
                    
                    Button(action: {
                        withAnimation {
                            cameraManager.showSets.toggle()
                        }
                    }) {
                        Text(cameraManager.showSets ? "Hide Sets" : "Show Sets")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .background(cameraManager.showSets ? Color.blue : Color.gray.opacity(0.8))
                            .cornerRadius(10)
                    }
                }
                .padding(.top, 50)
                .padding(.horizontal)
                
                Spacer() // Pushes content to top/bottom
                
                // Camera switcher (Bottom)
                if cameraManager.availableCameras.count > 1 {
                    HStack(spacing: 12) {
                        ForEach(cameraManager.availableCameras) { camera in
                            Button(action: {
                                cameraManager.switchCamera(to: camera)
                            }) {
                                Text(camera.displayName)
                                    .font(.system(size: 14, weight: .semibold))
                                    .foregroundColor(cameraManager.currentCamera == camera ? .black : .white)
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 10)
                                    .background(
                                        cameraManager.currentCamera == camera 
                                            ? Color.yellow 
                                            : Color.black.opacity(0.6)
                                    )
                                    .clipShape(Capsule())
                            }
                        }
                    }
                    .padding(.bottom, 40)
                }
            }

            if let error = cameraManager.error {
                VStack {
                    Spacer()
                    Text("Error: \(error.localizedDescription)")
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.red.opacity(0.8))
                        .cornerRadius(10)
                        .padding()
                }
            }
        }
        .onAppear {
            cameraManager.configureSession()
            cameraManager.startSession()
        }
        .onDisappear {
            cameraManager.stopSession()
        }
    }

    private func transformBoundingBox(_ normalizedRect: CGRect, in viewSize: CGSize) -> CGRect {
        // YOLO output is already in top-left origin (same as SwiftUI)
        // Just scale to view size
        return CGRect(
            x: normalizedRect.origin.x * viewSize.width,
            y: normalizedRect.origin.y * viewSize.height,
            width: normalizedRect.width * viewSize.width,
            height: normalizedRect.height * viewSize.height
        )
    }
    
    private func getSetColor(_ index: Int) -> Color {
        let colors: [Color] = [.red, .blue, .orange, .purple, .pink, .cyan, .yellow, .mint, .indigo]
        return colors[index % colors.count]
    }
}
