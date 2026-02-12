import AVFoundation
import Vision
import UIKit
import CoreImage
import CoreML

struct CameraInfo: Identifiable, Equatable {
    let id: String
    let device: AVCaptureDevice
    var displayName: String {
        switch device.deviceType {
        case .builtInWideAngleCamera:
            return device.position == .front ? "Front" : "Wide"
        case .builtInUltraWideCamera:
            return "Ultra Wide"
        case .builtInTelephotoCamera:
            return "Telephoto"
        default:
            return device.localizedName
        }
    }
    
    static func == (lhs: CameraInfo, rhs: CameraInfo) -> Bool {
        lhs.id == rhs.id
    }
}

class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var predictions: [Prediction] = []
    @Published var validSets: [SetResult] = []
    @Published var showSets: Bool = false
    @Published var error: Error?
    @Published var availableCameras: [CameraInfo] = []
    @Published var currentCamera: CameraInfo?

    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.outlier.SETCardSolver.sessionQueue")
    private var currentInput: AVCaptureDeviceInput?

    private var detectionModel: VNCoreMLModel?
    private var mlModel: MLModel?

    private var isProcessing = false
    private var frameCounter = 0
    
    // Current frame dimensions (updated dynamically)
    private var frameWidth: CGFloat = 720
    private var frameHeight: CGFloat = 1280
    
    // Throttle factor: process only every Nth frame
    private let frameStride = 20
    
    // YOLO config
    private let inputSize: CGFloat = 640
    private let confidenceThreshold: Float = 0.3 
    private let iouThreshold: Float = 0.45
    
    // Class names - must match the order from training
    private let classNames: [String] = [
        "1-green-empty-diamond", "1-green-empty-oval", "1-green-empty-squiggle",
        "1-green-solid-diamond", "1-green-solid-oval", "1-green-solid-squiggle",
        "1-green-striped-diamond", "1-green-striped-oval", "1-green-striped-squiggle",
        "1-purple-empty-diamond", "1-purple-empty-oval", "1-purple-empty-squiggle",
        "1-purple-solid-diamond", "1-purple-solid-oval", "1-purple-solid-squiggle",
        "1-purple-striped-diamond", "1-purple-striped-oval", "1-purple-striped-squiggle",
        "1-red-empty-diamond", "1-red-empty-oval", "1-red-empty-squiggle",
        "1-red-solid-diamond", "1-red-solid-oval", "1-red-solid-squiggle",
        "1-red-striped-diamond", "1-red-striped-oval", "1-red-striped-squiggle",
        "2-green-empty-diamond", "2-green-empty-oval", "2-green-empty-squiggle",
        "2-green-solid-diamond", "2-green-solid-oval", "2-green-solid-squiggle",
        "2-green-striped-diamond", "2-green-striped-oval", "2-green-striped-squiggle",
        "2-purple-empty-diamond", "2-purple-empty-oval", "2-purple-empty-squiggle",
        "2-purple-solid-diamond", "2-purple-solid-oval", "2-purple-solid-squiggle",
        "2-purple-striped-diamond", "2-purple-striped-oval", "2-purple-striped-squiggle",
        "2-red-empty-diamond", "2-red-empty-oval", "2-red-empty-squiggle",
        "2-red-solid-diamond", "2-red-solid-oval", "2-red-solid-squiggle",
        "2-red-striped-diamond", "2-red-striped-oval", "2-red-striped-squiggle",
        "3-green-empty-diamond", "3-green-empty-oval", "3-green-empty-squiggle",
        "3-green-solid-diamond", "3-green-solid-oval", "3-green-solid-squiggle",
        "3-green-striped-diamond", "3-green-striped-oval", "3-green-striped-squiggle",
        "3-purple-empty-diamond", "3-purple-empty-oval", "3-purple-empty-squiggle",
        "3-purple-solid-diamond", "3-purple-solid-oval", "3-purple-solid-squiggle",
        "3-purple-striped-diamond", "3-purple-striped-oval", "3-purple-striped-squiggle",
        "3-red-empty-diamond", "3-red-empty-oval", "3-red-empty-squiggle",
        "3-red-solid-diamond", "3-red-solid-oval", "3-red-solid-squiggle",
        "3-red-striped-diamond", "3-red-striped-oval", "3-red-striped-squiggle"
    ]

    override init() {
        super.init()
        setupModels()
        discoverCameras()
    }

    private func setupModels() {
        do {
            print("[DEBUG] Loading YOLO model...")
            let modelURL = Bundle.main.url(forResource: "setDetectionYOLO", withExtension: "mlmodelc")!
            mlModel = try MLModel(contentsOf: modelURL)
            detectionModel = try VNCoreMLModel(for: mlModel!)

            print("[DEBUG] ✅ Model loaded successfully.")
        } catch {
            print("[DEBUG] 🔴 FAILED to load model: \(error.localizedDescription)")
            DispatchQueue.main.async { self.error = error }
        }
    }
    
    // MARK: - Camera Discovery
    private func discoverCameras() {
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [
                .builtInWideAngleCamera,
                .builtInUltraWideCamera,
                .builtInTelephotoCamera
            ],
            mediaType: .video,
            position: .unspecified
        )
        
        let cameras = discoverySession.devices.map { device in
            CameraInfo(id: device.uniqueID, device: device)
        }
        
        // Update synchronously since this is called from init
        self.availableCameras = cameras
        print("[DEBUG] Found \(cameras.count) cameras: \(cameras.map { $0.displayName })")
    }

    // MARK: - Session
    func configureSession() {
        sessionQueue.async {
            self.captureSession.beginConfiguration()
            self.captureSession.sessionPreset = .hd1280x720
            
            // Find default back wide-angle camera directly from available cameras
            let cameras = self.availableCameras
            let defaultCamera = cameras.first { 
                $0.device.position == .back && $0.device.deviceType == .builtInWideAngleCamera 
            } ?? cameras.first
            
            guard let camera = defaultCamera,
                  let input = try? AVCaptureDeviceInput(device: camera.device) else {
                print("[DEBUG] 🔴 Failed to get camera device, available: \(cameras.count)")
                self.captureSession.commitConfiguration()
                return
            }
            
            if self.captureSession.canAddInput(input) { 
                self.captureSession.addInput(input) 
                self.currentInput = input
                DispatchQueue.main.async {
                    self.currentCamera = camera
                }
            }

            self.videoOutput.setSampleBufferDelegate(self, queue: self.sessionQueue)
            self.videoOutput.alwaysDiscardsLateVideoFrames = true
            if self.captureSession.canAddOutput(self.videoOutput) {
                self.captureSession.addOutput(self.videoOutput)
            }

            self.configureVideoConnection()

            self.captureSession.commitConfiguration()
            print("[DEBUG] ✅ Camera session configured with \(camera.displayName)")
        }
    }
    
    private func configureVideoConnection() {
        // Set fixed rotation to portrait (90 degrees) regardless of device orientation
        if let conn = self.videoOutput.connection(with: .video) {
            if conn.isVideoRotationAngleSupported(90) {
                conn.videoRotationAngle = 90
            }
            // Mirror front camera
            if let currentCamera = currentInput?.device, currentCamera.position == .front {
                conn.isVideoMirrored = true
            }
        }
    }
    
    // MARK: - Camera Switching
    func switchCamera(to camera: CameraInfo) {
        sessionQueue.async {
            self.captureSession.beginConfiguration()
            
            // Remove current input
            if let currentInput = self.currentInput {
                self.captureSession.removeInput(currentInput)
            }
            
            // Add new input
            guard let newInput = try? AVCaptureDeviceInput(device: camera.device) else {
                print("[DEBUG] 🔴 Failed to create input for \(camera.displayName)")
                self.captureSession.commitConfiguration()
                return
            }
            
            if self.captureSession.canAddInput(newInput) {
                self.captureSession.addInput(newInput)
                self.currentInput = newInput
                DispatchQueue.main.async {
                    self.currentCamera = camera
                    self.predictions = [] // Clear predictions during switch
                }
            }
            
            self.configureVideoConnection()
            self.captureSession.commitConfiguration()
            print("[DEBUG] ✅ Switched to \(camera.displayName)")
        }
    }

    func startSession() {
        sessionQueue.async {
            if !self.captureSession.isRunning {
                print("[DEBUG] ▶️ Starting camera session")
                self.captureSession.startRunning()
            }
        }
    }

    func stopSession() {
        sessionQueue.async {
            if self.captureSession.isRunning {
                print("[DEBUG] ⏹ Stopping camera session")
                self.captureSession.stopRunning()
            }
        }
    }

    func getCaptureSession() -> AVCaptureSession { captureSession }

    // MARK: - Delegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        frameCounter += 1
        if frameCounter % frameStride != 0 { return } // Throttle
        
        guard !isProcessing,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Get actual frame dimensions (after rotation is applied)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        frameWidth = CGFloat(width)
        frameHeight = CGFloat(height)
        
        print("[DEBUG] 📸 Processing frame #\(frameCounter), size: \(width)x\(height)")
        isProcessing = true
        performDetection(on: pixelBuffer)
    }

    // MARK: - Detection with raw output parsing
    private func performDetection(on pixelBuffer: CVPixelBuffer) {
        guard let detectionModel = detectionModel else { isProcessing = false; return }

        let request = VNCoreMLRequest(model: detectionModel) { [weak self] request, error in
            guard let self = self else { return }
            if let error = error {
                print("[DEBUG] 🔴 Detection error: \(error.localizedDescription)")
                self.isProcessing = false
                return
            }

            // Try to get results as VNCoreMLFeatureValueObservation (raw output)
            if let results = request.results as? [VNCoreMLFeatureValueObservation],
               let firstResult = results.first,
               let multiArray = firstResult.featureValue.multiArrayValue {
                let predictions = self.parseYOLOOutput(multiArray)
                print("[DEBUG] 🟢 Parsed \(predictions.count) predictions after NMS")
                
                // Process Sets
                let cards = predictions.compactMap { SetGameEngine.parse(prediction: $0) }
                var sets = SetGameEngine.findSets(cards: cards)
                
                // Assign color indices to sets
                for i in 0..<sets.count {
                    sets[i].colorIndex = i
                }
                
                print("[DEBUG] 🃏 Found \(sets.count) valid sets")
                
                DispatchQueue.main.async {
                    self.predictions = predictions
                    self.validSets = sets
                    self.isProcessing = false
                }
            } else if let results = request.results as? [VNRecognizedObjectObservation] {
                // Fallback if Vision pipeline handles it
                let filtered = results.filter { $0.confidence >= self.confidenceThreshold }
                print("[DEBUG] 🟢 VNRecognizedObjectObservation: \(filtered.count) detections")
                let newPredictions = filtered.map { observation -> Prediction in
                    let label = observation.labels.first?.identifier ?? "unknown"
                    return Prediction(
                        boundingBox: observation.boundingBox,
                        label: label.replacingOccurrences(of: "-", with: " ").capitalized,
                        confidence: observation.confidence
                    )
                }
                
                // Process Sets
                let cards = newPredictions.compactMap { SetGameEngine.parse(prediction: $0) }
                var sets = SetGameEngine.findSets(cards: cards)
                for i in 0..<sets.count { sets[i].colorIndex = i }
                
                DispatchQueue.main.async {
                    self.predictions = newPredictions
                    self.validSets = sets
                    self.isProcessing = false
                }
            } else {
                print("[DEBUG] ⚠️ No recognized output format")
                self.isProcessing = false
            }
        }

        // Use scaleFit to letterbox the image (preserve aspect ratio)
        request.imageCropAndScaleOption = .scaleFit

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
        try? handler.perform([request])
    }
    
    // MARK: - YOLO Output Parsing
    // Output shape: (1, 85, 8400) where 85 = 4 (bbox) + 81 (classes)
    // The model receives a letterboxed image, so we need to adjust coordinates
    private func parseYOLOOutput(_ output: MLMultiArray) -> [Prediction] {
        let numClasses = classNames.count  // 81
        let numAnchors = output.shape[2].intValue  // 8400
        let stride = output.strides[1].intValue
        
        // Calculate letterbox padding dynamically based on actual frame dimensions
        // frameWidth and frameHeight are the actual rotated frame dimensions
        let imageAspect = Float(frameWidth / frameHeight)
        let modelSize = Float(inputSize)
        
        // Determine scale factor and offsets based on which dimension is limiting
        var xOffset: Float = 0
        var yOffset: Float = 0
        
        if imageAspect < 1.0 {
            // Portrait: width < height, so width is scaled smaller
            // Image fills height, width is padded
            let scale = modelSize / Float(frameHeight)
            let scaledWidth = Float(frameWidth) * scale
            xOffset = (modelSize - scaledWidth) / 2
        } else {
            // Landscape: width > height, so height is scaled smaller
            // Image fills width, height is padded
            let scale = modelSize / Float(frameWidth)
            let scaledHeight = Float(frameHeight) * scale
            yOffset = (modelSize - scaledHeight) / 2
        }
        
        var candidates: [(box: CGRect, classIdx: Int, confidence: Float)] = []
        
        // Pointer to raw data for faster access
        let pointer = UnsafeMutablePointer<Float>(OpaquePointer(output.dataPointer))
        
        for i in 0..<numAnchors {
            // YOLOv8 format: [cx, cy, w, h, class_scores...]
            let cx = pointer[0 * stride + i]
            let cy = pointer[1 * stride + i]
            let w = pointer[2 * stride + i]
            let h = pointer[3 * stride + i]
            
            // Find best class
            var maxScore: Float = 0
            var maxClassIdx = 0
            for c in 0..<numClasses {
                let score = pointer[(4 + c) * stride + i]
                if score > maxScore {
                    maxScore = score
                    maxClassIdx = c
                }
            }
            
            // Filter by confidence
            if maxScore >= confidenceThreshold {
                // Convert from model coordinates to normalized image coordinates
                // 1. Remove letterbox offset
                // 2. Scale to 0-1 range based on actual image dimensions
                
                let adjustedCx = (cx - xOffset) / (modelSize - 2 * xOffset)
                let adjustedCy = (cy - yOffset) / (modelSize - 2 * yOffset)
                let adjustedW = w / (modelSize - 2 * xOffset)
                let adjustedH = h / (modelSize - 2 * yOffset)
                
                // Convert center to corner format
                let x1 = adjustedCx - adjustedW / 2
                let y1 = adjustedCy - adjustedH / 2
                
                // Clamp to valid range
                let clampedX = max(0, min(1, x1))
                let clampedY = max(0, min(1, y1))
                let clampedW = max(0, min(1 - clampedX, adjustedW))
                let clampedH = max(0, min(1 - clampedY, adjustedH))
                
                let box = CGRect(
                    x: CGFloat(clampedX),
                    y: CGFloat(clampedY),
                    width: CGFloat(clampedW),
                    height: CGFloat(clampedH)
                )
                
                candidates.append((box: box, classIdx: maxClassIdx, confidence: maxScore))
            }
        }
        
        print("[DEBUG] Found \(candidates.count) candidates before NMS")
        
        // Apply NMS
        let nmsResults = applyNMS(candidates)
        
        return nmsResults.map { candidate in
            let label = classNames[candidate.classIdx]
            return Prediction(
                boundingBox: candidate.box,
                label: label.replacingOccurrences(of: "-", with: " ").capitalized,
                confidence: candidate.confidence
            )
        }
    }
    
    // MARK: - Non-Maximum Suppression
    private func applyNMS(_ candidates: [(box: CGRect, classIdx: Int, confidence: Float)]) -> [(box: CGRect, classIdx: Int, confidence: Float)] {
        guard !candidates.isEmpty else { return [] }
        
        // Sort by confidence descending
        let sorted = candidates.sorted { $0.confidence > $1.confidence }
        var selected: [(box: CGRect, classIdx: Int, confidence: Float)] = []
        var suppressed = [Bool](repeating: false, count: sorted.count)
        
        for i in 0..<sorted.count {
            if suppressed[i] { continue }
            
            selected.append(sorted[i])
            
            for j in (i + 1)..<sorted.count {
                if suppressed[j] { continue }
                
                let iou = calculateIoU(sorted[i].box, sorted[j].box)
                if iou > CGFloat(iouThreshold) {
                    suppressed[j] = true
                }
            }
        }
        
        return selected
    }
    
    private func calculateIoU(_ boxA: CGRect, _ boxB: CGRect) -> CGFloat {
        let intersection = boxA.intersection(boxB)
        if intersection.isNull { return 0 }
        
        let intersectionArea = intersection.width * intersection.height
        let unionArea = boxA.width * boxA.height + boxB.width * boxB.height - intersectionArea
        
        return unionArea > 0 ? intersectionArea / unionArea : 0
    }
}
