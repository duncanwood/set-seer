//
//  CameraPreviewView.swift
//  SetSeer
//
//  Created by Duncan on 8/15/25.
//

// CameraPreviewView.swift

import SwiftUI
import AVFoundation

struct CameraPreviewView: UIViewRepresentable {
    
    class VideoPreviewView: UIView {
        override class var layerClass: AnyClass {
            AVCaptureVideoPreviewLayer.self
        }
        
        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            return layer as! AVCaptureVideoPreviewLayer
        }
    }
    
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> VideoPreviewView {
        let view = VideoPreviewView()
        view.backgroundColor = .black
        
        view.videoPreviewLayer.session = session
        // FIX: Change to .resizeAspect to see the entire camera feed.
        // This will create letterboxes (black bars) if the aspect ratios don't match, but it won't crop the image.
        view.videoPreviewLayer.videoGravity = .resizeAspect
        view.videoPreviewLayer.connection?.videoOrientation = .portrait
        
        return view
    }
    
    func updateUIView(_ uiView: VideoPreviewView, context: Context) {
        // No need to update the frame here, the view's autoresizing handles it.
    }
}
