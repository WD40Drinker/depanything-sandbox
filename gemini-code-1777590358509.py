import cv2
import numpy as np
import pyrealsense2 as rs
import onnxruntime as ort
import time

class ONNXMetricDepthPredictor:
    def __init__(self, onnx_path="depth_anything_v2_metric.onnx", input_size=(224, 224)):
        self.input_size = input_size
        
        print("[INFO] Loading ONNX model...")
        
        # MINIMAL LATENCY CHANGE: 
        # Attempt to use TensorRT or CUDA if available, falling back to CPU.
        providers = [
            'TensorrtExecutionProvider',
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, frame):
        h, w = self.input_size
        # Fast resizing and normalization
        img = cv2.resize(frame, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, 0).astype(np.float32)

    def infer(self, frame):
        inp = self.preprocess(frame)
        # Assuming the metric model outputs raw distance in meters
        depth_map = self.session.run(
            [self.output_name],
            {self.input_name: inp}
        )[0]
        
        return depth_map[0]

    def infer_realsense(self):
        # Configure RealSense Pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # We only enable the RGB stream. No stereoscopic depth.
        # Keeping resolution low (640x480) to minimize frame transfer latency.
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        print("[INFO] Starting RealSense pipeline...")
        pipeline.start(config)
        
        try:
            while True:
                # Wait for the next set of frames
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert RealSense frame to numpy array
                frame = np.asanyarray(color_frame.get_data())

                # Run inference
                start_time = time.time()
                depth_map = self.infer(frame)
                inference_time = (time.time() - start_time) * 1000 # ms
                
                # Resize depth map back to original frame size for accurate center mapping
                h, w = frame.shape[:2]
                depth_map_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

                # ==========================================
                # MEASURE CENTER DISTANCE
                # ==========================================
                cy, cx = h // 2, w // 2
                
                # Average a 10x10 pixel patch in the center to reduce noise/flicker
                center_patch = depth_map_resized[cy-5:cy+5, cx-5:cx+5]
                
                # Assuming the Metric model outputs in meters. Convert to cm.
                center_distance_m = np.mean(center_patch)
                center_distance_cm = center_distance_m * 100.0

                # Display Logic
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.rectangle(frame, (cx-5, cy-5), (cx+5, cy+5), (0, 255, 0), 2)
                
                text = f"{center_distance_cm:.1f} cm | {inference_time:.0f}ms"
                cv2.putText(frame, text, (cx - 80, cy - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("RealSense RGB + Depth Anything V2", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    predictor = ONNXMetricDepthPredictor(
        onnx_path="depth_anything_v2_metric.onnx", # Must be a metric variant
        input_size=(224, 224)
    )
    predictor.infer_realsense()