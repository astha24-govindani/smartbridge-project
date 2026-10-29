import numpy as np
import time

class MultimodalSignalFusion:
    def __init__(self, time_window_seconds=1.0):
        self.time_window = time_window_seconds

    def fuse_signals(self, vision_stream_chunk, audio_stream_chunk):
        """
        Synchronizes and fuses asynchronous vision tracking metrics and audio features 
        over a defined time window to create a unified multimodal payload.
        
        Parameters:
        - vision_stream_chunk: list of dicts or state logs from face detection/tracking within the window.
        - audio_stream_chunk: list of audio frame features (RMS, spectral centroids, or ratios) within the window.
        
        Returns:
        - dict containing the fused timestamp, aggregated vision metrics, and aggregated audio metrics.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Aggregate Vision Metrics over the window
        total_detections = len(vision_stream_chunk)
        present_count = sum(1 for v in vision_stream_chunk if v.get('status') == 'Present')
        
        vision_summary = {
            "total_faces_detected": total_detections,
            "present_count": present_count,
            "attention_ratio": (present_count / total_detections) if total_detections > 0 else 0.0
        }

        # 2. Aggregate Audio Metrics over the window
        if len(audio_stream_chunk) > 0:
            avg_rms = np.mean([a.get('rms', 0.0) for a in audio_stream_chunk])
            speech_active_count = sum(1 for a in audio_stream_chunk if a.get('status') == 'Speech Active')
            audio_activity_ratio = speech_active_count / len(audio_stream_chunk)
        else:
            avg_rms = 0.0
            audio_activity_ratio = 0.0

        audio_summary = {
            "average_rms": float(avg_rms),
            "audio_activity_ratio": float(audio_activity_ratio)
        }

        # 3. Form Fused Multimodal Payload
        fused_payload = {
            "timestamp": timestamp,
            "vision_metrics": vision_summary,
            "audio_metrics": audio_summary,
            "fusion_status": "Synchronized"
        }

        return fused_payload

if __name__ == "__main__":
    # Test simulation with dummy streaming chunks
    fusion_engine = MultimodalSignalFusion()
    
    mock_vision_chunk = [
        {'id': 0, 'status': 'Present'},
        {'id': 1, 'status': 'Present'},
        {'id': 2, 'status': 'Unconfirmed'}
    ]
    
    mock_audio_chunk = [
        {'rms': 0.02, 'status': 'Speech Active'},
        {'rms': 0.01, 'status': 'Background Noise'},
        {'rms': 0.005, 'status': 'Silence'},
        {'rms': 0.025, 'status': 'Speech Active'}
    ]
    
    fused_data = fusion_engine.fuse_signals(mock_vision_chunk, mock_audio_chunk)
    
    print("--- Activity 4.2: Multimodal Signal Fusion Report ---")
    for key, value in fused_data.items():
        print(f" - {key.capitalize()}: {value}")