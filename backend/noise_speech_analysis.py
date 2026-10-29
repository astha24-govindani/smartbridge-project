import librosa
import numpy as np
import os

class NoiseAndSpeechAnalyzer:
    def __init__(self, sample_rate=22050, frame_length=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length

    def analyze_activity(self, file_path, rms_threshold=0.015):
        """
        Analyzes an audio file to differentiate between silence, background noise, 
        and active speech using Root Mean Square (RMS) energy and spectral centroids.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[ERROR] Audio file not found at: {file_path}")

        print(f"[INFO] Analyzing noise and speech activity for: {file_path}")
        y, sr = librosa.load(file_path, sr=self.sample_rate)

        # Calculate RMS Energy per frame
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        
        # Calculate Spectral Centroid per frame (helps differentiate speech/noise from low hums)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.frame_length, hop_length=self.hop_length)[0]

        # Time markers for each frame
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.hop_length)

        # Classification logic per frame
        activity_log = []
        speech_frames = 0
        noise_frames = 0
        silence_frames = 0

        for i, energy in enumerate(rms):
            centroid = spectral_centroids[i]
            
            if energy < rms_threshold:
                status = "Silence"
                silence_frames += 1
            elif centroid > 1000 and energy >= rms_threshold:
                # Active speech typically exhibits higher frequency concentration and energy
                status = "Speech Active"
                speech_frames += 1
            else:
                status = "Background Noise"
                noise_frames += 1
                
            activity_log.append((times[i], status, energy))

        total_frames = len(rms)
        speech_percentage = (speech_frames / total_frames) * 100 if total_frames > 0 else 0
        noise_percentage = (noise_frames / total_frames) * 100 if total_frames > 0 else 0
        silence_percentage = (silence_frames / total_frames) * 100 if total_frames > 0 else 0

        print(f"\n--- Activity Analysis Summary ---")
        print(f" - Active Speech Ratio: {speech_percentage:.2f}%")
        print(f" - Background Noise Ratio: {noise_percentage:.2f}%")
        print(f" - Silence Ratio: {silence_percentage:.2f}%")

        return {
            "speech_ratio": speech_percentage,
            "noise_ratio": noise_percentage,
            "silence_ratio": silence_percentage,
            "log": activity_log
        }

if __name__ == "__main__":
    analyzer = NoiseAndSpeechAnalyzer()
    
    sample_path = "sample_audio.wav"
    
    if os.path.exists(sample_path):
        analyzer.analyze_activity(sample_path)
    else:
        print(f"[INFO] Please place a file named '{sample_path}' inside your directory to execute analysis.")