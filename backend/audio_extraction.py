import librosa
import numpy as np
import soundfile as sf
import os

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def load_audio_file(self, file_path):
        """
        Loads an audio file and resamples it to the target sample rate.
        Supports .wav, .mp3, .flac, etc.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[ERROR] Audio file not found at: {file_path}")
            
        print(f"[INFO] Loading audio file: {file_path}")
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        return y, sr

    def extract_mfccs(self, y):
        """
        Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from the audio waveform.
        MFCCs capture the spectral envelope and are standard features for speech and noise characterization.
        """
        mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        return mfccs

    def extract_spectral_centroid(self, y):
        """
        Calculates the spectral centroid, which indicates where the center of mass
        of the power spectrum is located (useful for distinguishing speech/high frequency noise from silence).
        """
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)
        return centroid

    def process_audio(self, file_path):
        """
        Executes the full audio extraction pipeline.
        """
        try:
            y, sr = self.load_audio_file(file_path)
            
            # Compute feature shapes & statistics
            mfccs = self.extract_mfccs(y)
            centroid = self.extract_spectral_centroid(y)
            
            audio_duration = librosa.get_duration(y=y, sr=sr)
            
            print(f"[SUCCESS] Audio processed successfully!")
            print(f" - Duration: {audio_duration:.2f} seconds")
            print(f" - Sample Rate: {sr} Hz")
            print(f" - MFCC Matrix Shape: {mfccs.shape}")
            print(f" - Spectral Centroid Shape: {centroid.shape}")
            
            return {
                "waveform": y,
                "sample_rate": sr,
                "duration": audio_duration,
                "mfccs": mfccs,
                "spectral_centroid": centroid
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to process audio: {e}")
            return None

if __name__ == "__main__":
    # Example usage: Replace with an actual sample audio file path if testing locally
    extractor = AudioFeatureExtractor()
    sample_path = "sample_audio.wav"
    
    if os.path.exists(sample_path):
        extractor.process_audio(sample_path)
    else:
        print(f"[INFO] Place a sample file named '{sample_path}' in your root directory or pass a valid file path to test.")