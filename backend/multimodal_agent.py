import time

class MultimodalAgent:
    def __init__(self, attention_weight=0.6, audio_weight=0.4):
        self.attention_weight = attention_weight
        self.audio_weight = audio_weight

    def evaluate_engagement(self, vision_data, audio_data):
        """
        Evaluates overall classroom engagement by combining vision tracking metrics 
        and audio activity distribution.
        
        Parameters:
        - vision_data: dict containing active participants, present counts, etc.
        - audio_data: dict containing speech_ratio, noise_ratio, silence_ratio
        
        Returns:
        - dict containing overall engagement score, status, and explainable insights.
        """
        # 1. Compute Vision Component Score (Ratio of 'Present' vs total detected)
        total_tracked = len(vision_data.get('tracked_faces', {}))
        present_count = sum(1 for face in vision_data.get('tracked_faces', {}).values() if face.get('status') == 'Present')
        
        vision_score = (present_count / total_tracked) * 100 if total_tracked > 0 else 0.0

        # 2. Compute Audio Component Score (Active speech indicates positive participation; high silence/noise lowers score)
        speech_ratio = audio_data.get('speech_ratio', 0.0)
        noise_ratio = audio_data.get('noise_ratio', 0.0)
        
        # Audio score rewards balanced speech activity and penalizes excessive background noise
        audio_score = max(0.0, speech_ratio - (noise_ratio * 0.5))
        audio_score = min(100.0, audio_score * 2.5) # Scale to 0-100 range estimate

        # 3. Multimodal Signal Fusion
        overall_score = (vision_score * self.attention_weight) + (audio_score * self.audio_weight)

        # 4. Rule-Based Reasoning & Interpretability
        if overall_score >= 75.0:
            status = "Highly Engaged"
            insight = "Class participation and visual presence are optimal. Active discussion is ongoing."
            recommendation = "Maintain current pacing and interactive style."
        elif overall_score >= 40.0:
            status = "Stable / Moderate"
            insight = "Students are present, but active audio participation is moderate."
            recommendation = "Introduce an interactive question or prompt to boost engagement."
        else:
            status = "Low Engagement / Distracted"
            insight = "High silence/noise ratio detected combined with low visual attention markers."
            recommendation = "Consider re-capturing attention by switching topics or asking for student feedback."

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_score": round(overall_score, 2),
            "vision_score": round(vision_score, 2),
            "audio_score": round(audio_score, 2),
            "status": status,
            "insight": insight,
            "recommendation": recommendation
        }

if __name__ == "__main__":
    # Test simulation with mock multimodal data
    agent = MultimodalAgent()
    
    mock_vision = {
        'tracked_faces': {
            0: {'status': 'Present'},
            1: {'status': 'Present'},
            2: {'status': 'Unconfirmed'}
        }
    }
    
    mock_audio = {
        'speech_ratio': 45.0,
        'noise_ratio': 10.0,
        'silence_ratio': 45.0
    }
    
    result = agent.evaluate_engagement(mock_vision, mock_audio)
    
    print("--- Activity 4.1: Multimodal Agent Evaluation Report ---")
    for key, value in result.items():
        print(f" - {key.capitalize()}: {value}")