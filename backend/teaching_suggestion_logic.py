import time

class TeachingSuggestionEngine:
    def __init__(self):
        pass

    def generate_suggestions(self, engagement_score, audio_metrics, vision_metrics):
        """
        Evaluates session metrics to provide actionable, real-time recommendations 
        for instructors.
        
        Parameters:
        - engagement_score: float (0-100) representing overall class engagement.
        - audio_metrics: dict containing 'speech_ratio', 'noise_ratio', 'silence_ratio'.
        - vision_metrics: dict containing 'attention_ratio', 'present_count'.
        
        Returns:
        - dict containing the primary recommendation, category, and pedagogical rationale.
        """
        speech_ratio = audio_metrics.get('speech_ratio', 0.0)
        noise_ratio = audio_metrics.get('noise_ratio', 0.0)
        silence_ratio = audio_metrics.get('silence_ratio', 0.0)
        attention_ratio = vision_metrics.get('attention_ratio', 0.0) * 100.0

        # Rule-Based Pedagogy and Recommendation Logic
        if noise_ratio > 30.0:
            suggestion_type = "Classroom Management"
            action = "High Background Noise Detected"
            advice = "Consider pausing briefly or asking students to reduce side conversations to maintain acoustic clarity."
            rationale = f"Background noise level is at {noise_ratio:.1f}%, which hinders audio analysis and focus."

        elif silence_ratio > 60.0 and engagement_score < 50.0:
            suggestion_type = "Interactive Engagement"
            action = "Prolonged Silence / Low Participation"
            advice = "Introduce a quick poll, open-ended question, or breakout prompt to stimulate verbal participation."
            rationale = f"Silence ratio is high ({silence_ratio:.1f}%) with an overall engagement score of {engagement_score:.1f}."

        elif attention_ratio < 50.0:
            suggestion_type = "Attention Recovery"
            action = "Low Visual Attention"
            advice = "Change the presentation slide, summarize a key concept, or use a visual cue to re-engage student focus."
            rationale = f"Visual attention ratio has dropped to {attention_ratio:.1f}%."

        else:
            suggestion_type = "Pacing Maintenance"
            action = "Optimal Engagement"
            advice = "Current pacing and engagement levels are healthy. Continue with the planned lesson flow."
            rationale = "Both visual and acoustic feedback metrics are within optimal thresholds."

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "suggestion_type": suggestion_type,
            "recommended_action": action,
            "instructor_advice": advice,
            "pedagogical_rationale": rationale
        }

if __name__ == "__main__":
    # Test execution with mock data simulating a low-engagement/high-silence classroom scenario
    engine = TeachingSuggestionEngine()
    
    sample_engagement = 38.5
    sample_audio = {'speech_ratio': 10.0, 'noise_ratio': 5.0, 'silence_ratio': 85.0}
    sample_vision = {'attention_ratio': 0.45}
    
    result = engine.generate_suggestions(sample_engagement, sample_audio, sample_vision)
    
    print("--- Activity 5.1: Teaching Suggestion Report ---")
    for key, val in result.items():
        print(f" - {key.replace('_', ' ').capitalize()}: {val}")