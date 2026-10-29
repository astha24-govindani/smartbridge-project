import time

class EngagementScoringEngine:
    def __init__(self, vision_weight=0.5, audio_weight=0.5):
        self.vision_weight = vision_weight
        self.audio_weight = audio_weight

    def calculate_score(self, vision_metrics, audio_metrics):
        """
        Computes a weighted engagement score combining visual attention and acoustic participation.
        
        Parameters:
        - vision_metrics: dict with keys like 'attention_ratio' (0.0 to 1.0)
        - audio_metrics: dict with keys like 'audio_activity_ratio' (0.0 to 1.0)
        
        Returns:
        - dict containing the final score, engagement category, and explanation.
        """
        # Extract metrics (defaulting to 0 if missing)
        vision_ratio = vision_metrics.get('attention_ratio', 0.0)
        audio_ratio = audio_metrics.get('audio_activity_ratio', 0.0)

        # Scale ratios to 0-100 percentage values
        vision_score = vision_ratio * 100.0
        audio_score = audio_ratio * 100.0

        # Compute Final Weighted Engagement Score
        final_score = (vision_score * self.vision_weight) + (audio_score * self.audio_weight)

        # Categorization & Explainable Insights Logic
        if final_score >= 75.0:
            category = "Highly Engaged"
            explanation = "Strong visual presence and active acoustic participation detected across the session."
        elif final_score >= 40.0:
            category = "Moderate Engagement"
            explanation = "Participants are visually present, but verbal or acoustic participation is subdued."
        else:
            category = "Low Engagement / At Risk"
            explanation = "Low visual attendance combined with minimal audio activity indicates potential distraction."

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_engagement_score": round(final_score, 2),
            "vision_component_score": round(vision_score, 2),
            "audio_component_score": round(audio_score, 2),
            "engagement_category": category,
            "explainable_insight": explanation
        }

if __name__ == "__main__":
    # Test execution with mock data
    scorer = EngagementScoringEngine()
    
    sample_vision = {'attention_ratio': 0.85}  # 85% attendance/attention
    sample_audio = {'audio_activity_ratio': 0.50} # 50% active speech/participation
    
    result = scorer.calculate_score(sample_vision, sample_audio)
    
    print("--- Activity 4.3: Engagement Scoring Report ---")
    for key, val in result.items():
        print(f" - {key.replace('_', ' ').capitalize()}: {val}")