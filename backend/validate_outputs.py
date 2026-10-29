import unittest
import numpy as np

# Import your module components for validation
from multimodal_agent import MultimodalAgent
from engagement_scoring import EngagementScoringEngine
from teaching_suggestion_logic import TeachingSuggestionEngine
from student_session_analytics import StudentSessionAnalyticsGenerator

class TestMultimodalPipelineEdgeCases(unittest.TestCase):
    
    def setUp(self):
        self.agent = MultimodalAgent()
        self.scorer = EngagementScoringEngine()
        self.suggestion_engine = TeachingSuggestionEngine()
        self.analytics_generator = StudentSessionAnalyticsGenerator(session_duration_seconds=3600)

    def test_empty_vision_and_audio_streams(self):
        """Edge Case 1: Handle completely empty streams (no faces detected, zero audio activity)."""
        empty_vision = {'tracked_faces': {}}
        empty_audio = {'speech_ratio': 0.0, 'noise_ratio': 0.0, 'silence_ratio': 100.0}
        
        result = self.agent.evaluate_engagement(empty_vision, empty_audio)
        
        self.assertEqual(result['overall_score'], 0.0)
        self.assertEqual(result['status'], "Low Engagement / Distracted")
        print("[TEST PASSED] Empty streams handled correctly without crashing.")

    def test_engagement_scoring_zero_division_guard(self):
        """Edge Case 2: Ensure zero division safety when attention ratio or audio ratios are zero."""
        zero_vision_metrics = {'attention_ratio': 0.0}
        zero_audio_metrics = {'audio_activity_ratio': 0.0}
        
        score_result = self.scorer.calculate_score(zero_vision_metrics, zero_audio_metrics)
        
        self.assertEqual(score_result['final_engagement_score'], 0.0)
        self.assertEqual(score_result['engagement_category'], "Low Engagement / At Risk")
        print("[TEST PASSED] Zero-division / zero-metrics score computation verified.")

    def test_high_noise_teaching_suggestion(self):
        """Edge Case 3: Verify that excessive background noise triggers classroom management advice."""
        high_noise_audio = {'speech_ratio': 20.0, 'noise_ratio': 45.0, 'silence_ratio': 35.0}
        normal_vision = {'attention_ratio': 0.8}
        
        suggestion = self.suggestion_engine.generate_suggestions(
            engagement_score=60.0, 
            audio_metrics=high_noise_audio, 
            vision_metrics=normal_vision
        )
        
        self.assertEqual(suggestion['suggestion_type'], "Classroom Management")
        self.assertIn("Background Noise", suggestion['recommended_action'])
        print("[TEST PASSED] High noise edge-case suggestion correctly generated.")

    def test_student_session_analytics_absent_edge_case(self):
        """Edge Case 4: Validate analytics generation for an absent or unconfirmed student."""
        absent_history = {'total_duration': 0.0, 'status': 'Absent'}
        
        report = self.analytics_generator.generate_student_report(student_id=999, tracking_history=absent_history)
        
        self.assertEqual(report['attendance_status'], 'Absent')
        self.assertEqual(report['session_attendance_percentage'], 0.0)
        self.assertEqual(report['performance_tier'], "Low Consistency / At Risk")
        print("[TEST PASSED] Absent student analytics edge case verified.")

if __name__ == "__main__":
    print("--- Starting Activity 6.3: Validation & Edge Case Tests ---")
    unittest.main()