import time

class StudentSessionAnalyticsGenerator:
    def __init__(self, session_duration_seconds=3600):
        self.session_duration = session_duration_seconds

    def generate_student_report(self, student_id, tracking_history):
        """
        Generates a comprehensive analytics report for an individual student 
        based on their tracking history throughout the session.
        
        Parameters:
        - student_id: int or str identifier of the student.
        - tracking_history: dict or list containing interaction logs, 
                             such as total visible time, timestamps, and status.
        
        Returns:
        - dict containing student-specific session performance metrics.
        """
        total_visible_time = tracking_history.get('total_duration', 0.0)
        attendance_status = tracking_history.get('status', 'Absent')
        
        # Calculate consistency ratio (Visible time vs standard session duration)
        presence_ratio = min(1.0, total_visible_time / self.session_duration)
        attendance_percentage = round(presence_ratio * 100.0, 2)

        # Determine qualitative engagement tier for the student
        if attendance_percentage >= 75.0 and attendance_status == 'Present':
            performance_tier = "High Consistency"
            remarks = "Regularly present and maintained stable visual attendance throughout the session."
        elif attendance_percentage >= 40.0:
            performance_tier = "Moderate Consistency"
            remarks = "Present for part of the session, but experienced intermittent disconnections or absence."
        else:
            performance_tier = "Low Consistency / At Risk"
            remarks = "Minimal visibility recorded during the session. Follow-up recommended."

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "student_id": student_id,
            "total_visible_duration_seconds": round(total_visible_time, 2),
            "attendance_status": attendance_status,
            "session_attendance_percentage": attendance_percentage,
            "performance_tier": performance_tier,
            "instructor_remarks": remarks
        }

if __name__ == "__main__":
    # Test execution with mock tracking data for an individual student
    generator = StudentSessionAnalyticsGenerator(session_duration_seconds=1800) # e.g., 30-min session
    
    mock_student_history = {
        'total_duration': 1500.0, # Visible for 1500 seconds
        'status': 'Present'
    }
    
    report = generator.generate_student_report(student_id=102, tracking_history=mock_student_history)
    
    print("--- Activity 5.2: Student Session Analytics Report ---")
    for key, val in report.items():
        print(f" - {key.replace('_', ' ').capitalize()}: {val}")