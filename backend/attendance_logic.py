import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

class AttendanceTracker:
    def __init__(self, presence_threshold_seconds=5.0):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # Tracking state parameters
        self.tracked_faces = {}  # {object_id: {'first_seen': timestamp, 'last_seen': timestamp, 'total_duration': float, 'status': str}}
        self.next_object_id = 0
        self.max_disappeared = 15
        self.disappeared_counts = {}
        self.centroids = {}
        
        # Threshold required to mark attendance as "Present"
        self.presence_threshold_seconds = presence_threshold_seconds

    def register(self, centroid):
        object_id = self.next_object_id
        current_time = time.time()
        
        self.centroids[object_id] = centroid
        self.disappeared_counts[object_id] = 0
        self.tracked_faces[object_id] = {
            'first_seen': current_time,
            'last_seen': current_time,
            'total_duration': 0.0,
            'status': 'Unconfirmed'
        }
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.centroids:
            del self.centroids[object_id]
        if object_id in self.disappeared_counts:
            del self.disappeared_counts[object_id]

    def update(self, rects):
        current_time = time.time()
        
        if len(rects) == 0:
            for object_id in list(self.disappeared_counts.keys()):
                self.disappeared_counts[object_id] += 1
                if self.disappeared_counts[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.tracked_faces

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            input_centroids[i] = (int((startX + endX) / 2.0), int((startY + endY) / 2.0))

        if len(self.centroids) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.centroids.keys())
            object_centroids = list(self.centroids.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[0:len(rows)]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.centroids[object_id] = input_centroids[col]
                self.disappeared_counts[object_id] = 0
                
                # Update duration and status
                face_record = self.tracked_faces[object_id]
                face_record['last_seen'] = current_time
                face_record['total_duration'] = current_time - face_record['first_seen']
                
                if face_record['total_duration'] >= self.presence_threshold_seconds:
                    face_record['status'] = 'Present'

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared_counts[object_id] += 1
                    if self.disappeared_counts[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.tracked_faces

def run_attendance_system(source=0):
    cap = cv2.VideoCapture(source)
    # 5 seconds presence threshold for testing demo
    tracker = AttendanceTracker(presence_threshold_seconds=5.0)
    
    print("[INFO] Activity 2.2 Attendance Logic Started. Press 'q' to quit and export logs.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tracker.face_detection.process(rgb_frame)

        rects = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                startX = int(bbox.xmin * w)
                startY = int(bbox.ymin * h)
                endX = int(startX + bbox.width * w)
                endY = int(startY + bbox.height * h)
                rects.append((max(0, startX), max(0, startY), min(w, endX), min(h, endY)))
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

        tracked_objects = tracker.update(rects)

        # Display Attendance Info Overlay on Frame
        for object_id, info in tracked_objects.items():
            if object_id in tracker.centroids:
                cX, cY = tracker.centroids[object_id]
                status = info['status']
                duration = int(info['total_duration'])
                
                color = (0, 255, 0) if status == 'Present' else (0, 165, 255)
                label = f"ID {object_id}: {status} ({duration}s)"
                
                cv2.putText(frame, label, (cX - 40, cY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (cX, cY), 4, color, -1)

        cv2.imshow("Activity 2.2 - Presence-Based Attendance Logic", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Print Final Attendance Summary Log
    print("\n--- Final Attendance Summary Report ---")
    for obj_id, data in tracker.tracked_faces.items():
        print(f"Participant ID: {obj_id} | Status: {data['status']} | Total Time Visible: {int(data['total_duration'])}s")

if __name__ == "__main__":
    run_attendance_system(0)