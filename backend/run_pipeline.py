import cv2
import time
import os
import numpy as np

class MultimodalPipelineRunner:
    def __init__(self, camera_index=0, presence_threshold=3.0):
        self.camera_index = camera_index
        self.presence_threshold = presence_threshold
        
        # Initialize MediaPipe Face Detection
        import mediapipe as mp
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # Tracking State Variables
        self.tracked_faces = {}
        self.centroids = {}
        self.disappeared_counts = {}
        self.next_object_id = 0
        self.max_disappeared = 15

    def update_vision_tracker(self, rects, current_time):
        if len(rects) == 0:
            for obj_id in list(self.disappeared_counts.keys()):
                self.disappeared_counts[obj_id] += 1
                if self.disappeared_counts[obj_id] > self.max_disappeared:
                    del self.centroids[obj_id]
                    del self.disappeared_counts[obj_id]
            return self.tracked_faces

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (sx, sy, ex, ey) in enumerate(rects):
            input_centroids[i] = (int((sx + ex) / 2.0), int((sy + ey) / 2.0))

        if len(self.centroids) == 0:
            for i in range(len(input_centroids)):
                obj_id = self.next_object_id
                self.centroids[obj_id] = input_centroids[i]
                self.disappeared_counts[obj_id] = 0
                self.tracked_faces[obj_id] = {
                    'first_seen': current_time,
                    'total_duration': 0.0,
                    'status': 'Unconfirmed'
                }
                self.next_object_id += 1
        else:
            object_ids = list(self.centroids.keys())
            object_centroids = list(self.centroids.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[0:len(rows)]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                obj_id = object_ids[row]
                self.centroids[obj_id] = input_centroids[col]
                self.disappeared_counts[obj_id] = 0
                
                record = self.tracked_faces[obj_id]
                record['total_duration'] = current_time - record['first_seen']
                if record['total_duration'] >= self.presence_threshold:
                    record['status'] = 'Present'

                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(D.shape[0])).difference(used_rows):
                obj_id = object_ids[row]
                self.disappeared_counts[obj_id] += 1
                if self.disappeared_counts[obj_id] > self.max_disappeared:
                    del self.centroids[obj_id]
                    del self.disappeared_counts[obj_id]

            for col in set(range(D.shape[1])).difference(used_cols):
                obj_id = self.next_object_id
                self.centroids[obj_id] = input_centroids[col]
                self.disappeared_counts[obj_id] = 0
                self.tracked_faces[obj_id] = {
                    'first_seen': current_time,
                    'total_duration': 0.0,
                    'status': 'Unconfirmed'
                }
                self.next_object_id += 1

        return self.tracked_faces

    def execute_pipeline(self):
        cap = cv2.VideoCapture(self.camera_index)
        print("[INFO] Starting Full Multimodal Pipeline. Press 'q' to exit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab camera frame.")
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)

            rects = []
            if results.detections:
                for det in results.detections:
                    box = det.location_data.relative_bounding_box
                    sx, sy = int(box.xmin * w), int(box.ymin * h)
                    ex, ey = int(sx + box.width * w), int(sy + box.height * h)
                    rects.append((max(0, sx), max(0, sy), min(w, ex), min(h, ey)))
                    cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)

            current_time = time.time()
            tracked = self.update_vision_tracker(rects, current_time)

            # Compute real-time metrics summary for dashboard display
            total_faces = len(tracked)
            present_count = sum(1 for f in tracked.values() if f.get('status') == 'Present')
            attention_ratio = (present_count / total_faces) if total_faces > 0 else 0.0
            
            # Simulated audio metrics for live loop integration
            simulated_speech_ratio = 40.0
            simulated_noise_ratio = 10.0
            overall_engagement = (attention_ratio * 100 * 0.6) + (simulated_speech_ratio * 0.4)

            # Render Overlays on Video Stream
            cv2.putText(frame, f"Detected: {total_faces} | Present: {present_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Engagement Score: {overall_engagement:.1f}%", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            for obj_id, info in tracked.items():
                if obj_id in self.centroids:
                    cx, cy = self.centroids[obj_id]
                    status = info['status']
                    dur = int(info['total_duration'])
                    color = (0, 255, 0) if status == 'Present' else (0, 165, 255)
                    cv2.putText(frame, f"ID {obj_id}: {status} ({dur}s)", (cx - 40, cy - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.circle(frame, (cx, cy), 4, color, -1)

            cv2.imshow("Activity 6.1 - Core Application Integration Pipeline", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Pipeline execution terminated successfully.")

if __name__ == "__main__":
    runner = MultimodalPipelineRunner()
    runner.execute_pipeline()