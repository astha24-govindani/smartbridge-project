import cv2
import mediapipe as mp
import numpy as np
import time

class FaceDetectorTracker:
    def __init__(self, min_detection_confidence=0.5):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        
        # Simple centroid tracking state variables for unique IDs across frames
        self.next_object_id = 0
        self.tracked_faces = {}  # {object_id: {'centroid': (x, y), 'disappeared': 0}}
        self.max_disappeared = 15  # Frames allowed before losing ID

    def register(self, centroid):
        self.tracked_faces[self.next_object_id] = {
            'centroid': centroid,
            'disappeared': 0
        }
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.tracked_faces[object_id]

    def update(self, rects):
        """
        Updates tracked objects based on current frame bounding boxes (rects).
        rects format: [(startX, startY, endX, endY), ...]
        """
        if len(rects) == 0:
            # Increment disappeared counter for all current tracks
            for object_id in list(self.tracked_faces.keys()):
                self.tracked_faces[object_id]['disappeared'] += 1
                if self.tracked_faces[object_id]['disappeared'] > self.max_disappeared:
                    self.deregister(object_id)
            return self.tracked_faces

        # Calculate centroids for incoming detections
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        # If we are not currently tracking any faces, register all input centroids
        if len(self.tracked_faces) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Match existing tracked objects to new input centroids based on distance
            object_ids = list(self.tracked_faces.keys())
            object_centroids = [self.tracked_faces[object_id]['centroid'] for object_id in object_ids]

            # Compute Euclidean distance between existing and new centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[0:len(rows)]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.tracked_faces[object_id]['centroid'] = input_centroids[col]
                self.tracked_faces[object_id]['disappeared'] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Handle disappeared objects
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.tracked_faces[object_id]['disappeared'] += 1
                    if self.tracked_faces[object_id]['disappeared'] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.tracked_faces

def run_face_tracking(source=0):
    cap = cv2.VideoCapture(source)
    tracker = FaceDetectorTracker()
    
    print("[INFO] Starting Face Detection and Tracking. Press 'q' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame or video ended.")
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
                
                # Clip coordinates to frame boundaries
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                
                rects.append((startX, startY, endX, endY))
                
                # Draw raw detection box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Update tracking IDs
        tracked_objects = tracker.update(rects)

        # Render unique IDs and centroids on the frame
        for object_id, info in tracked_objects.items():
            cX, cY = info['centroid']
            cv2.putText(frame, f"ID: {object_id}", (cX - 10, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

        cv2.imshow("Activity 2.1 - Face Detection & Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Pass 0 for default webcam, or provide path to a video file
    run_face_tracking(0)