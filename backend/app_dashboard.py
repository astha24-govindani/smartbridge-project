import threading
from datetime import datetime

import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


st.set_page_config(
    page_title="Multimodal Attendance Agent | Attendance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e6ebf2;
            padding: 1rem 1.15rem;
            border-radius: 14px;
            box-shadow: 0 3px 12px rgba(25, 45, 75, 0.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self, confidence_threshold: float) -> None:
        self.detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=confidence_threshold
        )
        self.lock = threading.Lock()
        self.frame_count = 0
        self.current_faces = 0
        self.total_faces = 0
        self.max_faces = 0
        self.confidence_total = 0.0
        self.detection_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(image_rgb)
        confidences = []

        if results.detections:
            image_height, image_width = image.shape[:2]
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                x1 = max(0, int(box.xmin * image_width))
                y1 = max(0, int(box.ymin * image_height))
                x2 = min(image_width, int((box.xmin + box.width) * image_width))
                y2 = min(image_height, int((box.ymin + box.height) * image_height))
                confidence = float(detection.score[0])
                confidences.append(confidence)

                cv2.rectangle(image, (x1, y1), (x2, y2), (36, 180, 120), 2)
                cv2.putText(
                    image,
                    f"Face {confidence:.0%}",
                    (x1, max(y1 - 10, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (36, 180, 120),
                    2,
                    cv2.LINE_AA,
                )

        face_count = len(confidences)
        with self.lock:
            self.frame_count += 1
            self.current_faces = face_count
            self.total_faces += face_count
            self.max_faces = max(self.max_faces, face_count)
            self.confidence_total += sum(confidences)
            self.detection_count += face_count

        cv2.rectangle(image, (0, 0), (220, 42), (20, 35, 55), -1)
        cv2.putText(
            image,
            f"Faces detected: {face_count}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def get_metrics(self) -> dict[str, int | float]:
        with self.lock:
            average_faces = (
                self.total_faces / self.frame_count if self.frame_count else 0.0
            )
            average_confidence = (
                self.confidence_total / self.detection_count
                if self.detection_count
                else 0.0
            )
            return {
                "frame_count": self.frame_count,
                "current_faces": self.current_faces,
                "average_faces": average_faces,
                "max_faces": self.max_faces,
                "average_confidence": average_confidence,
            }


if "camera_history" not in st.session_state:
    st.session_state.camera_history = []
if "last_logged_frame" not in st.session_state:
    st.session_state.last_logged_frame = 0
if "last_processor_id" not in st.session_state:
    st.session_state.last_processor_id = None

st.sidebar.title("🎓 Multimodal Attendance Agent")
st.sidebar.caption("Multimodal attendance & engagement")
page = st.sidebar.radio("Dashboard", ["Live Camera", "Session Analytics"])

st.title("Attendance & Engagement Dashboard")
st.caption(
    "Live face detection for classroom monitoring. Faces are detected, not "
    "identified by name."
)
st.divider()

if page == "Live Camera":
    confidence_threshold = st.sidebar.slider(
        "Face detection confidence",
        min_value=0.3,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Higher values require a more confident face detection.",
    )
    st.subheader("Live camera")
    st.write(
        "Press **START** on the video control to begin. Your browser will ask for "
        "camera permission if it has not been granted yet."
    )

    rtc_context = webrtc_streamer(
        key="smartbridge-live-camera",
        video_processor_factory=lambda: FaceDetectionProcessor(confidence_threshold),
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if rtc_context.state.playing:
        st.success("Camera connected. Face boxes and the current face count appear in the video.")
    else:
        st.info(
            "If the camera does not start, use Chrome or Edge, allow camera access "
            "for this site, and reload the page. For access from another device, "
            "serve the app over HTTPS."
        )

    @st.fragment(run_every=1)
    def show_live_metrics() -> None:
        processor = rtc_context.video_processor
        if processor is None:
            st.caption("Live metrics will appear after the camera connects.")
            return

        if id(processor) != st.session_state.last_processor_id:
            st.session_state.last_processor_id = id(processor)
            st.session_state.last_logged_frame = 0

        metrics = processor.get_metrics()
        metric_columns = st.columns(4)
        metric_columns[0].metric("Faces in current frame", metrics["current_faces"])
        metric_columns[1].metric(
            "Average faces per frame", f"{metrics['average_faces']:.1f}"
        )
        metric_columns[2].metric("Most faces in a frame", metrics["max_faces"])
        metric_columns[3].metric(
            "Average detection confidence",
            f"{metrics['average_confidence']:.1%}"
            if metrics["average_confidence"]
            else "—",
        )

        if metrics["frame_count"] > st.session_state.last_logged_frame:
            st.session_state.camera_history.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "faces": metrics["current_faces"],
                    "average_faces": round(metrics["average_faces"], 2),
                    "max_faces": metrics["max_faces"],
                }
            )
            st.session_state.camera_history = st.session_state.camera_history[-300:]
            st.session_state.last_logged_frame = metrics["frame_count"]

    show_live_metrics()

elif page == "Session Analytics":
    st.subheader("Session analytics")
    history = st.session_state.camera_history

    if not history:
        st.info(
            "Start the live camera to collect face-detection session metrics."
        )
    else:
        metric_columns = st.columns(3)
        metric_columns[0].metric("Logged camera samples", len(history))
        metric_columns[1].metric(
            "Highest faces in one frame",
            max(sample["max_faces"] for sample in history),
        )
        metric_columns[2].metric(
            "Average faces per sample",
            f"{sum(sample['faces'] for sample in history) / len(history):.1f}",
        )

        chart_column, table_column = st.columns([1.3, 1], gap="large")
        with chart_column:
            st.markdown("#### Live face count")
            st.line_chart(
                {
                    "Faces detected": [
                        sample["faces"] for sample in history
                    ]
                }
            )
        with table_column:
            st.markdown("#### Recent samples")
            st.dataframe(
                list(reversed(history[-30:])),
                hide_index=True,
                use_container_width=True,
            )

        report = "time,faces,average_faces,max_faces\n" + "\n".join(
            f"{sample['time']},{sample['faces']},"
            f"{sample['average_faces']},{sample['max_faces']}"
            for sample in history
        )
        st.download_button(
            "Download session report",
            data=report,
            file_name="smartbridge_session_analytics.csv",
            mime="text/csv",
        )
