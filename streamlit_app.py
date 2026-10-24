import streamlit as st
import threading

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from touchguard.pipeline import TouchGuardPipeline
from touchguard.config import load_settings


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="TouchGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🛡️ TouchGuard AI")

st.subheader(
    "Real-Time Face-Touch Detection & Voice Warning System"
)

st.write(
    "Place your hand near your face. "
    "TouchGuard AI detects face-touch activity "
    "and counts each confirmed touch."
)


# =========================================================
# SHARED TOUCHGUARD STATE
# =========================================================

class TouchGuardState:

    def __init__(self):

        self.lock = threading.Lock()

        self.touch_count = 0
        self.warning_count = 0

        self.touching = False

        self.faces = 0
        self.hands = 0

        self.distance = 0.0
        self.fps = 0.0

    def update(self, analysis):

        with self.lock:

            self.touch_count = analysis.touch_count
            self.warning_count = analysis.warning_count

            self.touching = analysis.touching

            self.faces = len(analysis.faces)
            self.hands = len(analysis.hands)

            self.distance = analysis.distance
            self.fps = analysis.fps

    def get(self):

        with self.lock:

            return {
                "touch_count": self.touch_count,
                "warning_count": self.warning_count,
                "touching": self.touching,
                "faces": self.faces,
                "hands": self.hands,
                "distance": self.distance,
                "fps": self.fps,
            }


# =========================================================
# CREATE SESSION STATE
# =========================================================

if "touchguard_state" not in st.session_state:

    st.session_state.touchguard_state = TouchGuardState()


state = st.session_state.touchguard_state


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class TouchGuardProcessor(VideoProcessorBase):

    def __init__(self):

        self.settings = load_settings()

        self.pipeline = TouchGuardPipeline(
            self.settings
        )

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV image
        img = frame.to_ndarray(
            format="bgr24"
        )

        try:

            # Run TouchGuard processing
            analysis = self.pipeline.process(img)

            # Send results to dashboard
            state.update(analysis)

            # Get processed frame
            output = analysis.frame

            # Return processed frame
            return frame.from_ndarray(
                output,
                format="bgr24"
            )

        except Exception as e:

            # Keep camera alive even if one frame fails
            return frame.from_ndarray(
                img,
                format="bgr24"
            )

    def __del__(self):

        try:

            self.pipeline.close()

        except Exception:

            pass


# =========================================================
# LIVE CAMERA
# =========================================================

st.markdown("### 📷 Live Camera")

st.info(
    "Click START and allow camera permission "
    "when your browser asks."
)


ctx = webrtc_streamer(

    key="touchguard-camera",

    video_processor_factory=TouchGuardProcessor,

    media_stream_constraints={
        "video": {
            "width": {
                "ideal": 640
            },
            "height": {
                "ideal": 480
            },
            "frameRate": {
                "ideal": 30
            },
        },
        "audio": False,
    },

    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    },

    async_processing=True,
)


# =========================================================
# LIVE DASHBOARD
# =========================================================

st.markdown("---")


@st.fragment(run_every="1s")
def live_dashboard():

    st.markdown("### 📊 Live TouchGuard Dashboard")

    data = state.get()

    # -----------------------------------------------------
    # FIRST ROW
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Face Touches",
            data["touch_count"]
        )

    with col2:

        st.metric(
            "Warnings",
            data["warning_count"]
        )

    with col3:

        st.metric(
            "Faces Detected",
            data["faces"]
        )

    with col4:

        st.metric(
            "Hands Detected",
            data["hands"]
        )

    # -----------------------------------------------------
    # TOUCH STATUS
    # -----------------------------------------------------

    if data["touching"]:

        st.error(
            "⚠️ FACE TOUCH DETECTED"
        )

    else:

        st.success(
            "✅ No Face Touch Detected"
        )

    # -----------------------------------------------------
    # SECOND ROW
    # -----------------------------------------------------

    col5, col6 = st.columns(2)

    with col5:

        st.metric(
            "Hand-Face Distance",
            f"{data['distance']:.1f} px"
        )

    with col6:

        st.metric(
            "Processing FPS",
            f"{data['fps']:.1f}"
        )


# Start dashboard
live_dashboard()


# =========================================================
# INFORMATION
# =========================================================

st.markdown("---")

st.info(
    "💡 Tip: Keep your face and hand clearly visible "
    "to the camera. Each confirmed touch is counted "
    "by the system."
)


# =========================================================
# PROJECT FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🛡️ TouchGuard AI — Real-Time Face-Touch Detection"
)