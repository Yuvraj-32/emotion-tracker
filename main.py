"""
==========================================================
  ULTIMATE REAL-TIME EMOTION & ATTENTION AI  v2.0
==========================================================
  Architecture:
    - YOLO v8  → Person detection + persistent ID tracking
    - MediaPipe FaceLandmarker → 468-point face mesh + gaze + EAR
    - DeepFace  → Emotion classification on precise face crops
    - DNN SSD   → Fallback face detector when MediaPipe misses
    - Multi-threaded pipeline for maximum FPS

  Features (v2 NEW):
    ★ Drowsiness detection via Eye Aspect Ratio
    ★ Real-time Engagement Score gauge (0-100)
    ★ Scrolling emotion timeline graph
    ★ CSV session logging (timestamped)
    ★ Auto-screenshot on distraction / drowsiness
    ★ Session summary report on exit

  Existing:
    - Multi-person tracking with unique IDs
    - Temporal emotion smoothing (majority vote + EMA boxes)
    - Head pose / gaze estimation (pitch & yaw)
    - Distraction detection with visual + sound + email alerts
    - Polished HUD with emotion breakdown bars

  Controls:
    q  → Quit
    d  → Toggle emotion debug bars
    f  → Toggle FPS display
    g  → Toggle timeline graph
    Click the ALARM button → Toggle sound alarm
==========================================================
"""

import cv2
import time
import threading
import queue
import os
import csv
import math
import signal
import argparse
import numpy as np
import traceback
import smtplib
from datetime import datetime
from email.message import EmailMessage
from collections import deque, Counter
from deepface import DeepFace

# Suppress TensorFlow info spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ══════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# YOLO settings
YOLO_MODEL = 'yolov8n.pt'
YOLO_SKIP_FRAMES = 3
YOLO_IMG_SIZE = 320
YOLO_CONFIDENCE = 0.5

# Face analysis
FACE_PADDING_W = 0.20
FACE_PADDING_H_TOP = 0.30
FACE_PADDING_H_BOT = 0.15
DNN_CONFIDENCE = 0.6

# Smoothing
EMA_ALPHA = 0.6
EMOTION_HISTORY_LEN = 12

# Gaze / attention
GAZE_PITCH_THRESHOLD = 20
GAZE_YAW_THRESHOLD = 20
DISTRACTION_DURATION_SECONDS = 20  # Must look away this long before flagging distraction

# Drowsiness
EAR_THRESHOLD = 0.22          # Eye Aspect Ratio below this = eyes closing
DROWSY_DURATION_SECONDS = 3   # Eyes must be closed this long before flagging drowsy
EAR_SMOOTHING_WINDOW = 5      # Rolling average window for EAR noise filtering

# Alerts
SOUND_ENABLED_DEFAULT = False
DISTRACTION_ALARM_COOLDOWN = 2.0
SCREENSHOT_COOLDOWN = 10.0    # Seconds between auto-screenshots
LOG_INTERVAL_SECONDS = 5      # Only log to CSV every N seconds (reduces file size)
ABSENCE_FRAME_THRESHOLD = 30  # Frames without detection before marking person as absent (~1 sec)

# Email
SENDER_EMAIL = "yuvrajmakhecha4@gmail.com"
APP_PASSWORD = ""
RECEIVER_EMAIL = "yuvrajmakhecha4@gmail.com"
EMAIL_COOLDOWN_SECONDS = 60

# Display
SHOW_FPS_DEFAULT = True
SHOW_DEBUG_DEFAULT = False
SHOW_GRAPH_DEFAULT = True
DISPLAY_TOP_N_EMOTIONS = 4
GRAPH_HISTORY_SECONDS = 60    # Timeline shows last N seconds

# ══════════════════════════════════════════════
#  EMOTION STYLING
# ══════════════════════════════════════════════
EMOTION_MAP = {
    'happy': 'Happy', 'sad': 'Tired / Sad', 'angry': 'Frustrated',
    'disgust': 'Frustrated', 'surprise': 'Surprised',
    'fear': 'Neutral', 'neutral': 'Neutral', 'Scanning...': 'Scanning...',
}

EMOTION_COLORS = {
    'Happy': (0, 230, 118), 'Tired / Sad': (0, 165, 255),
    'Frustrated': (0, 0, 255), 'Surprised': (0, 255, 255),
    'Neutral': (255, 180, 0), 'Scanning...': (200, 200, 200),
}

EMOTION_POSITIVE = {'Happy', 'Surprised'}
EMOTION_NEGATIVE = {'Tired / Sad', 'Frustrated'}

# Engagement score weights for each friendly emotion
ENGAGEMENT_EMOTION_SCORE = {
    'Happy': 90, 'Surprised': 75, 'Neutral': 50,
    'Tired / Sad': 25, 'Frustrated': 15, 'Scanning...': 50,
}

# MediaPipe landmark indices for Eye Aspect Ratio
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


# ══════════════════════════════════════════════
#  SESSION LOGGER (CSV + screenshots)
# ══════════════════════════════════════════════
class SessionLogger:
    def __init__(self):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(base, exist_ok=True)
        self.screenshot_dir = os.path.join(base, "screenshots")
        os.makedirs(self.screenshot_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_path = os.path.join(base, f"session_{ts}.csv")
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            'timestamp', 'person_id', 'emotion', 'is_looking', 'is_drowsy',
            'engagement', 'focus_duration_sec', 'distraction_count',
            'drowsy_count', 'emotion_stability', 'session_elapsed_min',
            'productivity'
        ])
        self.last_screenshot_time = 0
        self.start_time = time.time()

        # Counters for session summary
        self.total_entries = 0
        self.emotion_counts = Counter()
        self.distraction_count = 0
        self.drowsy_count = 0
        self.total_focus_duration = 0.0
        self.total_emotion_stability = 0.0
        self.last_log_time = 0
        print(f"[LOG] Session logging to: {self.csv_path}")

    def log(self, tid, emotion, looking, drowsy, engagement,
             focus_dur, distraction_cnt, drowsy_cnt, emotion_stab, session_elapsed,
             productivity='N/A'):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.writer.writerow([
            ts, tid, emotion, looking, drowsy,
            f"{engagement:.0f}", f"{focus_dur:.1f}",
            distraction_cnt, drowsy_cnt,
            f"{emotion_stab:.2f}", f"{session_elapsed:.1f}",
            productivity
        ])
        self.csv_file.flush()  # Flush immediately so dashboard can read live data
        self.total_entries += 1
        self.emotion_counts[emotion] += 1
        self.total_focus_duration += focus_dur
        self.total_emotion_stability += emotion_stab
        if not looking:
            self.distraction_count += 1
        if drowsy:
            self.drowsy_count += 1

    def screenshot(self, frame, reason="alert"):
        now = time.time()
        if now - self.last_screenshot_time > SCREENSHOT_COOLDOWN:
            self.last_screenshot_time = now
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.screenshot_dir, f"{reason}_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"[LOG] Screenshot saved: {path}")

    def get_summary_text(self):
        """Generate a formatted session summary string."""
        duration = time.time() - self.start_time
        mins = int(duration // 60)
        secs = int(duration % 60)

        lines = []
        lines.append("═" * 45)
        lines.append("   SESSION SUMMARY REPORT")
        lines.append("═" * 45)
        lines.append(f"  Date:              {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"  Duration:          {mins}m {secs}s")
        lines.append(f"  Data points:       {self.total_entries}")
        lines.append(f"  Distractions:      {self.distraction_count}")
        lines.append(f"  Drowsy events:     {self.drowsy_count}")

        if self.total_entries > 0:
            focus_pct = ((self.total_entries - self.distraction_count - self.drowsy_count)
                         / self.total_entries * 100)
            avg_focus = self.total_focus_duration / self.total_entries
            avg_stab = self.total_emotion_stability / self.total_entries
            lines.append(f"  Focus rate:        {focus_pct:.1f}%")
            lines.append(f"  Avg focus streak:  {avg_focus:.1f}s")
            lines.append(f"  Emotion stability: {avg_stab:.0%}")

        if self.emotion_counts:
            top = self.emotion_counts.most_common(5)
            lines.append("")
            lines.append("  Emotion Breakdown:")
            for emo, count in top:
                pct = count / self.total_entries * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                lines.append(f"    {emo:<15} {bar} {pct:.1f}%")

        lines.append("")
        lines.append(f"  CSV log:           {self.csv_path}")
        lines.append(f"  Screenshots:       {self.screenshot_dir}")
        lines.append("═" * 45)
        return "\n".join(lines)

    def print_summary(self):
        print("\n" + self.get_summary_text())

    def close(self):
        self.csv_file.close()


# ══════════════════════════════════════════════
#  EYE ASPECT RATIO (drowsiness)
# ══════════════════════════════════════════════
def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Where p1..p6 are the 6 eye landmarks. A low EAR means the eye is closing.
    """
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
    # Vertical distances
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    # Horizontal distance
    h = math.dist(pts[0], pts[3])
    if h == 0:
        return 0.3
    return (v1 + v2) / (2.0 * h)


# ══════════════════════════════════════════════
#  FACE PREPROCESSING (low-quality video boost)
# ══════════════════════════════════════════════
def preprocess_face_crop(crop):
    """
    Enhance a face crop for better emotion classification,
    especially on low-resolution or poorly-lit webcams.

    Pipeline:
      1. Upscale small crops to 224x224 (DeepFace input size) with LANCZOS4
      2. CLAHE contrast enhancement on luminance channel
      3. Bilateral filter to denoise while preserving edges
    """
    h, w = crop.shape[:2]

    # 1. Upscale small faces to DeepFace's expected input size
    if h < 224 or w < 224:
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    #    Boost local contrast — huge help in dark/uneven lighting
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    crop = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 3. Bilateral filter — smooths noise but keeps edges sharp
    crop = cv2.bilateralFilter(crop, d=5, sigmaColor=50, sigmaSpace=50)

    return crop


# ══════════════════════════════════════════════
#  ENGAGEMENT SCORE
# ══════════════════════════════════════════════
class EngagementTracker:
    """
    Composite score (0-100) from:
      - 40% gaze (looking at screen)
      - 30% emotion positivity
      - 30% alertness (not drowsy)
    Smoothed over time with EMA.
    """
    def __init__(self):
        self.score = 50.0
        self.history = deque(maxlen=GRAPH_HISTORY_SECONDS * 5)  # ~5 updates/sec

    def update(self, is_looking, emotion_label, is_drowsy):
        gaze_score = 100 if is_looking else 0
        emo_score = ENGAGEMENT_EMOTION_SCORE.get(emotion_label, 50)
        alert_score = 0 if is_drowsy else 100

        raw = gaze_score * 0.40 + emo_score * 0.30 + alert_score * 0.30
        self.score = self.score * 0.7 + raw * 0.3  # EMA
        self.history.append((time.time(), self.score))
        return self.score

    def get_score(self):
        return self.score


# ══════════════════════════════════════════════
#  TIMELINE GRAPH
# ══════════════════════════════════════════════
class TimelineGraph:
    """Stores engagement history points and draws a scrolling graph."""
    def __init__(self, max_seconds=GRAPH_HISTORY_SECONDS):
        self.max_seconds = max_seconds
        self.points = deque(maxlen=max_seconds * 10)

    def add_point(self, score, is_drowsy=False, is_distracted=False):
        self.points.append((time.time(), score, is_drowsy, is_distracted))

    def draw(self, frame):
        h, w = frame.shape[:2]
        graph_h = 80
        graph_y = h - graph_h - 10
        graph_x = 15
        graph_w = w - 30

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (graph_x - 5, graph_y - 20),
                       (graph_x + graph_w + 5, graph_y + graph_h + 5),
                       (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "ENGAGEMENT TIMELINE", (graph_x, graph_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        # Grid lines at 25%, 50%, 75%
        for pct in [25, 50, 75]:
            gy = graph_y + graph_h - int(graph_h * pct / 100)
            cv2.line(frame, (graph_x, gy), (graph_x + graph_w, gy), (50, 50, 50), 1)
            cv2.putText(frame, str(pct), (graph_x + graph_w + 2, gy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100, 100, 100), 1)

        if len(self.points) < 2:
            return

        now = time.time()
        min_t = now - self.max_seconds

        # Map points to pixel coords
        px_points = []
        for t, score, drowsy, distracted in self.points:
            if t < min_t:
                continue
            px_x = graph_x + int((t - min_t) / self.max_seconds * graph_w)
            px_y = graph_y + graph_h - int(graph_h * max(0, min(100, score)) / 100)
            px_points.append((px_x, px_y, score, drowsy, distracted))

        if len(px_points) < 2:
            return

        # Draw the line with gradient coloring
        for i in range(1, len(px_points)):
            x1, y1, s1, d1, dis1 = px_points[i - 1]
            x2, y2, s2, d2, dis2 = px_points[i]

            if d2 or dis2:
                color = (0, 0, 255)  # Red for alerts
            elif s2 > 65:
                color = (0, 230, 118)  # Green for good
            elif s2 > 35:
                color = (0, 200, 255)  # Yellow for medium
            else:
                color = (0, 100, 255)  # Orange for low
            cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════
def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
    if r < 2:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_corner_accents(img, x1, y1, x2, y2, color, length=15, thickness=3):
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def draw_emotion_bars(img, x, y, scores, top_n=DISPLAY_TOP_N_EMOTIONS):
    if not scores:
        return
    sorted_emo = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    bar_w, bar_h = 100, 12
    for i, (emo, score) in enumerate(sorted_emo):
        by = y + i * 20
        cv2.putText(img, emo[:3].upper(), (x, by + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
        bx = x + 35
        cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
        fill = int(bar_w * score / 100.0)
        bar_color = EMOTION_COLORS.get(EMOTION_MAP.get(emo, emo), (200, 200, 200))
        cv2.rectangle(img, (bx, by), (bx + fill, by + bar_h), bar_color, -1)
        cv2.putText(img, f"{score:.0f}%", (bx + bar_w + 4, by + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA)


def draw_engagement_gauge(img, score, x, y, radius=35):
    """Draw a semi-circular gauge showing the engagement score."""
    # Background arc
    cv2.ellipse(img, (x, y), (radius, radius), 0, 180, 360, (50, 50, 50), 4)

    # Score arc (colored by value)
    if score > 65:
        color = (0, 230, 118)
    elif score > 35:
        color = (0, 200, 255)
    else:
        color = (0, 0, 255)

    angle = 180 + int(180 * score / 100)
    cv2.ellipse(img, (x, y), (radius, radius), 0, 180, angle, color, 4, cv2.LINE_AA)

    # Score text
    cv2.putText(img, f"{int(score)}", (x - 12, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "ENGAGE", (x - 24, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)


def draw_face_hud(img, x1, y1, x2, y2, tid, label, is_distracted, is_drowsy,
                  pitch, yaw, ear_l, ear_r, raw_scores, show_debug):
    color = EMOTION_COLORS.get(label, (200, 200, 200))

    if is_distracted:
        color = (0, 0, 255)
    if is_drowsy:
        color = (0, 100, 255)  # Deep orange for drowsy

    draw_rounded_rect(img, (x1, y1), (x2, y2), color, 2, radius=12)
    draw_corner_accents(img, x1, y1, x2, y2, color)

    # Top label
    info = f"ID:{tid} | {label.upper()}"
    (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_y = max(y1 - 8, th + 10)
    cv2.rectangle(img, (x1, label_y - th - 8), (x1 + tw + 12, label_y + 4), color, -1)
    cv2.putText(img, info, (x1 + 6, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # Gaze angles + EAR
    cv2.putText(img, f"P:{pitch} Y:{yaw} EAR:{(ear_l+ear_r)/2:.2f}", (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Alerts
    if is_drowsy:
        cv2.putText(img, "DROWSY!", (x1, y2 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2, cv2.LINE_AA)
    elif is_distracted:
        cv2.putText(img, "LOOKING AWAY!", (x1, y2 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # Debug bars
    if show_debug and raw_scores:
        bar_x = x2 + 10
        if bar_x + 160 < img.shape[1]:
            draw_emotion_bars(img, bar_x, y1, raw_scores)


# ══════════════════════════════════════════════
#  SOUND ALARM
# ══════════════════════════════════════════════
class SoundAlert:
    def __init__(self):
        self.is_playing = False

    def play(self):
        if not self.is_playing and state['sound_enabled']:
            self.is_playing = True
            threading.Thread(target=self._beep, daemon=True).start()

    def _beep(self):
        try:
            import winsound
            winsound.Beep(1000, 500)
        except ImportError:
            print('\a', end='', flush=True)
        time.sleep(DISTRACTION_ALARM_COOLDOWN)
        self.is_playing = False


# ══════════════════════════════════════════════
#  EMAIL ALARM
# ══════════════════════════════════════════════
class EmailAlert:
    def __init__(self):
        self.last_sent = 0

    def send_alert(self, reason="distraction"):
        if SENDER_EMAIL == "YOUR_EMAIL@gmail.com" or not APP_PASSWORD:
            return
        now = time.time()
        if now - self.last_sent > EMAIL_COOLDOWN_SECONDS:
            self.last_sent = now
            threading.Thread(target=self._dispatch, args=(reason,), daemon=True).start()

    def send_summary(self, summary_text, csv_path=None):
        """Send the full session summary via email when the session ends."""
        if SENDER_EMAIL == "YOUR_EMAIL@gmail.com" or not APP_PASSWORD:
            return
        print(">> Sending session summary email...")
        # Run synchronously so the app doesn't exit before the email sends
        try:
            msg = EmailMessage()
            msg['Subject'] = f'📊 Emotion Tracker — Session Report ({datetime.now().strftime("%b %d, %H:%M")})'
            msg['From'] = SENDER_EMAIL
            msg['To'] = RECEIVER_EMAIL
            msg.set_content(summary_text)

            # Attach the CSV log if it exists
            if csv_path and os.path.exists(csv_path):
                with open(csv_path, 'rb') as f:
                    csv_data = f.read()
                msg.add_attachment(
                    csv_data,
                    maintype='text', subtype='csv',
                    filename=os.path.basename(csv_path)
                )

            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(">> ✅ Session summary email sent!")
        except Exception as e:
            print(f">> ❌ Failed to send summary email: {e}")

    def _dispatch(self, reason):
        try:
            msg = EmailMessage()
            msg.set_content(f"Alert: {reason} detected at the workstation.")
            msg['Subject'] = f'⚠️ Workstation Alert: {reason.title()}'
            msg['From'] = SENDER_EMAIL
            msg['To'] = RECEIVER_EMAIL
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
            server.quit()
        except Exception:
            pass


# ══════════════════════════════════════════════
#  DNN FACE DETECTOR (fallback)
# ══════════════════════════════════════════════
def load_dnn_face_detector():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    proto = os.path.join(model_dir, "deploy.prototxt")
    caffe = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    if not os.path.exists(proto) or not os.path.exists(caffe):
        import urllib.request
        print("[INFO] Downloading DNN face detector model (~10 MB)...")
        if not os.path.exists(proto):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", proto)
        if not os.path.exists(caffe):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", caffe)
        print("[INFO] DNN model downloaded.")
    return cv2.dnn.readNetFromCaffe(proto, caffe)


def detect_face_dnn(net, roi):
    h, w = roi.shape[:2]
    if h < 30 or w < 30:
        return None
    blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    dets = net.forward()
    best, best_conf = None, DNN_CONFIDENCE
    for i in range(dets.shape[2]):
        conf = dets[0, 0, i, 2]
        if conf > best_conf:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            best = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
            best_conf = conf
    return best


# ══════════════════════════════════════════════
#  CAMERA THREAD
# ══════════════════════════════════════════════
stop_event = threading.Event()

def camera_worker(frame_q):
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam!")
        stop_event.set()
        return
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        if not frame_q.empty():
            try: frame_q.get_nowait()
            except queue.Empty: pass
        frame_q.put(frame)
    cap.release()


# ══════════════════════════════════════════════
#  ANALYSIS THREAD
#  MediaPipe mesh + gaze + EAR + DeepFace emotion
# ══════════════════════════════════════════════
def analysis_worker(input_q, result_q):
    print("[AI] Loading models...")

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        import urllib.request
        print("[AI] Downloading MediaPipe FaceLandmarker...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            model_path)

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    fl_options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_faces=3,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5,
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(fl_options)

    dnn_net = load_dnn_face_detector()

    try:
        DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8),
                         actions=['emotion'], enforce_detection=False, silent=True)
    except Exception:
        pass
    print("[AI] All models ready!")

    while not stop_event.is_set():
        try:
            tid, full_img, yolo_coords = input_q.get(timeout=0.1)
        except queue.Empty:
            continue

        yx1, yy1, yx2, yy2 = yolo_coords
        emotion = "Scanning..."
        is_looking = True
        pitch, yaw = 0, 0
        ear_left, ear_right = None, None  # None = no landmarks found (preserves previous EAR)
        raw_scores = {}
        h, w = full_img.shape[:2]

        try:
            img_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            detection = face_landmarker.detect(mp_image)

            face_found = False

            if detection.face_landmarks and detection.facial_transformation_matrixes:
                for idx, landmarks in enumerate(detection.face_landmarks):
                    nose_x = int(landmarks[1].x * w)
                    nose_y = int(landmarks[1].y * h)

                    if yx1 <= nose_x <= yx2 and yy1 <= nose_y <= yy2:
                        face_found = True

                        # Gaze
                        matrix = detection.facial_transformation_matrixes[idx]
                        rmat = matrix[:3, :3]
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                        pitch = int(angles[0])
                        yaw = int(angles[1])
                        if abs(pitch) > GAZE_PITCH_THRESHOLD or abs(yaw) > GAZE_YAW_THRESHOLD:
                            is_looking = False

                        # EAR (drowsiness)
                        ear_left = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
                        ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)

                        # Precise face crop
                        x_coords = [lm.x * w for lm in landmarks]
                        y_coords = [lm.y * h for lm in landmarks]
                        fx1, fy1 = int(min(x_coords)), int(min(y_coords))
                        fx2, fy2 = int(max(x_coords)), int(max(y_coords))
                        mw = int((fx2 - fx1) * FACE_PADDING_W)
                        mh_top = int((fy2 - fy1) * FACE_PADDING_H_TOP)
                        mh_bot = int((fy2 - fy1) * FACE_PADDING_H_BOT)
                        crop = full_img[
                            max(0, fy1 - mh_top):min(h, fy2 + mh_bot),
                            max(0, fx1 - mw):min(w, fx2 + mw)
                        ]

                        if crop.size > 0 and crop.shape[0] > 48 and crop.shape[1] > 48:
                            crop = preprocess_face_crop(crop)
                            obj = DeepFace.analyze(
                                crop, actions=['emotion'],
                                enforce_detection=False,
                                detector_backend='skip', silent=True)
                            res = obj[0] if isinstance(obj, list) else obj
                            raw_scores = res.get('emotion', {})
                            dom = res.get('dominant_emotion', 'neutral')
                            if raw_scores.get('angry', 0) > 15:
                                emotion = 'angry'
                            elif raw_scores.get('disgust', 0) > 20:
                                emotion = 'disgust'
                            elif dom == 'fear':
                                emotion = 'neutral'
                            else:
                                emotion = dom
                        break

            # Fallback: DNN
            if not face_found:
                person_roi = full_img[max(0, yy1):min(h, yy2), max(0, yx1):min(w, yx2)]
                dnn_face = detect_face_dnn(dnn_net, person_roi)
                if dnn_face:
                    dx1, dy1, dx2, dy2 = dnn_face
                    dw, dh = dx2 - dx1, dy2 - dy1
                    pw, ph = int(dw * FACE_PADDING_W), int(dh * FACE_PADDING_H_TOP)
                    crop = person_roi[
                        max(0, dy1 - ph):min(person_roi.shape[0], dy2 + int(dh * FACE_PADDING_H_BOT)),
                        max(0, dx1 - pw):min(person_roi.shape[1], dx2 + pw)
                    ]
                    if crop.size > 0 and crop.shape[0] > 48 and crop.shape[1] > 48:
                        crop = preprocess_face_crop(crop)
                        obj = DeepFace.analyze(
                            crop, actions=['emotion'],
                            enforce_detection=False,
                            detector_backend='skip', silent=True)
                        res = obj[0] if isinstance(obj, list) else obj
                        raw_scores = res.get('emotion', {})
                        dom = res.get('dominant_emotion', 'neutral')
                        if raw_scores.get('angry', 0) > 15:
                            emotion = 'angry'
                        elif raw_scores.get('disgust', 0) > 20:
                            emotion = 'disgust'
                        elif dom == 'fear':
                            emotion = 'neutral'
                        else:
                            emotion = dom
        except Exception:
            pass

        result_q.put((tid, emotion, is_looking, pitch, yaw, ear_left, ear_right, raw_scores))


# ══════════════════════════════════════════════
#  MOUSE CALLBACK
# ══════════════════════════════════════════════
state = {
    'sound_enabled': SOUND_ENABLED_DEFAULT,
    'show_fps': SHOW_FPS_DEFAULT,
    'show_debug': SHOW_DEBUG_DEFAULT,
    'show_graph': SHOW_GRAPH_DEFAULT,
}

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 480 < x < 635 and 10 < y < 50:
            state['sound_enabled'] = not state['sound_enabled']


# ══════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Emotion & Attention Tracker')
    parser.add_argument('--headless', action='store_true',
                        help='Run without OpenCV window (background mode)')
    args = parser.parse_args()
    headless = args.headless

    # Signal file for launcher to toggle window visibility
    signal_dir = os.path.dirname(os.path.abspath(__file__))
    show_signal = os.path.join(signal_dir, '.show_window')
    hide_signal = os.path.join(signal_dir, '.hide_window')
    window_visible = not headless

    # Graceful shutdown on SIGTERM (from launcher) or Ctrl+C
    def _graceful_shutdown(signum, frame):
        print(f"[MAIN] Received signal {signum}, shutting down...")
        stop_event.set()
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    try:
        signal.signal(signal.SIGBREAK, _graceful_shutdown)  # Windows Ctrl+Break
    except AttributeError:
        pass

    print("=" * 55)
    print("   ULTIMATE EMOTION & ATTENTION AI  v2.0")
    print("=" * 55)
    if headless:
        print(" Running in HEADLESS mode (no video window)")
    else:
        print(" q=Quit  d=Debug  f=FPS  g=Graph")
        print(" Click ALARM button to toggle sound")
    print("=" * 55)

    window_name = 'Ultimate Emotion & Attention AI v2.0'
    if not headless:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)

    frame_q = queue.Queue(maxsize=1)
    analysis_q = queue.Queue(maxsize=1)
    result_q = queue.Queue()

    threading.Thread(target=camera_worker, args=(frame_q,), daemon=True).start()
    threading.Thread(target=analysis_worker, args=(analysis_q, result_q), daemon=True).start()

    print("[MAIN] Loading YOLO...")
    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_MODEL)
    print("[MAIN] YOLO ready.")

    # Load ML productivity model if available
    productivity_model = None
    PRODUCTIVITY_LABELS = ['LOW', 'MEDIUM', 'HIGH']
    EMOTION_ENCODING = {
        'Happy': 4, 'Surprised': 3, 'Neutral': 2,
        'Tired / Sad': 1, 'Frustrated': 0, 'Scanning...': 2,
    }
    ml_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'models', 'productivity_model.pkl')
    try:
        import joblib
        if os.path.exists(ml_model_path):
            productivity_model = joblib.load(ml_model_path)
            print("[MAIN] ML productivity model loaded!")
        else:
            print("[MAIN] No ML model found — run train_model.py to enable predictions")
    except ImportError:
        print("[MAIN] joblib not installed — ML predictions disabled")
    except Exception as e:
        print(f"[MAIN] Could not load ML model: {e}")

    current_productivity = 'N/A'

    alarm = SoundAlert()
    email_alert = EmailAlert()
    logger = SessionLogger()
    engagement = EngagementTracker()
    timeline = TimelineGraph()

    tracks = {}
    frame_count = 0
    prev_time = time.time()

    print("[MAIN] Waiting for camera...")
    while frame_q.empty() and not stop_event.is_set():
        time.sleep(0.1)
    print("[MAIN] Running!\n")

    try:
        while not stop_event.is_set():
            try:
                img = frame_q.get(timeout=1.0)
            except queue.Empty:
                continue

            frame_count += 1
            h_frame, w_frame = img.shape[:2]

            # ── Consume analysis results ──
            while not result_q.empty():
                tid, raw_emo, is_looking, p, y, ear_l, ear_r, raw_scores = result_q.get()
                friendly = EMOTION_MAP.get(raw_emo, raw_emo)
                if tid in tracks:
                    tracks[tid]['history'].append(friendly)
                    vote = Counter(tracks[tid]['history']).most_common(1)[0][0]
                    tracks[tid]['label'] = vote
                    tracks[tid]['is_looking'] = is_looking
                    tracks[tid]['angles'] = (p, y)
                    tracks[tid]['raw_scores'] = raw_scores
                    if ear_l is not None and ear_r is not None:
                        tracks[tid]['ear'] = (ear_l, ear_r)

                    # Distraction detection (time-based: 20 seconds)
                    if not is_looking:
                        if tracks[tid]['looking_away_since'] is None:
                            tracks[tid]['looking_away_since'] = time.time()
                        elapsed_away = time.time() - tracks[tid]['looking_away_since']
                        tracks[tid]['is_distracted'] = elapsed_away >= DISTRACTION_DURATION_SECONDS
                    else:
                        tracks[tid]['looking_away_since'] = None
                        tracks[tid]['is_distracted'] = False

                    # Drowsiness detection (time-based: 3 seconds)
                    # Only update EAR if landmarks were found (not DNN fallback)
                    if ear_l is not None and ear_r is not None:
                        tracks[tid]['ear'] = (ear_l, ear_r)
                        tracks[tid]['ear_history'].append((ear_l + ear_r) / 2.0)

                    # Use smoothed EAR (rolling average) to filter noise
                    ear_hist = tracks[tid]['ear_history']
                    if len(ear_hist) > 0:
                        avg_ear = sum(ear_hist) / len(ear_hist)
                    else:
                        avg_ear = 0.3  # Default open-eye

                    if avg_ear < EAR_THRESHOLD:
                        if tracks[tid]['eyes_closed_since'] is None:
                            tracks[tid]['eyes_closed_since'] = time.time()
                        elapsed = time.time() - tracks[tid]['eyes_closed_since']
                        tracks[tid]['is_drowsy'] = elapsed >= DROWSY_DURATION_SECONDS
                    else:
                        tracks[tid]['eyes_closed_since'] = None
                        tracks[tid]['is_drowsy'] = False

                    # Track focus streaks
                    if is_looking and not tracks[tid].get('is_drowsy', False):
                        if tracks[tid]['focus_start_time'] is None:
                            tracks[tid]['focus_start_time'] = time.time()
                    else:
                        tracks[tid]['focus_start_time'] = None

                    # Count distraction events (rising edge)
                    if tracks[tid]['is_distracted'] and not tracks[tid].get('was_distracted', False):
                        tracks[tid]['total_distractions'] += 1
                    tracks[tid]['was_distracted'] = tracks[tid]['is_distracted']

                    # Count drowsy events (rising edge)
                    if tracks[tid]['is_drowsy'] and not tracks[tid].get('was_drowsy', False):
                        tracks[tid]['total_drowsy_events'] += 1
                    tracks[tid]['was_drowsy'] = tracks[tid]['is_drowsy']

                    # Log (throttled)
                    eng_score = engagement.update(is_looking, vote, tracks[tid]['is_drowsy'])
                    now_log = time.time()
                    if now_log - logger.last_log_time >= LOG_INTERVAL_SECONDS:
                        logger.last_log_time = now_log

                        # Compute derived features
                        fst = tracks[tid]['focus_start_time']
                        focus_dur = (now_log - fst) if fst else 0.0

                        hist = tracks[tid]['history']
                        if len(hist) > 0:
                            emotion_stab = hist.count(vote) / len(hist)
                        else:
                            emotion_stab = 1.0

                        session_elapsed = (now_log - logger.start_time) / 60.0

                        logger.log(
                            tid, vote, is_looking,
                            tracks[tid]['is_drowsy'], eng_score,
                            focus_dur,
                            tracks[tid]['total_distractions'],
                            tracks[tid]['total_drowsy_events'],
                            emotion_stab,
                            session_elapsed,
                            current_productivity
                        )

                        # ML productivity prediction
                        if productivity_model is not None:
                            try:
                                features = [[
                                    EMOTION_ENCODING.get(vote, 2),
                                    1 if is_looking else 0,
                                    1 if tracks[tid]['is_drowsy'] else 0,
                                    focus_dur,
                                    tracks[tid]['total_distractions'],
                                    tracks[tid]['total_drowsy_events'],
                                    emotion_stab,
                                    session_elapsed,
                                    eng_score,
                                ]]
                                pred = productivity_model.predict(features)[0]
                                current_productivity = PRODUCTIVITY_LABELS[pred]
                            except Exception:
                                current_productivity = 'N/A'

            # ── YOLO tracking ──
            if frame_count % YOLO_SKIP_FRAMES == 0:
                results = yolo_model.track(
                    img, classes=[0], persist=True,
                    imgsz=YOLO_IMG_SIZE, verbose=False,
                    conf=YOLO_CONFIDENCE, stream=True)
                for r in results:
                    for box in r.boxes:
                        if box.id is None:
                            continue
                        tid = int(box.id.item())
                        nx1, ny1, nx2, ny2 = map(int, box.xyxy[0])

                        if tid not in tracks:
                            tracks[tid] = {
                                'coords': [nx1, ny1, nx2, ny2],
                                'history': deque(maxlen=EMOTION_HISTORY_LEN),
                                'label': 'Scanning...', 'is_looking': True,
                                'is_drowsy': False, 'eyes_closed_since': None,
                                'is_distracted': False, 'looking_away_since': None,
                                'angles': (0, 0), 'raw_scores': {},
                                'ear': (0.3, 0.3), 'last_seen': frame_count,
                                'ear_history': deque(maxlen=EAR_SMOOTHING_WINDOW),
                                'focus_start_time': time.time(),
                                'first_seen_time': time.time(),
                                'total_distractions': 0,
                                'total_drowsy_events': 0,
                                'was_distracted': False,
                                'was_drowsy': False,
                            }
                            print(f"[TRACK] Person #{tid} appeared")
                        else:
                            ox1, oy1, ox2, oy2 = tracks[tid]['coords']
                            a = EMA_ALPHA
                            tracks[tid]['coords'] = [
                                int(ox1 * (1 - a) + nx1 * a),
                                int(oy1 * (1 - a) + ny1 * a),
                                int(ox2 * (1 - a) + nx2 * a),
                                int(oy2 * (1 - a) + ny2 * a),
                            ]
                            tracks[tid]['last_seen'] = frame_count

                        if analysis_q.empty():
                            analysis_q.put((tid, img.copy(), tracks[tid]['coords']))

            # ── Detect absent persons (left the camera) ──
            stale_tids = []
            for tid, data in list(tracks.items()):
                if frame_count - data['last_seen'] > ABSENCE_FRAME_THRESHOLD:
                    stale_tids.append(tid)

            for tid in stale_tids:
                data = tracks.pop(tid)
                first_seen = data.get('first_seen_time', time.time())
                duration = time.time() - first_seen
                mins = int(duration // 60)
                secs = int(duration % 60)
                label = data.get('label', 'Unknown')
                print(f"[ABSENT] Person #{tid} left after {mins}m {secs}s (emotion: {label})")

                # Log departure to CSV
                session_elapsed = (time.time() - logger.start_time) / 60.0
                logger.log(
                    tid, 'ABSENT', False, False, 0,
                    duration, data.get('total_distractions', 0),
                    data.get('total_drowsy_events', 0),
                    0.0, session_elapsed, 'N/A'
                )
                logger.screenshot(img, f"absent_person_{tid}")

            # ── Draw each tracked person ──
            anyone_distracted = False
            anyone_drowsy = False

            for tid, data in list(tracks.items()):
                if frame_count - data['last_seen'] > 15:
                    continue

                x1, y1, x2, y2 = data['coords']
                label = data['label']
                is_looking = data['is_looking']
                is_drowsy = data.get('is_drowsy', False)
                is_distracted = data.get('is_distracted', False)
                p, y_angle = data['angles']
                ear_l, ear_r = data.get('ear', (0.3, 0.3))
                raw_scores = data.get('raw_scores', {})

                if is_distracted:
                    anyone_distracted = True
                if is_drowsy:
                    anyone_drowsy = True

                draw_face_hud(img, x1, y1, x2, y2, tid, label,
                              is_distracted, is_drowsy, p, y_angle,
                              ear_l, ear_r, raw_scores, state['show_debug'])

            # ── Timeline graph update ──
            eng_score = engagement.get_score()
            timeline.add_point(eng_score, anyone_drowsy, anyone_distracted)

            # ── Global alerts ──
            alert_triggered = False
            alert_reason = ""

            if anyone_drowsy:
                alert_triggered = True
                alert_reason = "DROWSINESS"
            elif anyone_distracted:
                alert_triggered = True
                alert_reason = "DISTRACTION"

            if alert_triggered:
                alpha = abs(math.sin(time.time() * 4))
                overlay = img.copy()
                color = (0, 100, 255) if anyone_drowsy else (0, 0, 255)
                cv2.rectangle(overlay, (0, 55), (w_frame, 100), color, -1)
                cv2.addWeighted(overlay, 0.3 + 0.3 * alpha, img, 0.7 - 0.3 * alpha, 0, img)
                cv2.putText(img, f"ALERT: {alert_reason} DETECTED!",
                            (30, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)
                alarm.play()
                logger.screenshot(img, alert_reason.lower())

            # ── Check for show/hide signals from launcher ──
            if os.path.exists(show_signal):
                try:
                    os.remove(show_signal)
                except OSError:
                    pass
                if not window_visible:
                    cv2.namedWindow(window_name)
                    cv2.setMouseCallback(window_name, on_mouse)
                    window_visible = True
            if os.path.exists(hide_signal):
                try:
                    os.remove(hide_signal)
                except OSError:
                    pass
                if window_visible:
                    cv2.destroyWindow(window_name)
                    window_visible = False

            # ── HUD + Display (only when window is visible) ──
            if window_visible:
                # Alarm button
                btn_color = (0, 200, 0) if state['sound_enabled'] else (60, 60, 60)
                btn_text = "ALARM: ON" if state['sound_enabled'] else "ALARM: OFF"
                cv2.rectangle(img, (480, 10), (635, 50), btn_color, -1)
                cv2.rectangle(img, (480, 10), (635, 50), (255, 255, 255), 2)
                cv2.putText(img, btn_text, (492, 37),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # FPS
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                if state['show_fps']:
                    cv2.putText(img, f"FPS: {int(fps)}", (15, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

                # Engagement gauge
                draw_engagement_gauge(img, eng_score, w_frame - 55, 90)

                # Productivity badge (ML prediction)
                if current_productivity != 'N/A':
                    prod_colors = {
                        'HIGH': (0, 230, 118),
                        'MEDIUM': (0, 200, 255),
                        'LOW': (0, 0, 255)
                    }
                    pc = prod_colors.get(current_productivity, (200, 200, 200))
                    cv2.rectangle(img, (w_frame - 100, 115), (w_frame - 10, 143), pc, -1)
                    cv2.rectangle(img, (w_frame - 100, 115), (w_frame - 10, 143), (255, 255, 255), 1)
                    cv2.putText(img, current_productivity, (w_frame - 93, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

                # Person count
                n_people = sum(1 for d in tracks.values() if frame_count - d['last_seen'] <= 15)
                graph_top = h_frame - 120 if state['show_graph'] else h_frame - 20
                cv2.putText(img, f"Tracking: {n_people} person(s)", (15, graph_top),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

                # Timeline graph
                if state['show_graph']:
                    timeline.draw(img)

                cv2.imshow(window_name, img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop_event.set()
                    break
                elif key == ord('d'):
                    state['show_debug'] = not state['show_debug']
                elif key == ord('f'):
                    state['show_fps'] = not state['show_fps']
                elif key == ord('g'):
                    state['show_graph'] = not state['show_graph']
            else:
                # In headless mode, just update timing and sleep briefly
                now = time.time()
                prev_time = now
                time.sleep(0.03)  # ~30 FPS pacing

    finally:
        if window_visible:
            cv2.destroyAllWindows()
        logger.print_summary()
        summary_text = logger.get_summary_text()
        csv_path = logger.csv_path
        logger.close()  # Close CSV BEFORE sending email so it can be read
        email_alert.send_summary(summary_text, csv_path)
        # Clean up signal files
        for f in [show_signal, hide_signal]:
            try:
                os.remove(f)
            except OSError:
                pass
        print("[INFO] Application stopped.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 50)
        print("  CRITICAL ERROR")
        traceback.print_exc()
        if not any('--headless' in a for a in os.sys.argv):
            input("\nPress Enter to close...")
    finally:
        stop_event.set()
