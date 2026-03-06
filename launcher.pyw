"""
==========================================================
  EMOTION TRACKER — SYSTEM TRAY LAUNCHER
==========================================================
  Runs the tracker (main.py) and dashboard (dashboard.py)
  as background processes, controlled via a system tray icon.

  Menu:
    Show Window   → Make the tracker video window visible
    Hide Window   → Hide the video window
    Open Dashboard → Open http://localhost:5000 in browser
    Quit          → Graceful shutdown of all processes
==========================================================
"""

import os
import sys
import time
import signal
import subprocess
import webbrowser
import threading
from PIL import Image, ImageDraw, ImageFont

# Try to import pystray — required for system tray
try:
    import pystray
    from pystray import MenuItem, Menu
except ImportError:
    print("ERROR: pystray is required. Install it with: pip install pystray Pillow")
    sys.exit(1)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")
DASHBOARD_SCRIPT = os.path.join(BASE_DIR, "dashboard.py")
SHOW_SIGNAL = os.path.join(BASE_DIR, ".show_window")
HIDE_SIGNAL = os.path.join(BASE_DIR, ".hide_window")

# Python executable
PYTHON = sys.executable


def create_icon_image():
    """Create a simple brain emoji-style tray icon."""
    size = 64
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dark circle background
    draw.ellipse([2, 2, size - 2, size - 2], fill=(30, 30, 50, 230))

    # Brain-like shape (simplified)
    draw.ellipse([12, 10, 36, 38], fill=(120, 100, 220, 255))
    draw.ellipse([28, 10, 52, 38], fill=(140, 110, 230, 255))
    draw.ellipse([14, 24, 38, 52], fill=(110, 90, 210, 255))
    draw.ellipse([28, 24, 52, 52], fill=(130, 100, 225, 255))

    # Center divider line
    draw.line([(32, 12), (32, 50)], fill=(30, 30, 50, 200), width=2)

    # Border ring
    draw.ellipse([2, 2, size - 2, size - 2], outline=(100, 200, 255, 200), width=2)

    return img


class EmotionTrackerLauncher:
    def __init__(self):
        self.tracker_proc = None
        self.dashboard_proc = None
        self.icon = None
        self.window_visible = False

    def start_tracker(self):
        """Start main.py in headless mode."""
        print("[LAUNCHER] Starting tracker (headless)...")
        self.tracker_proc = subprocess.Popen(
            [PYTHON, MAIN_SCRIPT, "--headless"],
            cwd=BASE_DIR,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        print(f"[LAUNCHER] Tracker PID: {self.tracker_proc.pid}")

    def start_dashboard(self):
        """Start dashboard.py as a background process."""
        print("[LAUNCHER] Starting dashboard...")
        self.dashboard_proc = subprocess.Popen(
            [PYTHON, DASHBOARD_SCRIPT],
            cwd=BASE_DIR,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        print(f"[LAUNCHER] Dashboard PID: {self.dashboard_proc.pid}")

    def show_window(self, icon=None, item=None):
        """Signal the tracker to show the OpenCV window."""
        if self.tracker_proc and self.tracker_proc.poll() is None:
            # Remove hide signal if present
            try:
                os.remove(HIDE_SIGNAL)
            except OSError:
                pass
            # Create show signal
            with open(SHOW_SIGNAL, 'w') as f:
                f.write('show')
            self.window_visible = True
            print("[LAUNCHER] Show window signal sent")

    def hide_window(self, icon=None, item=None):
        """Signal the tracker to hide the OpenCV window."""
        if self.tracker_proc and self.tracker_proc.poll() is None:
            # Remove show signal if present
            try:
                os.remove(SHOW_SIGNAL)
            except OSError:
                pass
            # Create hide signal
            with open(HIDE_SIGNAL, 'w') as f:
                f.write('hide')
            self.window_visible = False
            print("[LAUNCHER] Hide window signal sent")

    def open_dashboard(self, icon=None, item=None):
        """Open the web dashboard in the default browser."""
        webbrowser.open("http://localhost:5000")

    def quit_app(self, icon=None, item=None):
        """Gracefully shut down all processes."""
        print("[LAUNCHER] Shutting down...")

        # Stop tracker gracefully
        if self.tracker_proc and self.tracker_proc.poll() is None:
            try:
                # Send CTRL_BREAK to the tracker's process group for graceful shutdown
                os.kill(self.tracker_proc.pid, signal.CTRL_BREAK_EVENT)
                print("[LAUNCHER] Sent shutdown signal to tracker, waiting...")
                self.tracker_proc.wait(timeout=30)  # Wait up to 30s for summary email
            except Exception as e:
                print(f"[LAUNCHER] Force-killing tracker: {e}")
                self.tracker_proc.kill()

        # Stop dashboard
        if self.dashboard_proc and self.dashboard_proc.poll() is None:
            try:
                self.dashboard_proc.terminate()
                self.dashboard_proc.wait(timeout=5)
            except Exception:
                self.dashboard_proc.kill()

        # Clean up signal files
        for f in [SHOW_SIGNAL, HIDE_SIGNAL]:
            try:
                os.remove(f)
            except OSError:
                pass

        print("[LAUNCHER] All processes stopped.")

        # Stop the tray icon
        if self.icon:
            self.icon.stop()

    def _check_processes(self):
        """Monitor subprocesses — restart on crash, shut down on clean exit."""
        while self.icon and self.icon.visible:
            if self.tracker_proc and self.tracker_proc.poll() is not None:
                exit_code = self.tracker_proc.returncode
                if exit_code == 0:
                    # Clean exit (user pressed 'q') — shut everything down
                    print("[LAUNCHER] Tracker exited normally (user quit). Shutting down...")
                    self.quit_app()
                    return
                else:
                    # Crash — auto-restart
                    print(f"[LAUNCHER] Tracker crashed (code {exit_code}), restarting...")
                    self.start_tracker()
            if self.dashboard_proc and self.dashboard_proc.poll() is not None:
                print("[LAUNCHER] Dashboard process died, restarting...")
                self.start_dashboard()
            time.sleep(5)

    def run(self):
        """Main entry point — start everything and show tray icon."""
        # Start subprocesses
        self.start_tracker()
        self.start_dashboard()

        # Start process monitor thread
        monitor = threading.Thread(target=self._check_processes, daemon=True)
        monitor.start()

        # Build tray menu
        menu = Menu(
            MenuItem("🧠 Emotion Tracker", None, enabled=False),
            Menu.SEPARATOR,
            MenuItem("Show Window", self.show_window),
            MenuItem("Hide Window", self.hide_window),
            Menu.SEPARATOR,
            MenuItem("Open Dashboard", self.open_dashboard),
            Menu.SEPARATOR,
            MenuItem("Quit", self.quit_app),
        )

        # Create and run tray icon (this blocks)
        self.icon = pystray.Icon(
            name="EmotionTracker",
            icon=create_icon_image(),
            title="Emotion & Attention Tracker",
            menu=menu,
        )

        print("[LAUNCHER] System tray icon active!")
        self.icon.run()


if __name__ == "__main__":
    launcher = EmotionTrackerLauncher()
    try:
        launcher.run()
    except KeyboardInterrupt:
        launcher.quit_app()
