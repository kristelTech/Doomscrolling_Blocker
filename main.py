import cv2
import numpy as np
import random
import time
import threading
import subprocess
import os
from datetime import timedelta

class DoomscrollDetector:
    def __init__(self):
        # face landmark detection
        try:
            import dlib
            self.use_dlib = True
            self.detector = dlib.get_frontal_face_detector()
            # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            print("Using dlib for face tracking")
        except:
            # Fallback to OpenCV Haar Cascades
            self.use_dlib = False
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            print("Using OpenCV Haar Cascades for face tracking")

        # Roasting messages
        self.roasts = [
            "You'll fail if you don't stop!",
            "Your dreams called - they want your attention back!",
            "Scrolling won't make that deadline disappear!",
            "The phone can wait. Your future can't.",
            "Success doesn't scroll itself into existence!",
            "That screen won't study for you!",
            "Your goals > Your feed. Remember that.",
            "Your parents do not love you",
            "Future you is watching. They're disappointed.",
            "Every scroll is a step backward. Look up!",
            "The algorithm wins again. Pathetic.",
            "You will be alone forever",
            "Is this really more important than your goals?",
            "Your productivity just left the chat.",
            "Doomscrolling detected! You're better than this!",
            "PUT. THE. PHONE. DOWN. NOW.",
            "You re such a disappointment to your family",
            "This is why you're behind schedule."
        ]

        self.last_roast_time = 0
        self.roast_cooldown = 3  # seconds between roasts
        self.current_roast = ""
        self.prev_eye_ratio = 0.5

        # Rickroll video
        self.rickroll_path = "rickroll.mp4"
        self.rickroll_process = None
        self.is_rickrolling = False

        # Detection state tracking for stability
        self.doomscroll_count = 0
        self.normal_count = 0
        self.detection_threshold = 8  # Need 8 consecutive frames (~0.25 seconds) to trigger

        # ===== NEW: Time tracking and statistics =====
        self.session_start_time = time.time()
        self.focus_time = 0  # Total focused time in seconds
        self.doomscroll_time = 0  # Total doomscrolling time in seconds
        self.rickroll_triggers = 0  # Number of times rickroll was triggered
        self.last_state = None  # Track previous state ('focus' or 'doomscroll')
        self.state_start_time = time.time()  # When current state started
        self.has_triggered_rickroll_this_session = False  # Track if rickroll triggered in current doomscroll session
        
        # Calibration period - don't detect for first few seconds
        self.calibration_period = 3.0  # seconds
        self.is_calibrated = False

    def detect_doomscroll_dlib(self, frame, gray):
        """Detect doomscrolling using dlib landmarks"""
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)

            # Get key points
            nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
            chin = (landmarks.part(8).x, landmarks.part(8).y)
            forehead_approx = (landmarks.part(27).x, landmarks.part(27).y)

            # Left eye points (36-41)
            left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            # Right eye points (42-47)
            right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            # Calculate eye aspect ratio (detect if looking down)
            left_eye_top = (left_eye_points[1][1] + left_eye_points[2][1]) / 2
            left_eye_bottom = (left_eye_points[4][1] + left_eye_points[5][1]) / 2
            left_eye_center = (left_eye_points[0][1] + left_eye_points[3][1]) / 2

            right_eye_top = (right_eye_points[1][1] + right_eye_points[2][1]) / 2
            right_eye_bottom = (right_eye_points[4][1] + right_eye_points[5][1]) / 2
            right_eye_center = (right_eye_points[0][1] + right_eye_points[3][1]) / 2

            # Vertical eye position ratio
            left_ratio = abs(left_eye_center - left_eye_top) / (abs(left_eye_bottom - left_eye_top) + 1e-6)
            right_ratio = abs(right_eye_center - right_eye_top) / (abs(right_eye_bottom - right_eye_top) + 1e-6)
            eye_ratio = (left_ratio + right_ratio) / 2

            # Head tilt detection
            head_tilt = (chin[1] - nose_tip[1]) / (nose_tip[1] - forehead_approx[1] + 1e-6)

            # Looking down if head tilted forward or eyes positioned low
            is_looking_down = head_tilt > 1.4 or eye_ratio < 0.32

            # Draw debug points
            cv2.circle(frame, nose_tip, 3, (0, 255, 0), -1)
            cv2.circle(frame, chin, 3, (255, 0, 0), -1)
            for pt in left_eye_points + right_eye_points:
                cv2.circle(frame, pt, 2, (0, 255, 255), -1)

            return is_looking_down

        return False

    def detect_doomscroll_opencv(self, frame, gray):
        """Detect doomscrolling using OpenCV Haar Cascades with improved accuracy"""
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Region of interest for eyes
            roi_gray = gray[y:y+int(h*0.6), x:x+w]
            roi_color = frame[y:y+int(h*0.6), x:x+w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

            # Multiple detection criteria for better accuracy
            detection_score = 0

            # 1. Calculate face position - if face is in lower half = looking down
            face_center_y = y + h//2
            frame_height = frame.shape[0]
            face_position_ratio = face_center_y / frame_height

            if face_position_ratio > 0.62:  # Face very low in frame
                detection_score += 2
            elif face_position_ratio > 0.55:  # Face somewhat low
                detection_score += 1

            # 2. Check face aspect ratio (looking down = face appears shorter/wider)
            aspect_ratio = h / w
            if aspect_ratio < 1.08:  # Face appears significantly wider than normal
                detection_score += 1

            # 3. Also check eye positions
            if len(eyes) >= 2:
                eye_y_positions = [y + ey + eh//2 for (ex, ey, ew, eh) in eyes]
                avg_eye_y = sum(eye_y_positions) / len(eye_y_positions)
                eye_position_in_face = (avg_eye_y - y) / h

                # If eyes are in lower part of detected face region = looking down
                if eye_position_in_face > 0.62:  # Eyes very low in face
                    detection_score += 2
                elif eye_position_in_face > 0.55:  # Eyes somewhat low
                    detection_score += 1

                # Draw eye rectangles
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            elif len(eyes) < 2:
                # If we can't detect eyes well, might be looking down
                detection_score += 1

            # Decision: doomscrolling if score >= 3 (balanced sensitivity)
            is_looking_down = detection_score >= 3

            return is_looking_down

        return False

    def update_time_tracking(self, is_doomscrolling):
        """Update time tracking based on current state"""
        current_time = time.time()
        elapsed = current_time - self.state_start_time
        
        # Determine current state
        current_state = 'doomscroll' if is_doomscrolling else 'focus'
        
        # If state changed, update times
        if current_state != self.last_state and self.last_state is not None:
            if self.last_state == 'focus':
                self.focus_time += elapsed
            elif self.last_state == 'doomscroll':
                self.doomscroll_time += elapsed
                # Reset rickroll trigger flag when leaving doomscroll state
                self.has_triggered_rickroll_this_session = False
            
            self.state_start_time = current_time
            self.last_state = current_state
        elif self.last_state is None:
            # First state initialization
            self.last_state = current_state
            self.state_start_time = current_time

    def play_rickroll(self):
        """Play rickroll video with autoplay (only if not already playing)"""
        if not self.is_rickrolling and os.path.exists(self.rickroll_path):
            # Only increment counter once per doomscroll session
            if not self.has_triggered_rickroll_this_session:
                self.rickroll_triggers += 1
                self.has_triggered_rickroll_this_session = True
            
            self.is_rickrolling = True
            # Use system default video player with autoplay in background thread
            def start_video():
                if os.name == 'posix':  # macOS/Linux
                    if os.uname().sysname == 'Darwin':  # macOS
                        # Use afplay for audio or osascript to force QuickTime to play
                        self.rickroll_process = subprocess.Popen(
                            ['osascript', '-e', f'tell application "QuickTime Player" to open POSIX file "{os.path.abspath(self.rickroll_path)}"',
                             '-e', 'tell application "QuickTime Player" to play front document'],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    else:  # Linux
                        # Try vlc first because of autoplay, fallback to xdg-open
                        try:
                            self.rickroll_process = subprocess.Popen(
                                ['vlc', '--play-and-exit', self.rickroll_path],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                        except:
                            self.rickroll_process = subprocess.Popen(['xdg-open', self.rickroll_path])
                else:  # Windows
                    os.startfile(self.rickroll_path)

            # Start video in background thread to avoid blocking
            video_thread = threading.Thread(target=start_video, daemon=True)
            video_thread.start()

    def stop_rickroll(self):
        """Stop rickroll video"""
        if self.is_rickrolling:
            self.is_rickrolling = False
            if self.rickroll_process:
                try:
                    # Kill the video player process
                    if os.uname().sysname == 'Darwin':  # macOS
                        subprocess.run(['killall', 'QuickTime Player'], stderr=subprocess.DEVNULL)
                    self.rickroll_process.terminate()
                except:
                    pass
                self.rickroll_process = None

    def show_roast(self, frame):
        """Display roasting message on frame"""
        current_time = time.time()

        if current_time - self.last_roast_time > self.roast_cooldown:
            self.current_roast = random.choice(self.roasts)
            self.last_roast_time = current_time

        # Create semi-transparent overlay
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Draw red warning background
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Draw warning text
        cv2.putText(frame, "DOOMSCROLLING DETECTED!", (w//2 - 250, 50),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)

        # Draw roast message
        cv2.putText(frame, self.current_roast, (w//2 - 300, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def format_time(self, seconds):
        """Format seconds into readable time string"""
        return str(timedelta(seconds=int(seconds)))

    def show_stats(self, frame):
        """Display statistics on frame"""
        h, w = frame.shape[:2]
        
        # Current session info
        current_time = time.time()
        elapsed = current_time - self.state_start_time
        
        # Add current state time to totals for display
        display_focus_time = self.focus_time
        display_doomscroll_time = self.doomscroll_time
        
        if self.last_state == 'focus':
            display_focus_time += elapsed
        elif self.last_state == 'doomscroll':
            display_doomscroll_time += elapsed
        
        total_session_time = current_time - self.session_start_time
        
        # Create semi-transparent background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 140), (400, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Display stats
        y_offset = h - 120
        cv2.putText(frame, "=== SESSION STATS ===", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Focus Time: {self.format_time(display_focus_time)}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        y_offset += 25
        cv2.putText(frame, f"Doomscroll Time: {self.format_time(display_doomscroll_time)}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        y_offset += 25
        cv2.putText(frame, f"Rickrolls Triggered: {self.rickroll_triggers}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        y_offset += 25
        cv2.putText(frame, f"Total Time: {self.format_time(total_session_time)}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Doomscrolling Blocker Started!")
        print("Looking for your face...")
        print("Calibrating for 3 seconds - sit normally...")
        print("Press 'q' to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                continue

            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Check if still in calibration period
            time_elapsed = time.time() - self.session_start_time
            if not self.is_calibrated:
                if time_elapsed < self.calibration_period:
                    # Show calibration message
                    remaining = int(self.calibration_period - time_elapsed) + 1
                    cv2.putText(frame, f"CALIBRATING... {remaining}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                    cv2.imshow('Doomscrolling Blocker', frame)
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break
                    continue
                else:
                    self.is_calibrated = True
                    print("Calibration complete! Now monitoring...")

            # Detect doomscrolling
            if self.use_dlib:
                raw_detection = self.detect_doomscroll_dlib(frame, gray)
            else:
                raw_detection = self.detect_doomscroll_opencv(frame, gray)

            # Stabilize detection with frame counting to avoid flickering
            if raw_detection:
                self.doomscroll_count += 1
                self.normal_count = 0
            else:
                self.normal_count += 1
                self.doomscroll_count = 0

            # Only trigger if we've detected consistently for threshold frames
            is_doomscrolling = self.doomscroll_count >= self.detection_threshold
            is_normal = self.normal_count >= self.detection_threshold

            # Update time tracking
            self.update_time_tracking(is_doomscrolling)

            if is_doomscrolling:
                self.show_roast(frame)
                # Play rickroll when doomscrolling
                self.play_rickroll()
            elif is_normal:
                # Show encouraging message
                cv2.putText(frame, "Good posture! Keep it up!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Stop rickroll when back to normal
                self.stop_rickroll()
            else:
                # Transitioning state - show neutral message
                cv2.putText(frame, "Monitoring...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Always show stats
            self.show_stats(frame)

            # Display frame
            cv2.imshow('Doomscrolling Blocker', frame)

            # Exit on 'q'
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # Final stats before cleanup
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Total Focus Time: {self.format_time(self.focus_time)}")
        print(f"Total Doomscroll Time: {self.format_time(self.doomscroll_time)}")
        print(f"Rickrolls Triggered: {self.rickroll_triggers}")
        print(f"Total Session Time: {self.format_time(time.time() - self.session_start_time)}")
        print("="*50)

        # Cleanup
        self.stop_rickroll()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = DoomscrollDetector()
    detector.run()
