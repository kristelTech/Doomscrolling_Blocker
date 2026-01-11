import cv2
import numpy as np
import random
import time
import threading
import subprocess
import os

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
            "Future you is watching. They're disappointed.",
            "Every scroll is a step backward. Look up!",
            "The algorithm wins again. Pathetic.",
            "Is this really more important than your goals?",
            "Your productivity just left the chat.",
            "Doomscrolling detected! You're better than this!",
            "PUT. THE. PHONE. DOWN. NOW.",
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

    def detect_doomscroll_dlib(self, frame, gray):
        """Detect doomscrolling using dlib landmarks"""
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)

            # Get key points
            nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
            chin = (landmarks.part(8).x, landmarks.part(8).y)
            forehead_approx = (landmarks.part(27).x, landmarks.part(27).y)

            # Left eye points (36-41) // I PLAYED AROUND WITH THESE NUMBERS IDK
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
            is_looking_down = head_tilt > 1.3 or eye_ratio < 0.35

            # Draw debug points
            cv2.circle(frame, nose_tip, 3, (0, 255, 0), -1)
            cv2.circle(frame, chin, 3, (255, 0, 0), -1)
            for pt in left_eye_points + right_eye_points:
                cv2.circle(frame, pt, 2, (0, 255, 255), -1)

            return is_looking_down

        return False

    def detect_doomscroll_opencv(self, frame, gray):
        """Detect doomscrolling using OpenCV Haar Cascades"""
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Region of interest for eyes
            roi_gray = gray[y:y+int(h*0.6), x:x+w]
            roi_color = frame[y:y+int(h*0.6), x:x+w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

            # Calculate face position - if face is in lower half = looking down
            face_center_y = y + h//2
            frame_height = frame.shape[0]
            face_position_ratio = face_center_y / frame_height

            # If face is in lower portion of frame and tilted, likely looking down
            is_looking_down = face_position_ratio > 0.55 and h < w * 1.2

            # Also check eye positions
            if len(eyes) >= 2:
                eye_y_positions = [y + ey + eh//2 for (ex, ey, ew, eh) in eyes]
                avg_eye_y = sum(eye_y_positions) / len(eye_y_positions)
                eye_position_in_face = (avg_eye_y - y) / h

                # If eyes are in lower part of detected face region = looking down
                if eye_position_in_face > 0.55:
                    is_looking_down = True

                # Draw eye rectangles
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            return is_looking_down

        return False

    def play_rickroll(self):
        #Play rickroll video with autoplay (only if not already playing)
        if not self.is_rickrolling and os.path.exists(self.rickroll_path):
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
                else:  # Windows - Someone test on windows pls
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

    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Doomscrolling Blocker Started!")
        print("Looking for your face...")
        print("Press 'q' to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                continue

            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect doomscrolling
            if self.use_dlib:
                is_doomscrolling = self.detect_doomscroll_dlib(frame, gray)
            else:
                is_doomscrolling = self.detect_doomscroll_opencv(frame, gray)

            if is_doomscrolling:
                self.show_roast(frame)
                # Play rickroll when doomscrolling
                self.play_rickroll()
            else:
                # Show encouraging message
                cv2.putText(frame, "Good posture! Keep it up!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Stop rickroll when back to normal
                self.stop_rickroll()

            # Display frame
            cv2.imshow('Doomscrolling Blocker', frame)

            # Exit on 'q'
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # Cleanup
        self.stop_rickroll()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = DoomscrollDetector()
    detector.run()
