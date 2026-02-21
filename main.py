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
        self.detection_threshold = 3  # Frames needed to confirm state change

        # Detection state tracking for stability
        self.doomscroll_count = 0
        self.normal_count = 0
        self.detection_threshold = 1  # Instant response

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

            if face_position_ratio > 0.58:
                detection_score += 2
            elif face_position_ratio > 0.52:
                detection_score += 1

            # 2. Check face aspect ratio (looking down = face appears shorter/wider)
            aspect_ratio = h / w
            if aspect_ratio < 1.1:  # Face appears wider than normal
                detection_score += 1

            # 3. Also check eye positions
            if len(eyes) >= 2:
                eye_y_positions = [y + ey + eh//2 for (ex, ey, ew, eh) in eyes]
                avg_eye_y = sum(eye_y_positions) / len(eye_y_positions)
                eye_position_in_face = (avg_eye_y - y) / h

                # If eyes are in lower part of detected face region = looking down
                if eye_position_in_face > 0.6:
                    detection_score += 2
                elif eye_position_in_face > 0.52:
                    detection_score += 1

                # Draw eye rectangles
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            elif len(eyes) < 2:
                # If we can't detect eyes well, might be looking down
                detection_score += 1

            # Decision: doomscrolling if score >= 3
            is_looking_down = detection_score >= 3

            return is_looking_down

        return False

    def detect_available_cameras(self):
        """Detect available cameras with their names"""
        available_cameras = []
        camera_names = self.get_camera_names()
        
        for camera_index in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                camera_name = camera_names.get(camera_index, f"Camera {camera_index}")
                available_cameras.append((camera_index, camera_name))
                cap.release()
        return available_cameras

    def get_camera_names(self):
        """Get camera names from the system"""
        camera_names = {}
        
        try:
            import platform
            if platform.system() == 'Windows':
                # Try to get camera names from Windows using PowerShell
                camera_names = self._get_windows_camera_names()
            elif platform.system() == 'Linux':
                # Try to get camera names from Linux
                camera_names = self._get_linux_camera_names()
            elif platform.system() == 'Darwin':
                # Try to get camera names from macOS
                camera_names = self._get_macos_camera_names()
        except:
            pass
        
        return camera_names

    def _get_windows_camera_names(self):
        """Get camera names on Windows"""
        camera_names = {}
        try:
            import subprocess
        
            ps_command = "Get-PnpDevice -Class Camera | Select-Object FriendlyName | Select-Object -ExpandProperty FriendlyName"
            try:
                result = subprocess.run(['powershell', '-Command', ps_command], 
                                       capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    camera_list = result.stdout.strip().split('\n')
                    for idx, name in enumerate(camera_list):
                        if name.strip():
                            camera_names[idx] = name.strip()
            except:
                pass
            
            # Fallback: try WMI module
            if not camera_names:
                try:
                    import wmi
                    c = wmi.WMI()
                    devices = c.Win32_PnPDevice()
                    idx = 0
                    for device in devices:
                        if device.Description and ('camera' in device.Description.lower() or 'video' in device.Description.lower() or 'webcam' in device.Description.lower()):
                            camera_names[idx] = device.Description
                            idx += 1
                except:
                    pass
            
            # Fallback: try registry access
            if not camera_names:
                try:
                    import winreg
                    # Try to read from the camera/video device registry
                    paths = [
                        r"SYSTEM\CurrentControlSet\Services\usbvideo",
                        r"SYSTEM\CurrentControlSet\Enum\USB"
                    ]
                    idx = 0
                    for path in paths:
                        try:
                            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
                            subkey_idx = 0
                            while True:
                                try:
                                    subkey_name = winreg.EnumKey(key, subkey_idx)
                                    subkey_idx += 1
                                    subkey = winreg.OpenKey(key, subkey_name)
                                    try:
                                        name, _ = winreg.QueryValueEx(subkey, "FriendlyName")
                                        if 'camera' in name.lower() or 'video' in name.lower():
                                            camera_names[idx] = name
                                            idx += 1
                                    except:
                                        pass
                                    finally:
                                        winreg.CloseKey(subkey)
                                except OSError:
                                    break
                            winreg.CloseKey(key)
                        except:
                            pass
                except:
                    pass
        except:
            pass
        
        return camera_names

    def _get_linux_camera_names(self):
        """Get camera names on Linux"""
        camera_names = {}
        try:
            import subprocess
            for i in range(10):
                path = f"/sys/class/video4linux/video{i}/name"
                try:
                    with open(path, 'r') as f:
                        name = f.read().strip()
                        camera_names[i] = name
                except:
                    pass
        except:
            pass
        
        return camera_names

    def _get_macos_camera_names(self):
        """Get camera names on macOS"""
        camera_names = {}
        try:
            import subprocess
            result = subprocess.run(['system_profiler', 'SPCameraDataType'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                idx = 0
                for line in lines:
                    if 'Model' in line:
                        name = line.split(':')[1].strip()
                        camera_names[idx] = name
                        idx += 1
        except:
            pass
        
        return camera_names

    def select_camera(self):
        """Prompt user to select a camera"""
        available_cameras = self.detect_available_cameras()
        
        if not available_cameras:
            print("No cameras found!")
            return 0
        
        print("\nAvailable cameras:")
        for idx, (camera_id, camera_name) in enumerate(available_cameras):
            print(f"  {idx + 1}. {camera_name}")
        
        while True:
            try:
                choice = input(f"\nSelect a camera (1-{len(available_cameras)}): ").strip()
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_cameras):
                    return available_cameras[choice_idx][0]
                else:
                    print(f"Please enter a number between 1 and {len(available_cameras)}")
            except ValueError:
                print("Invalid input. Please enter a number.")

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
        # Let user select camera
        selected_camera = self.select_camera()
        cap = cv2.VideoCapture(selected_camera)

        if not cap.isOpened():
            print(f"Error: Could not open camera {selected_camera}")
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
