# Doomscrolling Blocker 📱🚫

A Python program that uses your webcam to detect when you're looking down at your phone (aka doomscrolling) and roasts you to get back to work!

## Features

- **Real-time face and eye tracking** using OpenCV
- **Doomscrolling detection** - detects when you tilt your head down
- **Motivational roasting** - displays harsh but motivating messages when caught
- **Automatic fallback** - works with dlib or OpenCV Haar Cascades

## Installation

### Basic (OpenCV only)

```bash
pip install opencv-python numpy
```

### Advanced (Better accuracy with dlib)

```bash
pip install opencv-python numpy dlib

# Download the face landmarks model
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

## Usage

```bash
python main.py
```

- The program will open your webcam
- Look at the screen normally = Green "Good posture!" message
- Look down at your phone = Red warning with roasting messages
- Press **'q'** to quit

## How It Works

1. **Face Detection**: Detects your face using either dlib or OpenCV Haar Cascades
2. **Posture Analysis**: Tracks head tilt and eye position
3. **Doomscroll Detection**: Triggers when:
   - Your head tilts down significantly
   - Your face moves to the lower portion of the frame
   - Your eyes are positioned low in your face region
4. **Roasting**: Displays motivational (harsh) messages every 3 seconds when caught

## Sample Roasts

- "You'll fail if you don't stop!"
- "Your dreams called - they want your attention back!"
- "Future you is watching. They're disappointed."
- "The algorithm wins again. Pathetic."
- "PUT. THE. PHONE. DOWN. NOW."

## Requirements

- Python 3.13+
- Webcam
- OpenCV (`opencv-python`)
- NumPy
- dlib (optional, for better accuracy)

## Customization

Edit `main.py` to customize:
- **Roast messages**: Modify the `self.roasts` list
- **Detection sensitivity**: Adjust `face_position_ratio` thresholds
- **Roast frequency**: Change `self.roast_cooldown` (default: 3 seconds)

## License

Free to use. Stay productive! 💪
