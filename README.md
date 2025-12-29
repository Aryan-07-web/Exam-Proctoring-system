# 🎓 AI-Powered Online Exam Proctoring System

An AI-powered exam proctoring system that uses computer vision and machine learning to monitor students during online examinations. It tracks face presence, head pose, eye gaze, and movement patterns in real-time to detect potential violations and ensure academic integrity.

## ✨ Features

- **Real-Time Face Detection** - Advanced face detection using custom TensorFlow models
- **Head Pose Estimation** - Monitors head orientation (Yaw, Pitch, Roll) to detect when examinees turn away
- **Eye Gaze Tracking** - Detects where examinees are looking with warning system (3 warnings = violation)
- **Face Stability Monitoring** - Tracks face position stability to detect excessive movement
- **Multiple Violation Detection** - Detects multiple faces, camera obstruction, rapid movements, suspicious leaning
- **Real-Time Analytics** - Live dashboard with risk scoring and violation timeline
- **RAG-Powered Insights** - Post-session analysis with intelligent insights and recommendations

## 🛠️ Technology Stack

- **Python 3.x**
- **Streamlit** - Web interface
- **OpenCV** - Computer vision operations
- **TensorFlow/Keras** - Face detection model
- **NumPy & Pandas** - Data processing and analysis

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam/Camera access
- Windows/Linux/macOS

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name/FaceDetection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Required Model Files

The following model files are required for the application to work:

1. **Face Detection Model**: `facetracker_model_new.keras`
   - Place this file in the `FaceDetection` directory
   - If you don't have this model, you'll need to train your own or obtain it

2. **Facial Landmark Model**: `lbfmodel.yaml`
   - Download from: https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml
   - Or use the provided link in the code comments
   - Place this file in the `FaceDetection` directory

### 4. File Structure

Ensure your directory structure looks like this:

```
FaceDetection/
├── online_assessment.py          # Main application file
├── requirements.txt              # Python dependencies
├── facetracker_model_new.keras  # Face detection model (required)
├── lbfmodel.yaml                # Facial landmark model (required)
├── README.md                    # This file
└── .gitignore                   # Git ignore file
```

## 🎯 Running the Application

### Start the Application

```bash
streamlit run online_assessment.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Configure Settings**: Use the sidebar to configure detection features and violation thresholds
2. **Start Proctoring**: Click "▶️ Start Proctoring" to begin monitoring
3. **Monitor Session**: View real-time feed, metrics, and analytics
4. **Review Violations**: Check the Timeline tab for recorded violations
5. **Export Reports**: Export session data as JSON or CSV from the sidebar
6. **Generate Insights**: Use the RAG Insights tab for post-session analysis

## 📖 Usage Guidelines

### For Best Results:

- Ensure good lighting conditions
- Position camera at eye level
- Maintain consistent distance from camera (about 1-2 feet)
- Keep face visible at all times
- Minimize unnecessary head movements
- Use a quiet, distraction-free environment

### Detection Features:

- **Face Presence**: Ensures examinee is visible throughout the exam
- **Multiple Faces**: Detects if someone else enters the camera view
- **Head Pose**: Monitors if examinee turns head away from screen (±30° yaw, ±25° pitch)
- **Eye Gaze**: Tracks eye direction (warns when looking away, violation after 3 warnings)
- **Face Stability**: Detects rapid movement or instability
- **Leaning Detection**: Detects when examinee leans too close to camera

## 📁 Files to Upload to GitHub

### Required Files:
- ✅ `online_assessment.py` - Main application
- ✅ `requirements.txt` - Dependencies list
- ✅ `README.md` - Documentation
- ✅ `.gitignore` - Git ignore rules


## 🐛 Troubleshooting

### Camera Not Working
- Ensure camera permissions are granted
- Check if another application is using the camera
- Try changing camera index in code (default is 0)

### Model Loading Errors
- Verify `facetracker_model_new.keras` exists in the FaceDetection directory
- Ensure `lbfmodel.yaml` is downloaded and in the correct location
- Check that TensorFlow version is compatible

### OpenCV Issues
- Make sure both `opencv-python` and `opencv-contrib-python` are installed
- If face landmark detection fails, the system will use simplified calculations (still functional)

### Performance Issues
- Close other applications using the camera
- Reduce video resolution if needed
- Ensure adequate lighting for better detection


## 👤 Author

Aryan Rishi


- OpenCV community for computer vision tools
- TensorFlow team for ML framework
- Streamlit for web interface framework


