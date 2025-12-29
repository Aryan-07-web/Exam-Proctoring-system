"""
ADVANCED EXAM PROCTORING - OpenCV ONLY VERSION
===============================================
Uses OpenCV's built-in detectors + geometric algorithms
NO MediaPipe required!

REQUIRED:
pip install opencv-python opencv-contrib-python numpy pandas streamlit tensorflow

FEATURES:
1. Face presence detection (your model)
2. Multiple face detection (OpenCV Haar Cascade)
3. HEAD POSE ESTIMATION (yaw, pitch, roll) - using facial landmarks
4. EYE GAZE DETECTION - using eye region analysis
5. Face movement tracking (optical flow)
6. Face size tracking (leaning detection)
7. Camera obstruction detection
8. Temporal violation tracking
9. Risk scoring with evidence logging
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd

import time
import json
import os

st.set_page_config(layout="wide", page_title="Advanced Assesment Proctoring")

# ============================================================================
# OPENCV DETECTORS
# ============================================================================

@st.cache_resource
def load_opencv_detectors():
    """Load OpenCV's detectors"""
    detectors = {}
    
    try:
        # Face detector
        detectors['face'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Eye detector
        detectors['eye'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Try to load facial landmark detector (dlib-style)
        try:
            # Check if opencv-contrib-python is installed (has cv2.face module)
            if not hasattr(cv2, 'face'):
                raise ImportError("cv2.face module not found. Please install opencv-contrib-python: pip install opencv-contrib-python")
            
            # Get the absolute path to the model file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "lbfmodel.yaml")
            
            # Check if file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            # Create facemark detector
            facemark_detector = cv2.face.createFacemarkLBF()
            
            # Try loading the model - use forward slashes (Windows OpenCV sometimes prefers this)
            model_path_normalized = model_path.replace('\\', '/')
            
            # NOTE: Some OpenCV versions have a bug where loadModel() returns False
            # even when it successfully loads. We'll ignore the return value and 
            # assume success if no exception is raised.
            try:
                # Try loading with normalized path (forward slashes)
                facemark_detector.loadModel(model_path_normalized)
                # If no exception, assume it loaded (even if return value was False)
                detectors['facemark'] = facemark_detector
                st.success("✅ Facial landmark model loaded successfully!")
            except:
                # If normalized path failed, try original path
                try:
                    facemark_detector.loadModel(model_path)
                    detectors['facemark'] = facemark_detector
                    st.success("✅ Facial landmark model loaded successfully!")
                except Exception as load_error:
                    # Both attempts failed, raise error
                    raise RuntimeError(f"Failed to load model. Error: {str(load_error)}")
                
        except ImportError as e:
            detectors['facemark'] = None
            st.warning(f"⚠️ {str(e)}. Head pose will use simplified calculation.")
        except FileNotFoundError as e:
            detectors['facemark'] = None
            st.warning(f"⚠️ {str(e)}. Head pose will use simplified calculation.")
        except AttributeError as e:
            detectors['facemark'] = None
            st.warning(f"⚠️ cv2.face.createFacemarkLBF() not available: {str(e)}. Head pose will use simplified calculation.")
        except RuntimeError as e:
            detectors['facemark'] = None
            error_msg = str(e)
            st.warning(f"⚠️ Failed to load facial landmark model: {error_msg}. Head pose will use simplified calculation.")
            st.info("💡 **Note**: The simplified head pose calculation works well even without the landmark model.")
        except Exception as e:
            detectors['facemark'] = None
            st.warning(f"⚠️ Failed to load facial landmark model: {str(e)}. Head pose will use simplified calculation.")
        
    except Exception as e:
        st.warning(f"Some detectors failed to load: {e}")
    
    return detectors

# ============================================================================
# HEAD POSE ESTIMATION (Simplified without full landmarks)
# ============================================================================

class SimplifiedHeadPoseEstimator:
    """
    WHY: Estimates head orientation without MediaPipe
    HOW: Uses face bounding box position and eye positions
    """
    
    def __init__(self):
        self.baseline_face_size = None
        self.baseline_center = None
        
    def estimate_from_bbox_and_eyes(self, face_bbox, eye_positions, frame_shape):
        """
        Estimate head pose from face box and eye positions
        
        Returns: (yaw, pitch, roll) in degrees
        yaw: Left-Right head turn (-90 to +90)
        pitch: Up-Down head tilt (-90 to +90)  
        roll: Head tilt/rotation (-45 to +45)
        """
        h, w = frame_shape[:2]
        
        if face_bbox is None:
            return 0, 0, 0
        
        x1, y1, x2, y2 = face_bbox
        face_width = x2 - x1
        face_height = y2 - y1
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        
        # Establish baseline (first few frames)
        if self.baseline_face_size is None:
            self.baseline_face_size = face_width
            self.baseline_center = (face_center_x, face_center_y)
        
        # ==========================================
        # YAW (Left-Right turn)
        # ==========================================
        # If face moves to left of frame, they're turning right (positive yaw)
        # If face moves to right of frame, they're turning left (negative yaw)
        horizontal_offset = face_center_x - (w / 2)
        yaw = (horizontal_offset / (w / 2)) * 45  # Map to -45 to +45 degrees
        
        # Refine with face width (turned face appears narrower)
        width_ratio = face_width / self.baseline_face_size if self.baseline_face_size > 0 else 1
        if width_ratio < 0.85:  # Face appears narrower = significant turn
            yaw = yaw * 1.5  # Amplify yaw
        
        # ==========================================
        # PITCH (Up-Down tilt)
        # ==========================================
        # If face moves up in frame, they're tilting head up (negative pitch)
        # If face moves down in frame, they're tilting head down (positive pitch)
        vertical_offset = face_center_y - (h / 2)
        pitch = (vertical_offset / (h / 2)) * 30  # Map to -30 to +30 degrees
        
        # Refine with face height
        height_ratio = face_height / face_width if face_width > 0 else 1
        if height_ratio < 1.2:  # Face appears squashed = looking up
            pitch = pitch - 10
        elif height_ratio > 1.5:  # Face appears elongated = looking down
            pitch = pitch + 10
        
        # ==========================================
        # ROLL (Head tilt) - using eye positions
        # ==========================================
        roll = 0
        if eye_positions and len(eye_positions) == 2:
            left_eye, right_eye = eye_positions
            
            # Calculate eye line angle
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            
            if dx != 0:
                roll = np.degrees(np.arctan(dy / dx))
                roll = np.clip(roll, -45, 45)
        
        # Clip values to reasonable ranges
        yaw = np.clip(yaw, -90, 90)
        pitch = np.clip(pitch, -90, 90)
        
        return yaw, pitch, roll

# ============================================================================
# EYE GAZE ESTIMATION (Using eye region analysis)
# ============================================================================

class SimpleGazeEstimator:
    """
    WHY: Detects where eyes are looking without MediaPipe
    HOW: Analyzes pupil/iris position within eye region
    """
    
    def estimate_gaze(self, frame, eye_regions):
        """
        Estimate gaze direction from eye regions
        
        Returns: (gaze_direction, confidence)
        gaze_direction: 'center', 'left', 'right', 'up', 'down'
        """
        if not eye_regions or len(eye_regions) < 2:
            return 'center', 0.0
        
        gaze_scores = {'center': 0, 'left': 0, 'right': 0, 'up': 0, 'down': 0}
        
        for (ex, ey, ew, eh) in eye_regions:
            # Extract eye region
            eye_img = frame[ey:ey+eh, ex:ex+ew]
            
            if eye_img.size == 0:
                continue
            
            # Convert to grayscale
            eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY) if len(eye_img.shape) == 3 else eye_img
            
            # Apply threshold to find darkest regions (pupil/iris)
            _, threshold = cv2.threshold(eye_gray, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours (pupil candidates)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # Get largest contour (likely pupil)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get center of pupil
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                pupil_x = int(M["m10"] / M["m00"])
                pupil_y = int(M["m01"] / M["m00"])
                
                # Calculate position relative to eye region
                eye_center_x = ew / 2
                eye_center_y = eh / 2
                
                # Horizontal gaze
                horizontal_offset = (pupil_x - eye_center_x) / eye_center_x
                if horizontal_offset < -0.2:
                    gaze_scores['left'] += 1
                elif horizontal_offset > 0.2:
                    gaze_scores['right'] += 1
                else:
                    gaze_scores['center'] += 0.5
                
                # Vertical gaze
                vertical_offset = (pupil_y - eye_center_y) / eye_center_y
                if vertical_offset < -0.15:
                    gaze_scores['up'] += 1
                elif vertical_offset > 0.15:
                    gaze_scores['down'] += 1
                else:
                    gaze_scores['center'] += 0.5
        
        # Determine dominant gaze direction
        max_direction = max(gaze_scores, key=gaze_scores.get)
        confidence = gaze_scores[max_direction] / sum(gaze_scores.values()) if sum(gaze_scores.values()) > 0 else 0
        
        return max_direction, confidence

# ============================================================================
# FACE MOVEMENT TRACKING
# ============================================================================

class FaceMovementTracker:
    """Track face movement to detect suspicious behavior"""
    
    def __init__(self):
        self.previous_center = None
        self.movement_history = []
        self.rapid_movements = 0
        self.stability_scores_history = []  # Track stability scores over time
        self.rapid_decrease_start_time = None  # Track when stability starts decreasing rapidly
        
    def update(self, face_bbox):
        """Track face center movement"""
        if face_bbox is None:
            self.previous_center = None
            return 0, False
        
        # Calculate center
        x1, y1, x2, y2 = face_bbox
        current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        if self.previous_center is None:
            self.previous_center = current_center
            return 0, False
        
        # Calculate movement
        dx = current_center[0] - self.previous_center[0]
        dy = current_center[1] - self.previous_center[1]
        movement = np.sqrt(dx**2 + dy**2)
        
        self.movement_history.append(movement)
        if len(self.movement_history) > 30:  # Keep last 30 frames
            self.movement_history.pop(0)
        
        # Detect rapid movement (suspicious - looking around quickly)
        is_rapid = movement > 50  # Pixels
        if is_rapid:
            self.rapid_movements += 1
        
        self.previous_center = current_center
        
        return movement, is_rapid
    
    def get_movement_score(self):
        """Calculate movement stability score (0-100, higher = more stable)"""
        if len(self.movement_history) < 10:
            return 100
        
        avg_movement = np.mean(self.movement_history)
        
        # Lower average movement = more stable
        if avg_movement < 5:
            score = 100
        elif avg_movement < 15:
            score = 80
        elif avg_movement < 30:
            score = 60
        else:
            score = 40
        
        # Track stability scores for rapid decrease detection
        self.stability_scores_history.append(score)
        if len(self.stability_scores_history) > 90:  # Keep last 90 frames (~3 seconds at 30fps)
            self.stability_scores_history.pop(0)
        
        return score
    
    def check_rapid_stability_decrease(self, threshold_seconds=2.5):
        """
        Check if stability has been decreasing rapidly for more than threshold_seconds
        Returns: (is_violation, duration, previous_score, current_score)
        """
        if len(self.stability_scores_history) < 30:  # Need at least 1 second of data
            return False, 0, 100, 100
        
        current_score = self.stability_scores_history[-1]
        
        # Check last N frames (corresponding to threshold_seconds)
        # Assuming ~30 fps, so threshold_seconds * 30 frames
        check_window = int(threshold_seconds * 30)
        if len(self.stability_scores_history) < check_window:
            return False, 0, 100, current_score
        
        # Get scores from check_window frames ago and now
        past_score = self.stability_scores_history[-check_window]
        current_score = self.stability_scores_history[-1]
        
        # Calculate decrease rate (decrease of 20+ points in the window is considered rapid)
        score_decrease = past_score - current_score
        decrease_rate = score_decrease / threshold_seconds  # points per second
        
        # If stability decreased by 20+ points rapidly
        if score_decrease >= 20 and decrease_rate >= 8:  # At least 8 points per second
            if self.rapid_decrease_start_time is None:
                self.rapid_decrease_start_time = time.time()
            
            duration = time.time() - self.rapid_decrease_start_time
            if duration >= threshold_seconds:
                return True, duration, past_score, current_score
            else:
                return False, duration, past_score, current_score
        else:
            # Reset if stability is improving or not decreasing rapidly
            self.rapid_decrease_start_time = None
            return False, 0, past_score, current_score

# ============================================================================
# FACE SIZE TRACKING (Detects leaning in/out)
# ============================================================================

class FaceSizeTracker:
    """Track face size changes to detect leaning toward/away from camera"""
    
    def __init__(self):
        self.baseline_size = None
        self.size_history = []
        
    def update(self, face_bbox):
        """Track face size"""
        if face_bbox is None:
            return 0, "normal"
        
        x1, y1, x2, y2 = face_bbox
        width = x2 - x1
        height = y2 - y1
        size = width * height
        
        self.size_history.append(size)
        if len(self.size_history) > 30:
            self.size_history.pop(0)
        
        # Establish baseline
        if self.baseline_size is None and len(self.size_history) >= 10:
            self.baseline_size = np.mean(self.size_history)
        
        if self.baseline_size is None:
            return size, "normal"
        
        # Calculate size change
        size_ratio = size / self.baseline_size
        
        if size_ratio > 1.3:
            return size, "too_close"  # Leaning in (reading something?)
        elif size_ratio < 0.7:
            return size, "too_far"    # Leaning back (looking at something else?)
        else:
            return size, "normal"

# ============================================================================
# ENHANCED VIOLATION TRACKER
# ============================================================================

class EnhancedViolationTracker:
    """Enhanced violation tracking with head pose and gaze violations"""
    
    def __init__(self):
        self.violations = []
        self.current_violations = {}
        self.gaze_warnings = 0  # Track number of gaze warnings (violation after 3)
        self.last_gaze_warning_time = None  # Track when last warning was issued
        self.thresholds = {
            'face_absent': 3.0,
            'camera_blocked': 1.0,
            'multiple_faces': 0.5,
            'rapid_movement': 3.0,
            'leaning_suspicious': 5.0,
            'low_confidence': 5.0,
            'head_turned': 2.0,
            'eyes_off_screen': 2.0,
            'face_instability': 2.5    # NEW: Rapid stability decrease
        }
        self.severity_weights = {
            'face_absent': 10,
            'camera_blocked': 15,
            'multiple_faces': 20,
            'rapid_movement': 7,
            'leaning_suspicious': 5,
            'low_confidence': 3,
            'poor_lighting': 2,
            'head_turned': 8,
            'eyes_off_screen': 9,
            'face_instability': 6      # NEW: Rapid stability decrease
        }
        self.risk_score = 0
        self.screenshots_dir = "violation_screenshots"
        
        if not os.path.exists(self.screenshots_dir):
            os.makedirs(self.screenshots_dir)
    
    def start_violation(self, violation_type):
        if violation_type not in self.current_violations:
            self.current_violations[violation_type] = {
                'start_time': time.time(),
                'flagged': False
            }
    
    def end_violation(self, violation_type):
        if violation_type in self.current_violations:
            del self.current_violations[violation_type]
    
    def check_violation(self, violation_type, frame=None):
        if violation_type not in self.current_violations:
            return False, 0, None
        
        duration = time.time() - self.current_violations[violation_type]['start_time']
        threshold = self.thresholds.get(violation_type, 2.0)
        
        if duration >= threshold and not self.current_violations[violation_type]['flagged']:
            self.current_violations[violation_type]['flagged'] = True
            
            if duration >= threshold * 2:
                severity = 'HIGH'
            elif duration >= threshold * 1.5:
                severity = 'MEDIUM'
            else:
                severity = 'LOW'
            
            self.log_violation(violation_type, duration, severity, frame)
            return True, duration, severity
        
        return False, duration, None
    
    def log_violation(self, violation_type, duration, severity, frame=None):
        timestamp = datetime.now()
        
        violation_record = {
            'timestamp': timestamp.strftime("%H:%M:%S"),
            'type': violation_type,
            'duration': round(duration, 2),
            'severity': severity,
            'datetime': timestamp
        }
        
        self.violations.append(violation_record)
        
        weight = self.severity_weights.get(violation_type, 5)
        severity_multiplier = {'LOW': 1, 'MEDIUM': 1.5, 'HIGH': 2}
        self.risk_score += weight * severity_multiplier.get(severity, 1)
        
        if frame is not None:
            screenshot_path = os.path.join(
                self.screenshots_dir,
                f"{violation_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            cv2.imwrite(screenshot_path, frame)
            violation_record['screenshot'] = screenshot_path
    
    def get_risk_score(self):
        return min(100, self.risk_score)
    
    def get_violations_summary(self):
        if not self.violations:
            return {}
        
        df = pd.DataFrame(self.violations)
        
        summary = {
            'total_violations': len(self.violations),
            'by_type': df['type'].value_counts().to_dict(),
            'by_severity': df['severity'].value_counts().to_dict(),
            'risk_score': self.get_risk_score(),
            'violations': self.violations
        }
        
        return summary

# ============================================================================
# CAMERA INTEGRITY CHECKS
# ============================================================================

# ============================================================================
# HEAD POSE INTERPRETATION HELPERS
# ============================================================================

def _interpret_yaw(yaw):
    """Interpret yaw value in human-readable terms"""
    abs_yaw = abs(yaw)
    if abs_yaw < 15:
        return "Facing forward (Good)"
    elif abs_yaw < 30:
        return "Slightly turned" + (" right" if yaw > 0 else " left")
    elif abs_yaw < 60:
        return "Turned" + (" right" if yaw > 0 else " left") + " (Warning)"
    else:
        return "Looking away" + (" right" if yaw > 0 else " left") + " (High risk)"

def _interpret_pitch(pitch):
    """Interpret pitch value in human-readable terms"""
    abs_pitch = abs(pitch)
    if abs_pitch < 15:
        return "Head level (Good)"
    elif abs_pitch < 25:
        return "Slightly looking" + (" down" if pitch > 0 else " up")
    elif abs_pitch < 45:
        return "Looking" + (" down" if pitch > 0 else " up") + " (Warning)"
    else:
        return "Looking" + (" down" if pitch > 0 else " up") + " (High risk)"

def _interpret_roll(roll):
    """Interpret roll value in human-readable terms"""
    abs_roll = abs(roll)
    if abs_roll < 10:
        return "Head upright (Good)"
    elif abs_roll < 20:
        return "Slightly tilted"
    else:
        return "Tilted (Unusual position)"

# ============================================================================
# RAG INSIGHTS GENERATOR
# ============================================================================

def generate_rag_insights(report_data):
    """
    Generate insights from proctoring session report using RAG principles
    This is a simplified version - in production, this would use a proper RAG system
    with vector embeddings, semantic search, and LLM generation
    """
    insights = []
    
    # Extract key metrics
    total_violations = report_data.get('total_violations', 0)
    risk_score = report_data.get('risk_score', 0)
    movement_score = report_data.get('movement_score', 100)
    session_duration = report_data.get('session_duration', 0)
    violations_by_type = report_data.get('by_type', {})
    violations_by_severity = report_data.get('by_severity', {})
    
    # Insight 1: Overall Session Assessment
    if risk_score < 30:
        assessment = "Excellent"
        assessment_desc = "Your session showed minimal violations and excellent behavior."
    elif risk_score < 60:
        assessment = "Good"
        assessment_desc = "Your session was generally compliant with minor issues."
    elif risk_score < 80:
        assessment = "Fair"
        assessment_desc = "Your session had several violations that need attention."
    else:
        assessment = "Poor"
        assessment_desc = "Your session had significant violations that require immediate review."
    
    insights.append({
        'title': f'📊 Overall Session Assessment: {assessment}',
        'content': f"""
        **Risk Score**: {risk_score}/100
        
        {assessment_desc}
        
        **Key Statistics:**
        - Total Violations: {total_violations}
        - Stability Score: {movement_score}/100
        - Session Duration: {int(session_duration//60)} minutes {int(session_duration%60)} seconds
        """,
        'recommendations': [
            "Review the violation timeline to understand what triggered each violation",
            "Practice maintaining a stable position during exams" if movement_score < 80 else "Good job maintaining stability!",
            "Ensure consistent lighting and camera positioning" if violations_by_type.get('camera_blocked', 0) > 0 else None
        ]
    })
    
    # Remove None recommendations
    insights[-1]['recommendations'] = [r for r in insights[-1]['recommendations'] if r is not None]
    
    # Insight 2: Violation Analysis
    if total_violations > 0:
        violation_insight = "### 🚨 Violation Analysis\n\n"
        violation_insight += f"**Total Violations**: {total_violations}\n\n"
        
        if violations_by_type:
            violation_insight += "**Violations by Type:**\n"
            for vtype, count in violations_by_type.items():
                violation_insight += f"- {vtype.replace('_', ' ').title()}: {count}\n"
        
        if violations_by_severity:
            violation_insight += "\n**Violations by Severity:**\n"
            for sev, count in violations_by_severity.items():
                violation_insight += f"- {sev}: {count}\n"
        
        recommendations = []
        
        # Specific recommendations based on violation types
        if violations_by_type.get('head_turned', 0) > 0:
            recommendations.append("Try to keep your head facing forward - avoid turning away from the screen")
        if violations_by_type.get('eyes_off_screen', 0) > 0:
            recommendations.append("Focus your gaze on the screen center - avoid looking around")
        if violations_by_type.get('face_absent', 0) > 0:
            recommendations.append("Ensure your face remains visible to the camera at all times")
        if violations_by_type.get('multiple_faces', 0) > 0:
            recommendations.append("Make sure no one else is in the camera view during the exam")
        if violations_by_type.get('rapid_movement', 0) > 0 or violations_by_type.get('face_instability', 0) > 0:
            recommendations.append("Minimize head movements - try to remain as still as possible")
        
        insights.append({
            'title': '🚨 Violation Analysis',
            'content': violation_insight,
            'recommendations': recommendations if recommendations else ["Review your session to identify patterns in violations"]
        })
    
    # Insight 3: Stability Analysis
    if movement_score < 80:
        stability_desc = "Your face position showed significant instability during the session."
        if movement_score < 50:
            stability_desc += " Frequent movements were detected."
    else:
        stability_desc = "Your face position remained relatively stable throughout the session."
    
    insights.append({
        'title': '📊 Stability Analysis',
        'content': f"""
        **Stability Score**: {movement_score}/100
        
        {stability_desc}
        """,
        'recommendations': [
            "Find a comfortable position before starting the exam and try to maintain it",
            "Use a stable chair and desk setup to minimize unnecessary movements",
            "Take short breaks if you need to adjust your position" if movement_score < 60 else "Keep up the good work maintaining stability!"
        ]
    })
    
    # Insight 4: Behavioral Patterns
    metrics_history = report_data.get('metrics_history', {})
    if metrics_history and 'risk_scores' in metrics_history and len(metrics_history['risk_scores']) > 10:
        risk_scores = metrics_history['risk_scores']
        early_avg = np.mean(risk_scores[:len(risk_scores)//3]) if len(risk_scores) >= 3 else risk_scores[0]
        late_avg = np.mean(risk_scores[-len(risk_scores)//3:]) if len(risk_scores) >= 3 else risk_scores[-1]
        
        if late_avg > early_avg + 20:
            pattern_desc = "Your risk score increased throughout the session, suggesting declining compliance over time."
            pattern_rec = "Take breaks if needed, but maintain consistent behavior throughout the entire session"
        elif late_avg < early_avg - 10:
            pattern_desc = "Your risk score improved during the session, showing better compliance over time."
            pattern_rec = "Great improvement! Try to maintain this level of compliance from the start"
        else:
            pattern_desc = "Your risk score remained relatively consistent throughout the session."
            pattern_rec = "Maintain this consistent behavior in future sessions"
        
        insights.append({
            'title': '📈 Behavioral Patterns',
            'content': f"""
            **Risk Score Trend**: {pattern_desc}
            
            - Early session average: {early_avg:.1f}/100
            - Late session average: {late_avg:.1f}/100
            """,
            'recommendations': [pattern_rec]
        })
    
    return insights

def detect_camera_issues(frame):
    """Detect blur and poor lighting"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(100, laplacian_var / 10)
    is_obstructed = blur_score < 10
    
    # Brightness detection
    brightness = np.mean(gray)
    is_poor_lighting = brightness < 50 or brightness > 200
    
    return is_obstructed, is_poor_lighting, blur_score, brightness

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🎓 Advanced Exam Proctoring System")
    st.markdown("*Using OpenCV + Your Face Detection Model*")
    
    # Load models
    @st.cache_resource
    def load_model():
        try:
            return tf.keras.models.load_model("facetracker_model_new.keras")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            return None
    
    model = load_model()
    detectors = load_opencv_detectors()
    
    if model is None:
        st.stop()
    
    # Initialize trackers
    if 'violation_tracker' not in st.session_state:
        st.session_state.violation_tracker = EnhancedViolationTracker()
    if 'movement_tracker' not in st.session_state:
        st.session_state.movement_tracker = FaceMovementTracker()
    if 'size_tracker' not in st.session_state:
        st.session_state.size_tracker = FaceSizeTracker()
    if 'head_pose_estimator' not in st.session_state:
        st.session_state.head_pose_estimator = SimplifiedHeadPoseEstimator()
    if 'gaze_estimator' not in st.session_state:
        st.session_state.gaze_estimator = SimpleGazeEstimator()
    if 'exam_running' not in st.session_state:
        st.session_state.exam_running = False
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = None
    if 'session_metrics_history' not in st.session_state:
        st.session_state.session_metrics_history = {
            'timestamps': [],
            'risk_scores': [],
            'face_detected': [],
            'head_yaw': [],
            'head_pitch': [],
            'head_roll': []
        }
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Proctoring Settings")
        
        st.subheader("Detection Features")
        detect_multiple = st.checkbox("Detect Multiple Faces", value=True, help="Uses OpenCV Haar Cascade")
        detect_movement = st.checkbox("Track Suspicious Movement", value=True)
        detect_leaning = st.checkbox("Detect Leaning Behavior", value=True)
        detect_head_pose = st.checkbox("Head Pose Estimation", value=True, help="Track head orientation (yaw, pitch, roll)")
        detect_gaze = st.checkbox("Eye Gaze Tracking", value=True, help="Detect where eyes are looking")
        
        st.markdown("---")
        st.subheader("Violation Thresholds")
        face_absent_threshold = st.slider("Face Absent (sec)", 1, 10, 3)
        rapid_movement_threshold = st.slider("Rapid Movement (sec)", 1, 5, 3)
        
        st.session_state.violation_tracker.thresholds['face_absent'] = float(face_absent_threshold)
        st.session_state.violation_tracker.thresholds['rapid_movement'] = float(rapid_movement_threshold)
        
        st.markdown("---")
        st.subheader("📊 Statistics")
        
        summary = st.session_state.violation_tracker.get_violations_summary()
        
        if summary:
            st.metric("Violations", summary.get('total_violations', 0))
            st.metric("Risk Score", f"{summary.get('risk_score', 0)}/100")
            
            movement_score = st.session_state.movement_tracker.get_movement_score()
            st.metric("Stability Score", f"{movement_score}/100")
            
            st.markdown("### Recent Violations")
            if 'violations' in summary and summary['violations']:
                recent = summary['violations'][-5:]
                for v in reversed(recent):
                    st.markdown(f"**{v['timestamp']}** - {v['type']} ({v['severity']})")
        
        st.markdown("---")
        st.subheader("📥 Export Options")
        
        if st.button("📄 Export JSON Report", use_container_width=True):
            summary = st.session_state.violation_tracker.get_violations_summary()
            if summary:
                # Add comprehensive data
                summary['movement_score'] = st.session_state.movement_tracker.get_movement_score()
                summary['rapid_movements'] = st.session_state.movement_tracker.rapid_movements
                summary['session_duration'] = time.time() - st.session_state.session_start_time if st.session_state.session_start_time else 0
                summary['session_start'] = datetime.fromtimestamp(st.session_state.session_start_time).isoformat() if st.session_state.session_start_time else None
                summary['session_end'] = datetime.now().isoformat()
                summary['metrics_history'] = st.session_state.session_metrics_history
                
                report_json = json.dumps(summary, indent=2, default=str)
                st.download_button(
                    "⬇️ Download JSON",
                    data=report_json,
                    file_name=f"proctoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("No data to export yet. Start a session first.")
        
        if st.button("📊 Export CSV Data", use_container_width=True):
            if st.session_state.violation_tracker.violations:
                df = pd.DataFrame(st.session_state.violation_tracker.violations)
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv,
                    file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No violations to export yet.")
    
    # Session info
    if st.session_state.exam_running and st.session_state.session_start_time:
        elapsed = time.time() - st.session_state.session_start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        session_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        session_time = "00:00:00"
    
    # Header with session info
    header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
    with header_col1:
        st.markdown("### 🎓 Advanced Exam Proctoring System")
    with header_col2:
        st.markdown(f"**Session Time:** {session_time}")
    with header_col3:
        if st.session_state.exam_running:
            st.markdown('<div style="background: #d4edda; padding: 0.5rem; border-radius: 0.5rem; text-align: center;"><strong>🟢 LIVE</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background: #f8d7da; padding: 0.5rem; border-radius: 0.5rem; text-align: center;"><strong>🔴 STOPPED</strong></div>', unsafe_allow_html=True)
    
    # Controls
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("▶️ Start Proctoring", type="primary", disabled=st.session_state.exam_running, use_container_width=True):
            st.session_state.exam_running = True
            st.session_state.session_start_time = time.time()
            st.session_state.session_metrics_history = {
                'timestamps': [],
                'risk_scores': [],
                'face_detected': [],
                'head_yaw': [],
                'head_pitch': [],
                'head_roll': []
            }
            st.rerun()
    
    with col_btn2:
        if st.button("⏹️ Stop Proctoring", disabled=not st.session_state.exam_running, use_container_width=True):
            st.session_state.exam_running = False
            st.session_state.session_start_time = None
            st.rerun()
    
    with col_btn3:
        if st.button("🔄 Reset Session", disabled=st.session_state.exam_running, use_container_width=True):
            st.session_state.violation_tracker = EnhancedViolationTracker()
            st.session_state.movement_tracker = FaceMovementTracker()
            st.session_state.size_tracker = FaceSizeTracker()
            st.session_state.head_pose_estimator = SimplifiedHeadPoseEstimator()
            st.session_state.gaze_estimator = SimpleGazeEstimator()
            st.session_state.session_metrics_history = {
                'timestamps': [],
                'risk_scores': [],
                'face_detected': [],
                'head_yaw': [],
                'head_pitch': [],
                'head_roll': []
            }
            st.rerun()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📹 Live Feed", "📊 Analytics", "📋 Timeline", "📖 Help & Info", "🤖 RAG Insights"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📹 Live Proctoring Feed")
            FRAME_WINDOW = st.empty()
            warning_display = st.empty()
        
        with col2:
            st.subheader("📊 Live Metrics")
            metric1 = st.empty()
            metric2 = st.empty()
            metric3 = st.empty()
            metric4 = st.empty()
            metric5 = st.empty()
            metric6 = st.empty()
            metric7 = st.empty()
    
    # Tab2 and Tab3 - Display data directly (Streamlit reruns, so this will show current data)
    tracker = st.session_state.violation_tracker
    
    with tab2:
        st.subheader("📊 Session Analytics")
        
        analytics_col1, analytics_col2 = st.columns(2)
        
        # Display charts if we have data
        if len(st.session_state.session_metrics_history['timestamps']) > 1:
            with analytics_col1:
                # Risk Score Over Time
                st.markdown("**Risk Score Over Time**")
                risk_df = pd.DataFrame({
                    'Time (seconds)': list(range(len(st.session_state.session_metrics_history['risk_scores']))),
                    'Risk Score': st.session_state.session_metrics_history['risk_scores']
                })
                risk_df = risk_df.set_index('Time (seconds)')
                st.line_chart(risk_df, height=250)
                
                # Face Detection Status
                st.markdown("**Face Detection Status**")
                face_df = pd.DataFrame({
                    'Time (seconds)': list(range(len(st.session_state.session_metrics_history['face_detected']))),
                    'Face Detected': st.session_state.session_metrics_history['face_detected']
                })
                face_df = face_df.set_index('Time (seconds)')
                st.area_chart(face_df, height=250)
            
            with analytics_col2:
                # Head Pose (Yaw, Pitch, Roll)
                if len(st.session_state.session_metrics_history['head_yaw']) > 0:
                    st.markdown("**Head Pose (Yaw, Pitch, Roll)**")
                    head_pose_df = pd.DataFrame({
                        'Time (seconds)': list(range(len(st.session_state.session_metrics_history['head_yaw']))),
                        'Yaw': st.session_state.session_metrics_history['head_yaw'],
                        'Pitch': st.session_state.session_metrics_history['head_pitch'],
                        'Roll': st.session_state.session_metrics_history['head_roll']
                    })
                    head_pose_df = head_pose_df.set_index('Time (seconds)')
                    st.line_chart(head_pose_df, height=250)
                
                # Violations by Type
                if tracker.violations:
                    st.markdown("**Violations by Type**")
                    df_violations = pd.DataFrame(tracker.violations)
                    violation_counts = df_violations['type'].value_counts()
                    
                    # Create a DataFrame for the bar chart
                    violations_df = pd.DataFrame({
                        'Violation Type': violation_counts.index,
                        'Count': violation_counts.values
                    })
                    violations_df = violations_df.set_index('Violation Type')
                    st.bar_chart(violations_df, height=250)
                else:
                    st.info("No violations recorded yet.")
        else:
            st.info("📊 Start a proctoring session to see analytics. Data will appear here once the session begins.")
    
    with tab3:
        st.subheader("📋 Violation Timeline")
        
        if tracker.violations and len(tracker.violations) > 0:
            # Create a scrollable container for violations
            timeline_container = st.container()
            with timeline_container:
                for v in reversed(tracker.violations):  # Show all violations, most recent first
                    severity_color = {'LOW': '#ffc107', 'MEDIUM': '#ff9800', 'HIGH': '#f44336'}.get(v.get('severity', 'LOW'), '#757575')
                    violation_type = v.get('type', 'unknown').replace('_', ' ').title()
                    timestamp = v.get('timestamp', 'N/A')
                    duration = v.get('duration', 0)
                    severity = v.get('severity', 'LOW')
                    
                    st.markdown(
                        f"""
                        <div style='background: #f5f5f5; padding: 0.75rem; margin: 0.5rem 0; border-radius: 0.5rem; border-left: 4px solid {severity_color};'>
                            <strong>{timestamp}</strong> - {violation_type}<br>
                            <small>Duration: {duration:.2f}s | Severity: <span style='color: {severity_color}; font-weight: bold;'>{severity}</span></small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("📋 No violations recorded yet. Start a proctoring session and violations will appear here.")
    
    # Help & Info Tab
    with tab4:
        st.subheader("📖 Understanding Proctoring Features")
        
        info_tabs = st.tabs(["🎯 Head Pose Estimation", "👁️ Eye Gaze Detection", "📊 Face Stability"])
        
        with info_tabs[0]:
            st.markdown("### 🎯 Head Pose Estimation (Yaw, Pitch, Roll)")
            st.markdown("""
            **Head pose estimation** tracks the orientation of your head in 3D space. It measures three angles:
            """)
            
            col_yaw, col_pitch, col_roll = st.columns(3)
            
            with col_yaw:
                st.markdown("""
                #### 🔄 **YAW** (Left-Right Turn)
                - **What it measures**: How much you turn your head left or right
                - **Value range**: -90° to +90°
                - **Positive values**: Turning right (head moves left in frame)
                - **Negative values**: Turning left (head moves right in frame)
                - **0°**: Facing directly forward
                - **Used for**: Detecting if you're looking away from the screen
                - **Violation threshold**: ±30° sustained for 2+ seconds
                """)
            
            with col_pitch:
                st.markdown("""
                #### ⬆️⬇️ **PITCH** (Up-Down Tilt)
                - **What it measures**: How much you tilt your head up or down
                - **Value range**: -90° to +90°
                - **Positive values**: Tilting head down (looking down)
                - **Negative values**: Tilting head up (looking up)
                - **0°**: Head level, looking straight ahead
                - **Used for**: Detecting if you're looking at your desk/phone
                - **Violation threshold**: ±25° sustained for 2+ seconds
                """)
            
            with col_roll:
                st.markdown("""
                #### 🔃 **ROLL** (Head Tilt/Rotation)
                - **What it measures**: How much you tilt your head sideways (like tilting your ear toward shoulder)
                - **Value range**: -45° to +45°
                - **Positive values**: Tilting right ear down
                - **Negative values**: Tilting left ear down
                - **0°**: Head upright, not tilted
                - **Used for**: Detecting unusual head positions
                - **Note**: Usually not a violation unless extreme
                """)
            
            st.markdown("---")
            st.markdown("### 📊 Real-Time Interpretation")
            
            if st.session_state.exam_running:
                # Get latest values from session state if available
                yaw = st.session_state.get('last_yaw', 0)
                pitch = st.session_state.get('last_pitch', 0)
                roll = st.session_state.get('last_roll', 0)
                
                st.markdown(f"""
                **Current Values:**
                - **Yaw**: {yaw:.1f}° - {_interpret_yaw(yaw)}
                - **Pitch**: {pitch:.1f}° - {_interpret_pitch(pitch)}
                - **Roll**: {roll:.1f}° - {_interpret_roll(roll)}
                """)
                
                # Visual indicators
                yaw_status = "✅ Normal" if abs(yaw) < 30 else "⚠️ Turned away"
                pitch_status = "✅ Normal" if abs(pitch) < 25 else "⚠️ Looking up/down"
                roll_status = "✅ Normal" if abs(roll) < 15 else "⚠️ Tilted"
                
                st.markdown(f"""
                **Status:**
                - **Yaw Status**: {yaw_status}
                - **Pitch Status**: {pitch_status}
                - **Roll Status**: {roll_status}
                """)
            else:
                st.info("Start a proctoring session to see real-time head pose values.")
            
            st.markdown("---")
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Keep your head facing forward (yaw ≈ 0°) for best results
            - Maintain a level head position (pitch ≈ 0°)
            - Avoid frequent head movements - stay as still as possible
            - Small movements are normal and won't trigger violations
            """)
        
        with info_tabs[1]:
            st.markdown("### 👁️ Eye Gaze Detection")
            st.markdown("""
            **Eye gaze detection** analyzes where you're looking by tracking the position of your pupils/irises within your eye regions.
            """)
            
            st.markdown("""
            #### How It Works
            
            1. **Eye Region Detection**: The system detects your eyes using computer vision
            2. **Pupil/Iris Tracking**: It identifies the darkest regions (pupils/irises) within each eye
            3. **Position Analysis**: It calculates where your pupils are positioned relative to the center of your eyes
            4. **Direction Classification**: Based on pupil position, it determines gaze direction
            
            #### Gaze Directions
            
            - **CENTER** ✅: Your eyes are looking straight ahead at the screen
              - Pupils are near the center of the eye regions
              - This is the expected position during an exam
            
            - **LEFT** ⬅️: Your eyes are looking to the left
              - Pupils shifted toward the left side of the eye regions
              - May indicate looking at something beside your screen
            
            - **RIGHT** ➡️: Your eyes are looking to the right
              - Pupils shifted toward the right side of the eye regions
              - May indicate looking at something beside your screen
            
            - **UP** ⬆️: Your eyes are looking upward
              - Pupils shifted toward the top of the eye regions
              - May indicate looking above your screen
            
            - **DOWN** ⬇️: Your eyes are looking downward
              - Pupils shifted toward the bottom of the eye regions
              - May indicate looking at your desk, phone, or keyboard
            
            #### Confidence Score
            
            The system provides a **confidence score** (0.0 to 1.0) indicating how certain it is about the detected gaze direction:
            - **High confidence (>0.6)**: Strong detection, likely accurate
            - **Medium confidence (0.4-0.6)**: Moderate detection
            - **Low confidence (<0.4)**: Uncertain detection, may be affected by lighting or positioning
            
            #### Violation Detection
            
            When your gaze is detected as **LEFT**, **RIGHT**, **UP**, or **DOWN** with **high confidence (>0.6)** for more than **2 seconds**, it may be flagged as a violation.
            
            **Note**: Brief glances are normal and won't trigger violations. Only sustained gazes away from the center are flagged.
            """)
            
            st.markdown("---")
            st.markdown("### 📊 Current Gaze Status")
            
            if st.session_state.exam_running:
                gaze_dir = st.session_state.get('last_gaze_direction', 'center')
                gaze_conf = st.session_state.get('last_gaze_confidence', 0.0)
                
                gaze_icons = {
                    'center': '✅',
                    'left': '⬅️',
                    'right': '➡️',
                    'up': '⬆️',
                    'down': '⬇️'
                }
                
                icon = gaze_icons.get(gaze_dir, '❓')
                status = "✅ Looking at screen" if gaze_dir == 'center' else f"⚠️ Looking {gaze_dir}"
                
                st.markdown(f"""
                **Current Gaze**: {icon} **{gaze_dir.upper()}**  
                **Confidence**: {gaze_conf:.2f}  
                **Status**: {status}
                """)
            else:
                st.info("Start a proctoring session to see real-time gaze detection.")
            
            st.markdown("---")
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Keep your eyes focused on the screen center
            - Avoid looking around frequently
            - Ensure good lighting so your eyes are clearly visible
            - Sit at a comfortable distance from the camera
            """)
        
        with info_tabs[2]:
            st.markdown("### 📊 Face Stability Tracking")
            st.markdown("""
            **Face stability** measures how still and stable your face position is during the exam.
            """)
            
            st.markdown("""
            #### How It Works
            
            1. **Position Tracking**: The system continuously tracks the center position of your face
            2. **Movement Calculation**: It calculates how much your face moves between frames
            3. **Stability Score**: It computes a stability score from 0-100:
               - **90-100**: Excellent - Very stable, minimal movement
               - **70-89**: Good - Mostly stable with occasional small movements
               - **50-69**: Fair - Moderate movement detected
               - **0-49**: Poor - Frequent or large movements
            
            4. **Rapid Decrease Detection**: The system monitors if stability decreases rapidly
            
            #### Violation: Rapid Stability Decrease
            
            If your face stability decreases rapidly (by 20+ points) for more than **2.5 seconds**, it's flagged as a violation.
            
            This may indicate:
            - Frequent head movements
            - Shifting position frequently
            - Fidgeting or restlessness
            
            When detected, you'll see a warning: **"⚠️ Face stability decreasing - Please remain still!"**
            """)
            
            st.markdown("---")
            st.markdown("### 📊 Current Stability Status")
            
            if st.session_state.exam_running and 'movement_tracker' in st.session_state:
                stability_score = st.session_state.movement_tracker.get_movement_score()
                
                if stability_score >= 90:
                    status = "✅ Excellent - Very stable"
                    color = "green"
                elif stability_score >= 70:
                    status = "✅ Good - Mostly stable"
                    color = "lightgreen"
                elif stability_score >= 50:
                    status = "⚠️ Fair - Some movement detected"
                    color = "orange"
                else:
                    status = "⚠️ Poor - Too much movement"
                    color = "red"
                
                st.markdown(f"""
                **Stability Score**: **{stability_score}/100**  
                **Status**: <span style='color: {color};'>{status}</span>
                """, unsafe_allow_html=True)
            else:
                st.info("Start a proctoring session to see real-time stability tracking.")
            
            st.markdown("---")
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Sit comfortably and maintain a consistent position
            - Avoid frequent adjustments or fidgeting
            - Take breaks if needed, but minimize movement during the exam
            - Find a comfortable chair and desk setup before starting
            """)
    
    # RAG Insights Tab
    with tab5:
        st.subheader("🤖 RAG-Powered Session Insights")
        st.markdown("""
        **Retrieval-Augmented Generation (RAG)** provides intelligent insights about your proctoring session.
        Upload your session report to get detailed analysis and recommendations.
        """)
        
        if not st.session_state.exam_running and st.session_state.session_start_time is None:
            # Upload section
            st.markdown("### 📤 Upload Session Report")
            
            uploaded_file = st.file_uploader(
                "Upload a JSON report file from a completed session",
                type=['json'],
                help="Upload the JSON report exported from a previous proctoring session"
            )
            
            if uploaded_file is not None:
                try:
                    report_data = json.load(uploaded_file)
                    
                    # Display report summary
                    st.success("✅ Report loaded successfully!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Violations", report_data.get('total_violations', 0))
                    with col2:
                        st.metric("Risk Score", f"{report_data.get('risk_score', 0)}/100")
                    with col3:
                        st.metric("Stability Score", f"{report_data.get('movement_score', 100)}/100")
                    with col4:
                        session_duration = report_data.get('session_duration', 0)
                        hours = int(session_duration // 3600)
                        minutes = int((session_duration % 3600) // 60)
                        st.metric("Session Duration", f"{hours:02d}:{minutes:02d}")
                    
                    # Generate insights using RAG
                    if st.button("🔍 Generate Insights", type="primary", use_container_width=True):
                        insights = generate_rag_insights(report_data)
                        
                        st.markdown("---")
                        st.markdown("### 💡 Generated Insights")
                        
                        for insight in insights:
                            st.markdown(f"#### {insight['title']}")
                            st.markdown(insight['content'])
                            if 'recommendations' in insight:
                                st.markdown("**Recommendations:**")
                                for rec in insight['recommendations']:
                                    st.markdown(f"- {rec}")
                            st.markdown("---")
                
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON file. Please upload a valid session report.")
                except Exception as e:
                    st.error(f"❌ Error processing report: {e}")
        else:
            st.info("📋 Complete your current session first, then export the report to generate insights.")
            
            # Show option to analyze current session when it ends
            if st.session_state.exam_running:
                st.warning("⏸️ Stop the current session to export and analyze the report.")
            elif st.session_state.session_start_time is not None:
                summary = st.session_state.violation_tracker.get_violations_summary()
                if summary:
                    st.markdown("### 📊 Current Session Summary")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Violations", summary.get('total_violations', 0))
                        st.metric("Risk Score", f"{summary.get('risk_score', 0)}/100")
                    with col2:
                        movement_score = st.session_state.movement_tracker.get_movement_score()
                        st.metric("Stability Score", f"{movement_score}/100")
                        st.metric("Rapid Movements", st.session_state.movement_tracker.rapid_movements)
                    
                    if st.button("🔍 Generate Insights for Current Session", type="primary", use_container_width=True):
                        # Prepare report data
                        report_data = summary.copy()
                        report_data['movement_score'] = movement_score
                        report_data['rapid_movements'] = st.session_state.movement_tracker.rapid_movements
                        report_data['session_duration'] = time.time() - st.session_state.session_start_time if st.session_state.session_start_time else 0
                        report_data['session_start'] = datetime.fromtimestamp(st.session_state.session_start_time).isoformat() if st.session_state.session_start_time else None
                        report_data['session_end'] = datetime.now().isoformat()
                        report_data['metrics_history'] = st.session_state.session_metrics_history
                        
                        insights = generate_rag_insights(report_data)
                        
                        st.markdown("---")
                        st.markdown("### 💡 Generated Insights")
                        
                        for insight in insights:
                            st.markdown(f"#### {insight['title']}")
                            st.markdown(insight['content'])
                            if 'recommendations' in insight:
                                st.markdown("**Recommendations:**")
                                for rec in insight['recommendations']:
                                    st.markdown(f"- {rec}")
                            st.markdown("---")
    
    # Proctoring loop
    if st.session_state.exam_running:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera")
            st.session_state.exam_running = False
            st.stop()
        
        tracker = st.session_state.violation_tracker
        movement_tracker = st.session_state.movement_tracker
        size_tracker = st.session_state.size_tracker
        head_pose_estimator = st.session_state.head_pose_estimator
        gaze_estimator = st.session_state.gaze_estimator
        
        while st.session_state.exam_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break
            
            # Check frame validity
            if frame is None or frame.size == 0:
                st.error("Invalid frame received")
                continue
            
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # ========================================
            # YOUR FACE DETECTION
            # ========================================
            # Use full frame for detection (model will resize anyway)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = tf.image.resize(rgb, (120, 120)) / 255.0
            
            classes, coords = model(
                tf.expand_dims(resized, axis=0),
                training=False
            )
            
            confidence = float(classes[0][0])
            coords = coords[0].numpy()
            
            face_detected = confidence > 0.5
            current_warnings = []
            face_bbox = None
            yaw, pitch, roll = 0, 0, 0
            gaze_direction = 'center'
            gaze_confidence = 0.0
            
            # ========================================
            # FACE PRESENCE
            # ========================================
            if not face_detected:
                tracker.start_violation('face_absent')
                should_flag, duration, severity = tracker.check_violation('face_absent', frame)
                if duration > 1:
                    current_warnings.append(f"⚠️ Face absent: {duration:.1f}s")
            else:
                tracker.end_violation('face_absent')
                
                x1, y1 = (coords[:2] * [w, h]).astype(int)
                x2, y2 = (coords[2:] * [w, h]).astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_bbox = (x1, y1, x2, y2)
                
                # Draw bounding box
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # ========================================
                # EYE DETECTION
                # ========================================
                eye_regions = []
                if detect_gaze and detectors is not None and 'eye' in detectors and detectors['eye'] is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_roi = gray[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        eyes = detectors['eye'].detectMultiScale(face_roi, 1.1, 4)
                        for (ex, ey, ew, eh) in eyes:
                            # Adjust coordinates to full frame
                            eye_x = x1 + ex
                            eye_y = y1 + ey
                            eye_regions.append((eye_x, eye_y, ew, eh))
                            
                            # Draw eye rectangles
                            cv2.rectangle(frame, (eye_x, eye_y), (eye_x+ew, eye_y+eh), (255, 0, 255), 1)
                
                # ========================================
                # HEAD POSE ESTIMATION
                # ========================================
                yaw, pitch, roll = 0, 0, 0
                if detect_head_pose and face_bbox:
                    # Get eye centers for roll calculation
                    eye_positions = None
                    if eye_regions and len(eye_regions) >= 2:
                        eye_positions = []
                        for (ex, ey, ew, eh) in eye_regions[:2]:
                            eye_positions.append((ex + ew//2, ey + eh//2))
                    
                    yaw, pitch, roll = head_pose_estimator.estimate_from_bbox_and_eyes(
                        face_bbox, eye_positions, frame.shape
                    )
                    
                    # Visualize head pose with arrows
                    face_center_x = (x1 + x2) // 2
                    face_center_y = (y1 + y2) // 2
                    
                    # Yaw visualization (horizontal arrow)
                    yaw_length = int(yaw * 2)
                    cv2.arrowedLine(frame, (face_center_x, face_center_y), 
                                  (face_center_x + yaw_length, face_center_y), 
                                  (0, 255, 255), 2, tipLength=0.3)
                    
                    # Pitch visualization (vertical arrow)
                    pitch_length = int(pitch * 2)
                    cv2.arrowedLine(frame, (face_center_x, face_center_y), 
                                  (face_center_x, face_center_y + pitch_length), 
                                  (255, 0, 255), 2, tipLength=0.3)
                    
                    # Display head pose values
                    pose_text = f"Yaw: {yaw:.1f}° Pitch: {pitch:.1f}° Roll: {roll:.1f}°"
                    cv2.putText(frame, pose_text, (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Check for head turned violation
                    if abs(yaw) > 30 or abs(pitch) > 25:
                        tracker.start_violation('head_turned')
                        should_flag, duration, severity = tracker.check_violation('head_turned', frame)
                        if should_flag:
                            current_warnings.append(f"⚠️ Head turned: {abs(yaw):.1f}°")
                    else:
                        tracker.end_violation('head_turned')
                
                # ========================================
                # EYE GAZE ESTIMATION
                # ========================================
                gaze_direction = 'center'
                gaze_confidence = 0.0
                if detect_gaze and eye_regions:
                    gaze_direction, gaze_confidence = gaze_estimator.estimate_gaze(frame, eye_regions)
                    
                    # Display gaze direction
                    gaze_text = f"Gaze: {gaze_direction.upper()} ({gaze_confidence:.2f})"
                    cv2.putText(frame, gaze_text, (x1, y2 + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Check for eyes off screen - NEW WARNING SYSTEM
                    if gaze_direction in ['left', 'right', 'up', 'down'] and gaze_confidence > 0.6:
                        # Issue warning when looking away (not using duration-based violation)
                        current_time = time.time()
                        # Only issue a warning if enough time has passed since last warning (prevent spam)
                        if tracker.last_gaze_warning_time is None or (current_time - tracker.last_gaze_warning_time) > 1.0:
                            tracker.gaze_warnings += 1
                            tracker.last_gaze_warning_time = current_time
                            current_warnings.append(f"⚠️ Eyes looking {gaze_direction} (Warning {tracker.gaze_warnings}/3)")
                            
                            # After 3 warnings, log a violation
                            if tracker.gaze_warnings >= 3:
                                tracker.log_violation('eyes_off_screen', 0, 'MEDIUM', frame)
                                current_warnings.append(f"🚨 VIOLATION: Eyes off screen (3 warnings reached)")
                                tracker.gaze_warnings = 0  # Reset after violation
                                tracker.last_gaze_warning_time = None
                    else:
                        # Reset warnings when gaze returns to center
                        if tracker.gaze_warnings > 0:
                            tracker.gaze_warnings = 0
                            tracker.last_gaze_warning_time = None
            
            # ========================================
            # MULTIPLE FACE DETECTION
            # ========================================
            if detect_multiple and detectors is not None and 'face' in detectors and detectors['face'] is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detectors['face'].detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 1:
                    tracker.start_violation('multiple_faces')
                    tracker.check_violation('multiple_faces', frame)
                    current_warnings.append(f"🚨 {len(faces)} faces detected!")
                    
                    # Draw boxes around additional faces
                    for (x, y, w_face, h_face) in faces:
                        cv2.rectangle(frame, (x, y), (x+w_face, y+h_face), (0, 0, 255), 2)
                else:
                    tracker.end_violation('multiple_faces')
            
            # ========================================
            # MOVEMENT TRACKING
            # ========================================
            movement_score = 100  # Default value
            if detect_movement:
                movement, is_rapid = movement_tracker.update(face_bbox)
                movement_score = movement_tracker.get_movement_score()
                
                if is_rapid:
                    tracker.start_violation('rapid_movement')
                    should_flag, duration, severity = tracker.check_violation('rapid_movement', frame)
                    if should_flag:
                        current_warnings.append(f"⚠️ Rapid head movement: {duration:.1f}s")
                else:
                    tracker.end_violation('rapid_movement')
                
                # Check for rapid stability decrease
                is_stability_violation, stability_duration, prev_score, curr_score = movement_tracker.check_rapid_stability_decrease(threshold_seconds=2.5)
                if is_stability_violation:
                    tracker.start_violation('face_instability')
                    should_flag, duration, severity = tracker.check_violation('face_instability', frame)
                    if should_flag:
                        current_warnings.append(f"⚠️ Face stability decreasing - Please remain still! ({duration:.1f}s)")
                else:
                    tracker.end_violation('face_instability')
            
            # ========================================
            # SIZE/LEANING TRACKING
            # ========================================
            if detect_leaning:
                size, position = size_tracker.update(face_bbox)
                
                if position == "too_close":
                    tracker.start_violation('leaning_suspicious')
                    should_flag, duration, severity = tracker.check_violation('leaning_suspicious', frame)
                    if should_flag:
                        current_warnings.append("⚠️ Leaning too close (reading?)")
                elif position == "too_far":
                    current_warnings.append("⚠️ Too far from camera")
                else:
                    tracker.end_violation('leaning_suspicious')
            
            # ========================================
            # CAMERA INTEGRITY
            # ========================================
            is_obstructed, is_poor_lighting, blur_score, brightness = detect_camera_issues(frame)
            
            if is_obstructed:
                tracker.start_violation('camera_blocked')
                tracker.check_violation('camera_blocked', frame)
                current_warnings.append("🚨 Camera obstructed!")
            else:
                tracker.end_violation('camera_blocked')
            
            if is_poor_lighting:
                current_warnings.append(f"⚠️ Poor lighting ({int(brightness)})")
            
            # ========================================
            # OVERLAY WARNINGS
            # ========================================
            for i, warning in enumerate(current_warnings):
                cv2.putText(frame, warning, (10, h - 30 - (i * 25)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Display metrics on frame
            risk_score = tracker.get_risk_score()
            color = (0, 255, 0) if risk_score < 30 else (0, 165, 255) if risk_score < 60 else (0, 0, 255)
            cv2.putText(frame, f"Risk: {risk_score}/100", (max(10, w - 150), 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(frame, f"Stability: {movement_score}/100", (max(10, w - 180), 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Session timer
            if st.session_state.session_start_time:
                elapsed = time.time() - st.session_state.session_start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                timer_text = f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}"
                cv2.putText(frame, timer_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update session metrics history
            current_time = time.time()
            if len(st.session_state.session_metrics_history['timestamps']) == 0 or \
               current_time - st.session_state.session_metrics_history['timestamps'][-1] > 1.0:  # Update every second
                st.session_state.session_metrics_history['timestamps'].append(current_time)
                st.session_state.session_metrics_history['risk_scores'].append(risk_score)
                st.session_state.session_metrics_history['face_detected'].append(1 if face_detected else 0)
                st.session_state.session_metrics_history['head_yaw'].append(yaw)
                st.session_state.session_metrics_history['head_pitch'].append(pitch)
                st.session_state.session_metrics_history['head_roll'].append(roll)
                
                # Keep only last 300 data points (5 minutes at 1 update/sec)
                if len(st.session_state.session_metrics_history['timestamps']) > 300:
                    for key in st.session_state.session_metrics_history:
                        st.session_state.session_metrics_history[key].pop(0)
            
            # Display in Live Feed tab
            FRAME_WINDOW.image(frame_rgb, channels="RGB")
            
            # Store latest values for Help & Info tab
            st.session_state.last_yaw = yaw
            st.session_state.last_pitch = pitch
            st.session_state.last_roll = roll
            st.session_state.last_gaze_direction = gaze_direction
            st.session_state.last_gaze_confidence = gaze_confidence
            
            # Update metrics in sidebar
            metric1.metric("Face Status", "✅ Detected" if face_detected else "❌ Not Detected")
            metric2.metric("Risk Score", f"{risk_score}/100")
            metric3.metric("Stability", f"{movement_score}/100")
            metric4.metric("Violations", len(tracker.violations))
            metric5.metric("Head Yaw", f"{yaw:.1f}°")
            metric6.metric("Head Pitch", f"{pitch:.1f}°")
            metric7.metric("Gaze", gaze_direction.upper())
            
            # Show warnings
            if current_warnings:
                warning_html = "<br>".join([f"<strong>{w}</strong>" for w in current_warnings])
                warning_display.markdown(f'<div style="background: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107; margin-top: 1rem;">{warning_html}</div>', unsafe_allow_html=True)
            else:
                warning_display.markdown('<div style="background: #d4edda; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #28a745; margin-top: 1rem;"><strong>✅ No violations detected</strong></div>', unsafe_allow_html=True)
            
            time.sleep(0.033)
        
        cap.release()

if __name__ == "__main__":
    main()