#!/usr/bin/env python3
"""Fixed version of app.py with delayed imports"""

print("Starting Gesture Recognition Flask App")

# Import only essential modules first
import os
import re
import hashlib
from flask import Flask, render_template, request, redirect, url_for, session, flash

print("Basic imports successful")

# Import config
try:
    from config import Config
    print("Config imported")
except Exception as e:
    print(f"Config import failed: {e}")
    exit(1)

# Create Flask app
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.config.from_object(Config)
print("Flask app created")

# Utility functions
def hash_password(password: str, salt=None) -> str:
    """Hash password with PBKDF2"""
    if salt is None:
        salt = Config.PASSWORD_SALT
    pwd = password.encode("utf-8")
    salt_b = salt.encode("utf-8")
    hashed = hashlib.pbkdf2_hmac("sha256", pwd, salt_b, 100_000)
    return hashed.hex()

def validate_input(data, field_name, min_length=1, max_length=50):
    """Validate and sanitize input data"""
    if not data:
        raise ValueError(f"{field_name} is required")
    
    data = str(data).strip()
    
    if len(data) < min_length or len(data) > max_length:
        raise ValueError(f"{field_name} must be between {min_length} and {max_length} characters")
    
    if not re.match(r'^[a-zA-Z0-9_@.-]+$', data):
        raise ValueError(f"{field_name} contains invalid characters")
    
    return data

# User management
def load_users():
    """Load all users from users.db"""
    users = {}
    try:
        if os.path.exists(Config.USERS_DB):
            with open(Config.USERS_DB, "r", encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(":")
                if len(parts) == 2:
                            username, password_hash = parts
                            users[username] = password_hash
    except Exception as e:
        print(f"Error loading users: {e}")
    return users

def save_user(username, hashed_pwd):
    """Append new user to users.db"""
    try:
        with open(Config.USERS_DB, "a", encoding='utf-8') as f:
            f.write(f"{username}:{hashed_pwd}\n")
    except Exception as e:
        print(f"Error saving user: {e}")
        raise

# Create default user if not exists
if not os.path.exists(Config.USERS_DB):
    try:
        with open(Config.USERS_DB, "w", encoding='utf-8') as f:
            f.write(f"{Config.DEFAULT_USERNAME}:{hash_password(Config.DEFAULT_PASSWORD)}\n")
        print(f"Default login created — Username: {Config.DEFAULT_USERNAME}")
    except Exception as e:
        print(f"Error creating default user: {e}")

# Delayed imports for ML and CV functions
def collect_data_for_label(label_name, num_samples=None):
    """Capture hand landmarks for the given label and save to dataset.csv"""
    try:
        # Import CV libraries only when needed
        import cv2
        import pandas as pd
        
        # Try MediaPipe first, fallback to OpenCV alternative
        try:
            import mediapipe as mp
            use_mediapipe = True
        except ImportError:
            print("MediaPipe not available, using OpenCV alternative...")
            use_mediapipe = False
        
        if num_samples is None:
            num_samples = Config.DEFAULT_SAMPLES_PER_GESTURE
        
        # Validate label name
        label_name = validate_input(label_name, "Gesture label", 1, 20)
        
        collected_data = []
        
        if use_mediapipe:
            # Use MediaPipe (original implementation)
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
            mp_drawing = mp.solutions.drawing_utils
        else:
            # Use OpenCV alternative
            hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
            if hand_cascade.empty():
                print("Warning: Hand cascade not found. Using basic contour detection.")
                hand_cascade = None

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Camera not available. Please check camera connection. If you have the web-based camera open, close it first.")
        
        print(f"\nCollecting {num_samples} samples for label: '{label_name}'")
        if not use_mediapipe:
            print("Using OpenCV hand detection (simplified landmarks)")

        count = 0
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break

            if use_mediapipe:
                # MediaPipe implementation
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        collected_data.append([label_name] + landmarks)
                        count += 1

                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        cv2.putText(frame, f"Collected: {count}/{num_samples}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No hand detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # OpenCV alternative implementation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if hand_cascade and not hand_cascade.empty():
                    hands_detected = hand_cascade.detectMultiScale(gray, 1.1, 5)
                else:
                    # Fallback: Use contour detection
                    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    hands_detected = []
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 5000:
                            x, y, w, h = cv2.boundingRect(contour)
                            hands_detected.append((x, y, w, h))
                
                if len(hands_detected) > 0:
                    # Create simplified landmarks for the first detected hand
                    x, y, w, h = hands_detected[0]
                    landmarks = []
                    
                    # Create 21 simplified hand landmarks
                    for i in range(21):
                        if i < 5:  # Fingertips
                            px = x + (i * w // 4)
                            py = y
                        elif i < 10:  # Finger joints
                            px = x + ((i-5) * w // 4)
                            py = y + h // 3
                        elif i < 15:  # Middle joints
                            px = x + ((i-10) * w // 4)
                            py = y + 2 * h // 3
                        else:  # Palm points
                            px = x + ((i-15) * w // 4)
                            py = y + h
                        
                        # Normalize coordinates
                        norm_x = px / frame.shape[1]
                        norm_y = py / frame.shape[0]
                        norm_z = 0.0
                        landmarks.extend([norm_x, norm_y, norm_z])
                    
                    collected_data.append([label_name] + landmarks)
                    count += 1
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Collected: {count}/{num_samples}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No hand detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Collecting Gesture Data - Press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        if use_mediapipe:
            hands.close()

        if collected_data:
            # Save or append data to dataset.csv
            df_new = pd.DataFrame(collected_data)
            if os.path.exists(Config.DATASET_PATH):
                df_existing = pd.read_csv(Config.DATASET_PATH, header=None)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(Config.DATASET_PATH, index=False, header=False)
            else:
                df_new.to_csv(Config.DATASET_PATH, index=False, header=False)
            
            print(f"Data collection completed for '{label_name}'. Collected {len(collected_data)} samples.")
            return f"Successfully collected {len(collected_data)} samples for '{label_name}'"
        else:
            return "No data collected. Make sure your hand is visible in the camera."
            
    except Exception as e:
        print(f"Error during data collection: {e}")
        return f"Error during data collection: {str(e)}"

def train_gesture_model():
    """Train gesture recognition model with error handling"""
    try:
        # Import ML libraries only when needed
        import pandas as pd
        import numpy as np
        import pickle
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report
        
        if not os.path.exists(Config.DATASET_PATH):
            return "No dataset found. Please collect gesture data first."
        
        # Load dataset
        df = pd.read_csv(Config.DATASET_PATH, header=None)
        
        if df.empty:
            return "Dataset is empty. Please collect some gesture data first."
        
        X = df.iloc[:, 1:].values        # landmarks
        y = df.iloc[:, 0].values         # labels
        
        # Validate data shape
        if X.shape[1] != 63:  # 21 landmarks * 3 coordinates
            return f"Invalid data format. Expected 63 features, got {X.shape[1]}"
        
        unique_labels = set(y)
        if len(unique_labels) < 2:
            return f"Need at least 2 different gesture labels to train. Found: {list(unique_labels)}"
        
        print(f"Training model with {len(y)} samples and {len(unique_labels)} classes: {list(unique_labels)}")
        
        # Encode labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=Config.MODEL_TEST_SIZE, random_state=42, stratify=y_enc
        )
        
        # Train SVM with better parameters
        clf = SVC(probability=True, kernel="rbf", C=5, gamma="scale", random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"Model trained. Accuracy: {acc:.2f}")
        print("Classification Report:")
        print(classification_report(y_test, preds, target_names=le.classes_))
        
        # Save model and label encoder
        with open(Config.MODEL_FILE, "wb") as f:
            pickle.dump(clf, f)
        with open(Config.LABEL_ENCODER_FILE, "wb") as f:
            pickle.dump(le, f)
        
        return f"Model trained successfully!\nAccuracy: {acc:.2f}\nClasses: {list(le.classes_)}\nTotal samples: {len(y)}"
        
    except Exception as e:
        error_msg = f"Error training model: {str(e)}"
        print(error_msg)
        return error_msg

def recognize_gesture():
    """Run real-time gesture recognition using the trained SVM model."""
    try:
        # Import CV and ML libraries only when needed
        import cv2
        import numpy as np
        import pickle
        
        # Try MediaPipe first, fallback to OpenCV alternative
        try:
            import mediapipe as mp
            use_mediapipe = True
        except ImportError:
            print("MediaPipe not available, using OpenCV alternative...")
            use_mediapipe = False
        
        if not (os.path.exists(Config.MODEL_FILE) and os.path.exists(Config.LABEL_ENCODER_FILE)):
            return "Model not found. Please train the model first."

        # Load model and label encoder
        with open(Config.MODEL_FILE, "rb") as f:
            clf = pickle.load(f)
        with open(Config.LABEL_ENCODER_FILE, "rb") as f:
            le = pickle.load(f)

        # Start webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Camera not available. Please check camera connection. If you have the web-based camera open, close it first."
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1)
        mp_drawing = mp.solutions.drawing_utils

        print("Starting gesture recognition. Press 'q' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            text = "No hand detected"
            confidence = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    if len(landmarks) == 63:  # Validate landmark count
                        X_input = np.array(landmarks).reshape(1, -1)
                        probs = clf.predict_proba(X_input)[0]
                        idx = probs.argmax()
                        confidence = probs[idx]
                        
                        if confidence > 0.6:  # Confidence threshold
                            gesture = le.inverse_transform([idx])[0]
                            text = f"{gesture} ({confidence:.2f})"
                        else:
                            text = f"Low confidence ({confidence:.2f})"
                    
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Color code based on confidence
                color = (0, 255, 0) if confidence > 0.6 else (0, 0, 255) if confidence > 0.3 else (255, 0, 0)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow("Gesture Recognition - Press 'q' to exit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        return "Recognition finished successfully."
        
    except Exception as e:
        error_msg = f"Error during recognition: {str(e)}"
        print(error_msg)
        return error_msg

# Routes
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    try:
        uname = validate_input(request.form.get('username'), "Username", 3, 20)
        pwd = request.form.get('password')
        
        if not pwd or len(pwd) < 3:
            return redirect(url_for('home', error="Password must be at least 3 characters long"))
        
        users = load_users()
        if users.get(uname) == hash_password(pwd):
            session['user'] = uname
            return redirect(url_for('dashboard'))
        else:
            return redirect(url_for('home', error="Invalid username or password"))
    except ValueError as e:
        return redirect(url_for('home', error=str(e)))
    except Exception as e:
        return redirect(url_for('home', error="An error occurred during login"))

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/register_user', methods=['POST'])
def register_user():
    try:
        uname = validate_input(request.form.get('username'), "Username", 3, 20)
        pwd = request.form.get('password')
        confirm_pwd = request.form.get('confirmPassword')
        
        # Enhanced password validation
        if not pwd or len(pwd) < 6:
            return redirect(url_for('register_page', error="Password must be at least 6 characters long"))
        
        if pwd != confirm_pwd:
            return redirect(url_for('register_page', error="Passwords do not match"))
        
        # Check password strength
        has_upper = any(c.isupper() for c in pwd)
        has_lower = any(c.islower() for c in pwd)
        has_digit = any(c.isdigit() for c in pwd)
        
        if not (has_upper and has_lower and has_digit):
            return redirect(url_for('register_page', error="Password must contain at least one uppercase letter, one lowercase letter, and one number"))
        
        
        users = load_users()
        if uname in users:
            return redirect(url_for('register_page', error="Username already exists"))
            
        save_user(uname, hash_password(pwd))
        return redirect(url_for('home', success="Registration successful! Please login with your new account."))
        
    except ValueError as e:
        return redirect(url_for('register_page', error=str(e)))
    except Exception as e:
        return redirect(url_for('register_page', error="An error occurred during registration"))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')
    return render_template('dashboard.html', user=session['user'])

@app.route('/webcam')
def webcam():
    if 'user' not in session:
        return redirect('/')
    return render_template('webcam.html', user=session['user'])


@app.route('/collect', methods=['POST'])
def collect():
    try:
        label = validate_input(request.form.get('label'), "Gesture label", 1, 20)
        result = collect_data_for_label(label)
        flash(result)
        return redirect('/dashboard')
    except ValueError as e:
        flash(str(e))
        return redirect('/dashboard')
    except Exception as e:
        flash(f"Error during data collection: {str(e)}")
    return redirect('/dashboard')

@app.route("/train", methods=["POST"])
def train():
    try:
        result = train_gesture_model()
        flash(result)
        return redirect('/dashboard')
    except Exception as e:
        flash(f"Error during training: {str(e)}")
        return redirect('/dashboard')

@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        result = recognize_gesture()
        flash(result)
        return redirect('/dashboard')
    except Exception as e:
        flash(f"Error during recognition: {str(e)}")
        return redirect('/dashboard')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home', success="Logged out successfully"))

if __name__ == "__main__":
    print(f"\nDataset: {Config.DATASET_PATH}")
    print(f"Default user: {Config.DEFAULT_USERNAME}")
    print(f"Debug mode: {Config.DEBUG}")
    print("\nAvailable endpoints:")
    print("  - / (login page)")
    print("  - /register (registration)")
    print("  - /dashboard (main interface)")
    print("  - /logout (logout)")
    print("\nUsage:")
    print("  1. Register/Login")
    print("  2. Collect gesture data")
    print("  3. Train model")
    print("  4. Start recognition")
    print("\n" + "="*50)
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)
