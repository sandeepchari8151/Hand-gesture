# 🤖 Gesture Recognition Flask App

This project is a **web-based hand gesture recognition system** built with Flask that lets you:
- Collect your own gesture data using a webcam
- Train a machine learning model on those gestures
- Run real-time recognition directly from the browser

Under the hood it uses **MediaPipe** for hand landmark detection and **scikit-learn (SVM)** for classification, wrapped in an easy-to-use dashboard so non-ML users can record, train, and test gestures without touching code.

## ✨ Features

- **User Authentication**: Secure login/registration system with password hashing
- **Data Collection**: Real-time hand landmark collection for custom gestures
- **Machine Learning**: SVM-based gesture classification with scikit-learn
- **Real-time Recognition**: Live gesture recognition with confidence scoring
- **Modern UI**: Clean, responsive web interface with CSS animations
- **Error Handling**: Comprehensive error handling and input validation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam/camera access
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd gesture_flask_app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   - Open your browser and go to `http://localhost:5000`
   - Default login: `admin` / `admin123`

## 📖 Usage Guide

### 1. Authentication
- **Login**: Use existing credentials or register a new account
- **Registration**: Create a new user account with username and password

### 2. Data Collection
1. Navigate to the dashboard after login
2. Enter a gesture label (e.g., "peace", "thumbs_up", "ok")
3. Click "Collect Data"
4. Position your hand in front of the camera
5. The system will collect 100 samples automatically
6. Press 'q' to quit early if needed

### 3. Model Training
1. Collect data for at least 2 different gestures
2. Click "Train Model" on the dashboard
3. The system will train an SVM classifier
4. View training accuracy and classification report

### 4. Gesture Recognition
1. Ensure you have a trained model
2. Click "Start Recognition"
3. Perform gestures in front of the camera
4. View real-time predictions with confidence scores
5. Press 'q' to exit recognition mode

## 🔧 Configuration

### Environment Variables (Optional)

Create a `.env` file in the project root to customize settings:

```env
# Flask Configuration
FLASK_SECRET_KEY=your-secret-key-here
FLASK_DEBUG=True

# Security
PASSWORD_SALT=your-random-salt-here

# Default User (change these!)
DEFAULT_USERNAME=admin
DEFAULT_PASSWORD=admin123

# Model Configuration
DEFAULT_SAMPLES_PER_GESTURE=100
MODEL_TEST_SIZE=0.2
```

### File Structure

```
gesture_flask_app/
├── app.py                 # Main Flask application
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── dataset.csv          # Training data (auto-generated)
├── gesture_model.pkl    # Trained model (auto-generated)
├── label_encoder.pkl    # Label encoder (auto-generated)
├── users.db            # User database (auto-generated)
├── .env.example        # Environment variables template
├── static/
│   └── style.css       # CSS styling
└── templates/
    ├── dashboard.html   # Main dashboard
    ├── login.html      # Login page
    └── register.html   # Registration page
```

## 🎯 Gesture Data Format

The system uses MediaPipe hand landmarks for gesture recognition:

- **21 hand landmarks** per gesture
- **3 coordinates** (x, y, z) per landmark
- **Total**: 63 features per gesture sample
- **Data format**: CSV with label + 63 coordinate values

## 🛠️ Technical Details

### Machine Learning Pipeline

1. **Data Collection**: MediaPipe extracts hand landmarks
2. **Preprocessing**: Normalize and validate landmark data
3. **Training**: SVM classifier with RBF kernel
4. **Evaluation**: Accuracy metrics and classification reports
5. **Prediction**: Real-time inference with confidence scoring

### Security Features

- **Password Hashing**: PBKDF2 with SHA-256
- **Input Validation**: Sanitized user inputs
- **Session Management**: Secure Flask sessions
- **Error Handling**: Graceful error recovery

## 🐛 Troubleshooting

### Common Issues

**Camera not detected**
- Ensure camera is connected and not used by other applications
- Check camera permissions in your OS settings

**Low recognition accuracy**
- Collect more training data (aim for 100+ samples per gesture)
- Ensure consistent hand positioning during data collection
- Train with diverse lighting conditions

**Model training fails**
- Ensure you have data for at least 2 different gestures
- Check that dataset.csv contains valid data
- Verify all dependencies are installed correctly

### Error Messages

- `"Camera not available"`: Check camera connection
- `"Model not found"`: Train a model first
- `"Need at least 2 different gesture labels"`: Collect more gesture types
- `"Invalid data format"`: Ensure proper landmark data structure

## 🔄 Development

### Adding New Features

1. **New Gesture Types**: Simply collect data with new labels
2. **UI Improvements**: Modify templates in `/templates/`
3. **Styling**: Update CSS in `/static/style.css`
4. **Backend Logic**: Extend functions in `app.py`

### Dependencies

- **Flask**: Web framework
- **MediaPipe**: Hand landmark detection
- **OpenCV**: Camera interface
- **scikit-learn**: Machine learning
- **pandas**: Data manipulation
- **numpy**: Numerical operations

## 📊 Performance Tips

1. **Lighting**: Use good, consistent lighting for better landmark detection
2. **Distance**: Maintain 30-60cm distance from camera
3. **Background**: Use plain backgrounds for better hand detection
4. **Training Data**: Collect diverse samples for robust recognition

## 🔒 Security Notes

- Change default credentials in production
- Use environment variables for sensitive configuration
- Implement HTTPS in production environments
- Regular security updates for dependencies

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

**Happy Gesturing!** 🎉
