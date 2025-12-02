import os

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed, using default values")

class Config:
    """Application configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'supersecretkey-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Security Configuration
    PASSWORD_SALT = os.getenv('PASSWORD_SALT', 'fixed_salt')
    
    # Default User Configuration
    DEFAULT_USERNAME = os.getenv('DEFAULT_USERNAME', 'admin')
    DEFAULT_PASSWORD = os.getenv('DEFAULT_PASSWORD', 'admin123')
    
    # Model Configuration
    DEFAULT_SAMPLES_PER_GESTURE = int(os.getenv('DEFAULT_SAMPLES_PER_GESTURE', '100'))
    MODEL_TEST_SIZE = float(os.getenv('MODEL_TEST_SIZE', '0.2'))
    
    # File Paths
    USERS_DB = "users.db"
    DATASET_PATH = "dataset.csv"
    MODEL_FILE = "gesture_model.pkl"
    LABEL_ENCODER_FILE = "label_encoder.pkl"
