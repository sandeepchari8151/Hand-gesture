#!/usr/bin/env python3
"""
Setup script for Gesture Recognition Flask App
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            shutil.copy(".env.example", ".env")
            print("✅ Created .env file from template")
            print("⚠️  Please update .env file with your own values")
        else:
            print("⚠️  No .env.example found")
    else:
        print("✅ .env file already exists")

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is available")
            cap.release()
            return True
        else:
            print("⚠️  Camera not detected")
            return False
    except ImportError:
        print("⚠️  OpenCV not installed yet")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Gesture Recognition Flask App")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create environment file
    create_env_file()
    
    # Check camera
    check_camera()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Update .env file with your configuration (optional)")
    print("2. Run: python app.py")
    print("3. Open browser: http://localhost:5000")
    print("4. Login with: admin / admin123")
    print("\n📖 See README.md for detailed usage instructions")

if __name__ == "__main__":
    main()
