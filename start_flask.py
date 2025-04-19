import subprocess
import time
import requests

FLASK_URL = "http://127.0.0.1:5000"

def is_flask_running():
    """Check if Flask is already running"""
    try:
        response = requests.get(FLASK_URL, timeout=3)
        if response.status_code == 200:
            print("✅ Flask is already running!")
            return True
    except requests.exceptions.RequestException:
        print("⚠️ Flask is not running.")
    return False

def start_flask():
    """Start Flask if not running"""
    if not is_flask_running():
        print("🔹 Starting Flask server...")
        process = subprocess.Popen(
            ["python", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(5)  # Wait for Flask to start
        if is_flask_running():
            print("✅ Flask started successfully!")
        else:
            print("❌ Failed to start Flask.")

if __name__ == "__main__":
    start_flask()
