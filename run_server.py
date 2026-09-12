import sys
import os

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.app import app
from waitress import serve

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Diabetic Retinopathy Application on http://127.0.0.1:{port}")
    serve(app, host="0.0.0.0", port=port, threads=4)
