import uvicorn
import os
import sys

# Ensure src/ is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

if __name__ == "__main__":
    print("Starting production server at http://localhost:5000")
    uvicorn.run("src.api.api:app", host="0.0.0.0", port=5000, reload=True)
