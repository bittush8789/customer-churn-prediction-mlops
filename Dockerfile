FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Pre-train the ML models so the API can start serving instantly
RUN python ml/train.py

# Expose port 5000
EXPOSE 5000

# Set environment variable for Python path
ENV PYTHONPATH=/app

# Start the Flask web application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "backend.app:app"]
