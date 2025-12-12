# Use a slim Python base image
FROM python:3.11-slim-bullseye

# Set environment variables to prevent Python buffering and warnings
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies needed for Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        ffmpeg \
        libsndfile1 \
        libffi-dev \
        libssl-dev \
        && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Copy your requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of your app code
COPY . .

# Expose Flask default port
EXPOSE 5000

# Command to run your Flask app
CMD ["python", "app.py"]
