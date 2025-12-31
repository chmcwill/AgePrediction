# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    build-essential \
    ca-certificates \
    git \
    curl \
    wget

# Clean up apt cache to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy all project files
COPY . .

# Expose port for Flask
EXPOSE 8080

# Set environment variables for Flask
ENV FLASK_APP=FlaskAgePred.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production
ENV FLASK_RUN_PORT=8080

# Gunicorn
CMD ["gunicorn", "FlaskAgePred:app", "--bind=0.0.0.0:8080", "--workers=1", "--max-requests=100", "--max-requests-jitter=10"]