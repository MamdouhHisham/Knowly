# Use a Python base image
FROM python:3.10-slim

# Install system dependencies for dlib
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libboost-all-dev \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Command to run your app (adjust if needed)
CMD ["python", "main.py"]
