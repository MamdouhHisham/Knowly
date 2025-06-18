FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libboost-all-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .


RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
