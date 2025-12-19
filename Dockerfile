FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install runpod

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
