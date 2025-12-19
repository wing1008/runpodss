FROM runpod/base:0.6.0-cuda11.8

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install runpod

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
