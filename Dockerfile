FROM python:3.10-slim

WORKDIR /workspace/scallion-lane-segmentation

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "inference_complete.py", "--help"]
