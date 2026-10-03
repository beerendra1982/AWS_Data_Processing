import os
import struct
import boto3
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SOURCE_BUCKET = "my-source-bucket-beeren"          # Input bucket
SOURCE_KEY    = "raw/sample_satellite_downlink_100MB.bin"          # Binary file path

DEST_BUCKET   = "my-processed-bucket-beeren"       # Output bucket
DEST_PREFIX   = "processed/"                # Folder in destination bucket

LOCAL_RAW_PATH = "/tmp/downlink.bin"
LOCAL_OUT_DIR  = "/tmp/processed"
os.makedirs(LOCAL_OUT_DIR, exist_ok=True)

# Binary frame structure (adjust if needed)
FRAME_HEADER_SIZE  = 16                     # 4 bytes frame_id, 4 bytes timestamp, 8 bytes reserved
FRAME_PAYLOAD_SIZE = 1024
FRAME_SIZE         = FRAME_HEADER_SIZE + FRAME_PAYLOAD_SIZE

# Output image size
IMG_SIZE = 32


# ---------------------------------------------------------
# STEP 1 — DOWNLOAD BINARY FILE FROM S3
# ---------------------------------------------------------
s3 = boto3.client("s3")

print(f"Downloading s3://{SOURCE_BUCKET}/{SOURCE_KEY} ...")
s3.download_file(SOURCE_BUCKET, SOURCE_KEY, LOCAL_RAW_PATH)
print("Download complete.")


# ---------------------------------------------------------
# STEP 2 — PARSE BINARY FILE INTO FRAMES
# ---------------------------------------------------------
frames = []

with open(LOCAL_RAW_PATH, "rb") as f:
    while True:
        chunk = f.read(FRAME_SIZE)
        if not chunk or len(chunk) < FRAME_SIZE:
            break

        header = chunk[:FRAME_HEADER_SIZE]
        payload = chunk[FRAME_HEADER_SIZE:]

        frame_id, timestamp, reserved = struct.unpack("<II8s", header)

        frames.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "payload": payload
        })

print(f"Parsed {len(frames)} frames.")


# ---------------------------------------------------------
# STEP 3 — CONVERT PAYLOADS INTO 32×32 IMAGES
# ---------------------------------------------------------
image_paths = []

for i, frame in enumerate(frames):
    arr = np.frombuffer(frame["payload"], dtype=np.uint8)

    if arr.size < IMG_SIZE * IMG_SIZE:
        continue

    img = arr[:IMG_SIZE * IMG_SIZE].reshape(IMG_SIZE, IMG_SIZE)

    img_path = f"{LOCAL_OUT_DIR}/img_{i}.png"
    Image.fromarray(img).save(img_path)
    image_paths.append(img_path)

print(f"Generated {len(image_paths)} images.")


# ---------------------------------------------------------
# STEP 4 — UPLOAD PROCESSED IMAGES TO DESTINATION S3 BUCKET
# ---------------------------------------------------------
for img_path in image_paths:
    filename = os.path.basename(img_path)
    key = f"{DEST_PREFIX}{filename}"

    s3.upload_file(img_path, DEST_BUCKET, key)

print(f"Uploaded {len(image_paths)} processed images to s3://{DEST_BUCKET}/{DEST_PREFIX}")
