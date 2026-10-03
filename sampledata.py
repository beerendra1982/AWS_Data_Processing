"""
Generate a 100 MB realistic satellite-style binary file.

- Each frame:
  - 4 bytes: frame_id (unsigned int, little-endian)
  - 4 bytes: timestamp (unsigned int, epoch seconds)
  - 8 bytes: reserved/metadata
  - N bytes: payload (random)
"""

import os
import struct
import time

OUTPUT_FILE = "sample_satellite_downlink_100MB.bin"
TARGET_SIZE_BYTES = 1 * 1024 * 1024  # 100 MB
FRAME_HEADER_SIZE = 4 + 4 + 8         # 16 bytes
FRAME_PAYLOAD_SIZE = 1024             # 1 KB payload
FRAME_SIZE = FRAME_HEADER_SIZE + FRAME_PAYLOAD_SIZE


def generate_file():
    bytes_written = 0
    frame_id = 0

    with open(OUTPUT_FILE, "wb") as f:
        while bytes_written < TARGET_SIZE_BYTES:
            timestamp = int(time.time())
            reserved = os.urandom(8)  # metadata placeholder
            header = struct.pack("<II8s", frame_id, timestamp, reserved)
            payload = os.urandom(FRAME_PAYLOAD_SIZE)

            frame = header + payload
            f.write(frame)

            bytes_written += len(frame)
            frame_id += 1

    print(f"Generated {OUTPUT_FILE} (~{bytes_written / (1024*1024):.2f} MB, {frame_id} frames)")


if __name__ == "__main__":
    generate_file()
