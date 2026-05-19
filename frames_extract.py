import os
import cv2

video_path = "/jepa/15573192_1920_1080_50fps.mp4"
output_folder = "frames"

frame_skip = 1

os.makedirs(output_folder, exist_ok=True)


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception(f"Could not open video: {video_path}")

frame_count = 0
saved_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % frame_skip == 0:

        frame_name = os.path.join(
            output_folder,
            f"frame_{saved_count:04d}.jpg"
        )

        cv2.imwrite(frame_name, frame)

        print(f"Saved: {frame_name}")

        saved_count += 1

    frame_count += 1


cap.release()

print(f"Done. Extracted {saved_count} frames.")
