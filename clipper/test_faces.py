"""Scan the FULL video for multi-face moments at 4 FPS (250ms intervals)."""
import os, sys, cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

VIDEO = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\advai\Downloads\vidssave.com The Future Mark Zuckerberg Is Trying To Build 720P.mp4"

DETECT_FPS = 4
DETECT_INTERVAL = 1.0 / DETECT_FPS  # 0.25 seconds

# Use both models
detectors = []
for model in ["blaze_face_full_range.tflite", "blaze_face_short_range.tflite"]:
    if os.path.isfile(model):
        opts = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model),
            min_detection_confidence=0.35,
        )
        detectors.append(FaceDetector.create_from_options(opts))

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {w}x{h}, {fps:.0f}fps, {duration:.0f}s ({duration/60:.1f}min)")
print(f"Sampling at {DETECT_FPS} FPS ({DETECT_INTERVAL*1000:.0f}ms intervals)")

current_time = 0.0
frame_count = 0
multi_face_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    t_sec = frame_count / fps if fps > 0 else 0
    if t_sec >= current_time:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        fh, fw = frame.shape[:2]

        all_faces = []
        for det in detectors:
            result = det.detect(mp_img)
            for d in result.detections:
                bb = d.bounding_box
                cx = (bb.origin_x + bb.width / 2) / fw
                area = (bb.width * bb.height) / (fw * fh)
                all_faces.append((cx, area))

        # Deduplicate
        all_faces.sort(key=lambda f: f[0])
        merged = []
        for face in all_faces:
            if merged and abs(face[0] - merged[-1][0]) < 0.10:
                if face[1] > merged[-1][1]:
                    merged[-1] = face
            else:
                merged.append(face)

        if len(merged) >= 2:
            multi_face_times.append(current_time)
            mins = int(current_time) // 60
            secs = current_time - mins * 60
            faces_str = ", ".join(f"cx={f[0]:.3f} area={f[1]:.4f}" for f in merged)
            print(f"  2+ FACES at {current_time:.2f}s ({mins}m{secs:.2f}s): {faces_str}")

        current_time += DETECT_INTERVAL
        if int(current_time) % 300 == 0 and abs(current_time % 1.0) < DETECT_INTERVAL:
            print(f"  ... scanned {current_time:.1f}s / {duration:.0f}s")
    frame_count += 1

cap.release()
for d in detectors:
    d.close()

print(f"\n--- SUMMARY ---")
print(f"Total duration: {duration:.0f}s")
print(f"Intervals with 2+ faces: {len(multi_face_times)} / {int(duration * DETECT_FPS)}")
if multi_face_times:
    # Find contiguous ranges (within 0.5s gap tolerance)
    ranges = []
    start = multi_face_times[0]
    prev = start
    for t in multi_face_times[1:]:
        if t - prev > 0.5:  # Gap > 0.5s means new range
            ranges.append((start, prev))
            start = t
        prev = t
    ranges.append((start, prev))
    print(f"Multi-face ranges ({len(ranges)} total):")
    for s, e in ranges:
        s_min, s_sec = int(s) // 60, s % 60
        e_min, e_sec = int(e) // 60, e % 60
        dur = e - s + DETECT_INTERVAL
        print(f"  {s_min}m{s_sec:05.2f}s - {e_min}m{e_sec:05.2f}s  ({dur:.2f}s)")
