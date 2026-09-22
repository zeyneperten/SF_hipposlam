import cv2

cap = cv2.VideoCapture("cropped.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set target slower fps (e.g., 10 or 12 fps)
out = cv2.VideoWriter("slower_clip.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 12, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()
print("Saved slower_clip.mp4!")