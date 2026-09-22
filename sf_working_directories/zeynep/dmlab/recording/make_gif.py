import cv2
from PIL import Image

input_video = "12_slower_clip.mp4"
output_gif = "block_switch.gif"

# Open the cropped video
cap = cv2.VideoCapture(input_video)
frames = []

# Optional: resize if the GIF file is too large (1.0 = original size, 0.7 = 70% size)
SCALE = 1.0

# Optional: skip frames if you don't need all of them (1 = every frame, 2 = every 2nd frame)
STEP = 1  

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % STEP == 0:
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        if SCALE != 1.0:
            w, h = pil_img.size
            pil_img = pil_img.resize((int(w * SCALE), int(h * SCALE)), Image.Resampling.LANCZOS)
            
        frames.append(pil_img)
        
    frame_idx += 1

cap.release()
print(f"Loaded {len(frames)} frames. Generating GIF...")

# duration=100 means 100ms per frame (10 FPS = 3x slower playback)
# loop=0 means loop forever
frames[0].save(
    output_gif,
    save_all=True,
    append_images=frames[1:],
    duration=100,      # 100ms = 10 FPS (change to 80ms for ~12 FPS, 120ms for ~8 FPS)
    loop=0,
    optimize=True
)

print(f"Done! Saved: {output_gif}")