import os
import deepmind_lab
import cv2
import numpy as np

# Set headless display for HPC
os.environ["QT_QPA_PLATFORM"] = "offscreen"

level_name = "ymaze_vol3_INSTR"  # or your exact level name
output_file = "spectator_map.png"

# Request the spectator top-down camera at high resolution (e.g. 1024x1024)
observations = [
    "RGB_INTERLEAVED",
    "DEBUG.CAMERA_INTERLEAVED.TOP_DOWN"
]

config = {
    "width": "1024",
    "height": "1024",
    "fps": "30",
    # If you want to change camera height/position, you can override here:
    # "camera.3": "1200"  # (camera altitude)
}

print(f"Loading level '{level_name}'...")
env = deepmind_lab.Lab(
    level=level_name,
    observations=observations,
    config=config
)

# Reset environment to generate the map & place entities
env.reset(seed=1)

# Advance 1 step so textures, cues, and lighting render
#env.step([0] * 7, num_steps=1)
action = np.zeros(7, dtype=np.intc)  # Assuming 7 discrete actions
env.step(action, num_steps=1)

# Grab the spectator observation
obs = env.observations()
if "DEBUG.CAMERA_INTERLEAVED.TOP_DOWN" in obs:
    top_down_img = obs["DEBUG.CAMERA_INTERLEAVED.TOP_DOWN"]
    # Convert RGB to BGR for OpenCV saving
    cv2.imwrite(output_file, cv2.cvtColor(top_down_img, cv2.COLOR_RGB2BGR))
    print(f"Successfully saved spectator picture to: {os.path.abspath(output_file)}")
else:
    print("DEBUG.CAMERA_INTERLEAVED.TOP_DOWN observation not found.")
    print("Available observations:", list(obs.keys()))

env.close()