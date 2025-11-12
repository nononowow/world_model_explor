import gymnasium as gym
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
# The environment version used in World Models was CarRacing-v0. We use the current v2.
# Using continuous=True for the original continuous action space.
ENV_NAME = "CarRacing-v3"
NUM_EPISODES = 1000  # Number of full episodes to collect
MAX_FRAMES_PER_EPISODE = 1000 # Max steps per episode (standard for CarRacing)
OUTPUT_DIR = "car_racing_rollouts"
SAVE_FILENAME = "dataset.npz"

# Desired output dimensions after cropping and resizing, as used in the paper (64x64)
TARGET_SIZE = (64, 64)
# Vertical crop: The bottom of the screen contains status bars which are usually cropped out
# The original World Model code cropped to frame[:84, :, :], which corresponds to
# an 84-pixel height crop from the original 96x96 image.
CROP_HEIGHT = 84 

# --- Helper Functions ---

def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Crops the bottom status bar and resizes the image to the target size (64x64x3).

    Args:
        frame: The raw (96, 96, 3) observation from the environment.

    Returns:
        The preprocessed (64, 64, 3) image as a float array in [0, 1].
    """
    # 1. Crop the bottom status bar (keeping the top 84 rows of 96)
    cropped_frame = frame[:CROP_HEIGHT, :, :]

    # 2. Resize to 64x64
    # PIL is often used for high-quality resizing in RL datasets
    img = Image.fromarray(cropped_frame, mode='RGB')
    resized_img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    
    # 3. Convert back to numpy array and normalize to [0, 1]
    obs = np.array(resized_img, dtype=np.float32) / 255.0
    return obs

def get_action(step: int) -> np.ndarray:
    """
    Generates a pseudo-random action to ensure varied data collection.
    
    CarRacing actions are: [steering (-1 to 1), gas (0 to 1), brake (0 to 1)].
    
    We use a basic policy: initial acceleration, then random steering/throttle.
    """
    # 1. Initial burst of acceleration to get the car moving
    if step < 50:
        # Action: [Steering=0.0, Gas=1.0, Brake=0.0]
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    # 2. Pseudo-random action
    # Random steering in [-1, 1]
    steering = np.random.uniform(-1.0, 1.0) 
    
    # Gas: usually high to keep moving, but add noise
    gas = np.random.uniform(0.3, 0.8) 
    
    # Brake: mostly zero, occasionally a small brake
    brake = np.random.choice([0.0, np.random.uniform(0.1, 0.3)], p=[0.9, 0.1])
    
    return np.array([steering, gas, brake], dtype=np.float32)

def collect_rollouts():
    """
    Main function to run episodes and collect the (obs, action, next_obs, done) transitions.
    """
    # Initialize environment
    print(f"Initializing environment: {ENV_NAME}")
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving dataset to: {OUTPUT_DIR}/{SAVE_FILENAME}")

    # Data lists to store all collected transitions
    observations = []
    actions = []
    rewards = []
    dones = []
    
    print(f"Starting data collection for {NUM_EPISODES} episodes...")

    for episode_num in tqdm(range(NUM_EPISODES), desc="Collecting Episodes"):
        # Reset environment for a new random track
        observation, info = env.reset(seed=episode_num)
        
        # Pre-process the initial frame
        current_frame = process_frame(observation)
        
        terminated = False
        truncated = False
        step = 0

        while not terminated and not truncated and step < MAX_FRAMES_PER_EPISODE:
            # 1. Choose action
            action = get_action(step)
            
            # 2. Take step
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            # 3. Pre-process the next frame
            next_frame = process_frame(next_observation)
            
            # 4. Record transition
            # Store the current frame (obs), the action taken, and the resulting state (next_obs)
            observations.append(current_frame)
            actions.append(action)
            rewards.append(reward)
            # Use 'terminated' as the done state for the VAE/RNN training
            dones.append(terminated or truncated)
            
            # Update for next loop iteration
            current_frame = next_frame
            step += 1
            
        # Optional: Log total steps/score for the episode
        # print(f"Episode {episode_num + 1} finished after {step} steps. Total reward: {sum(rewards[-step:])}")

    env.close()

    # --- Saving Data ---
    
    # Convert lists to NumPy arrays for efficient storage
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones, dtype=np.bool_)

    total_frames = observations.shape[0]
    
    print(f"\n--- Data Collection Complete ---")
    print(f"Total frames collected: {total_frames}")
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    
    # Save the data into a compressed NumPy archive
    save_path = os.path.join(OUTPUT_DIR, SAVE_FILENAME)
    np.savez_compressed(
        save_path, 
        obs=observations, 
        action=actions, 
        reward=rewards, 
        done=dones
    )
    
    print(f"Successfully saved all data to {save_path}")

if __name__ == "__main__":
    # Note: Requires the 'gymnasium' library and 'Pillow' (PIL) for image handling.
    # To run this script, ensure you have the required packages installed:
    # pip install gymnasium[box2d] numpy pillow tqdm
    collect_rollouts()