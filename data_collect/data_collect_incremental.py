"""
Incremental data collection for CarRacing environment.

This module implements incremental writing for both NPZ and HDF5 formats,
allowing data collection for large datasets without running out of memory.
Data is written to disk as it's collected, rather than accumulating in RAM.

Usage:
    python data_collect_incremental.py --format npz
    python data_collect_incremental.py --format hdf5
    python data_collect_incremental.py --format both
"""

import argparse
import gymnasium as gym
import numpy as np
import os
import h5py
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
# The environment version used in World Models was CarRacing-v0. We use the current v2.
# Using continuous=True for the original continuous action space.
ENV_NAME = "CarRacing-v3"
NUM_EPISODES = 1000  # Number of full episodes to collect
MAX_FRAMES_PER_EPISODE = 1000  # Max steps per episode (standard for CarRacing)
OUTPUT_DIR = "car_racing_rollouts"
SAVE_FILENAME_NPZ = "dataset.npz"
SAVE_FILENAME_HDF5 = "dataset.h5"

# Desired output dimensions after cropping and resizing, as used in the paper (64x64)
TARGET_SIZE = (64, 64)
# Vertical crop: The bottom of the screen contains status bars which are usually cropped out
# The original World Model code cropped to frame[:84, :, :], which corresponds to
# an 84-pixel height crop from the original 96x96 image.
CROP_HEIGHT = 84

# Estimate maximum size for pre-allocation (can be adjusted)
MAX_TOTAL_FRAMES = NUM_EPISODES * MAX_FRAMES_PER_EPISODE

# Flush interval for periodic disk writes
FLUSH_INTERVAL = 1000  # Flush every N frames

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

# --- NPZ Implementation (Memory-Mapped Arrays) ---

def collect_rollouts_npz():
    """
    Collect rollouts and write incrementally to NPZ format using memory-mapped arrays.
    This avoids loading all data into RAM by writing directly to disk.
    """
    # Initialize environment
    print(f"Initializing environment: {ENV_NAME}")
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving NPZ dataset to: {OUTPUT_DIR}/{SAVE_FILENAME_NPZ}")
    
    # Create temporary .npy files for memory-mapped arrays in the data_collect directory
    temp_dir = os.path.dirname(os.path.abspath(__file__))
    temp_obs_path = os.path.join(temp_dir, "temp_obs.npy")
    temp_action_path = os.path.join(temp_dir, "temp_action.npy")
    temp_reward_path = os.path.join(temp_dir, "temp_reward.npy")
    temp_done_path = os.path.join(temp_dir, "temp_done.npy")
    
    # Pre-allocate memory-mapped arrays
    print(f"Pre-allocating memory-mapped arrays (max {MAX_TOTAL_FRAMES} frames)...")
    print("This allows incremental writing without loading all data into RAM.")
    
    obs_mmap = None
    action_mmap = None
    reward_mmap = None
    done_mmap = None
    
    try:
        obs_mmap = np.memmap(
            temp_obs_path,
            dtype=np.float32,
            mode='w+',
            shape=(MAX_TOTAL_FRAMES, TARGET_SIZE[0], TARGET_SIZE[1], 3)
        )
        action_mmap = np.memmap(
            temp_action_path,
            dtype=np.float32,
            mode='w+',
            shape=(MAX_TOTAL_FRAMES, 3)
        )
        reward_mmap = np.memmap(
            temp_reward_path,
            dtype=np.float32,
            mode='w+',
            shape=(MAX_TOTAL_FRAMES,)
        )
        done_mmap = np.memmap(
            temp_done_path,
            dtype=np.bool_,
            mode='w+',
            shape=(MAX_TOTAL_FRAMES,)
        )
        
        # Counter for current position in arrays
        current_idx = 0
        
        print(f"Starting data collection for {NUM_EPISODES} episodes...")
        
        for episode_num in tqdm(range(NUM_EPISODES), desc="Collecting Episodes (NPZ)"):
            # Reset environment for a new random track
            observation, info = env.reset(seed=episode_num)
            
            # Pre-process the initial frame
            current_frame = process_frame(observation)
            
            terminated = False
            truncated = False
            step = 0

            while not terminated and not truncated and step < MAX_FRAMES_PER_EPISODE:
                # Check if we've exceeded pre-allocated size
                if current_idx >= MAX_TOTAL_FRAMES:
                    print(f"\nWarning: Reached maximum pre-allocated size at frame {current_idx}")
                    break
                
                # 1. Choose action
                action = get_action(step)
                
                # 2. Take step
                next_observation, reward, terminated, truncated, info = env.step(action)
                
                # 3. Pre-process the next frame
                next_frame = process_frame(next_observation)
                
                # 4. Write directly to memory-mapped arrays (incremental write)
                obs_mmap[current_idx] = current_frame
                action_mmap[current_idx] = action
                reward_mmap[current_idx] = reward
                done_mmap[current_idx] = terminated or truncated
                
                # Flush periodically to ensure data is written to disk
                # This prevents data loss if the process crashes
                if current_idx % FLUSH_INTERVAL == 0:
                    obs_mmap.flush()
                    action_mmap.flush()
                    reward_mmap.flush()
                    done_mmap.flush()
                
                # Update for next loop iteration
                current_frame = next_frame
                current_idx += 1
                step += 1
        
        # Flush remaining data to disk
        obs_mmap.flush()
        action_mmap.flush()
        reward_mmap.flush()
        done_mmap.flush()
        
        # Truncate arrays to actual size collected
        print(f"\nTruncating arrays to actual size: {current_idx} frames")
        obs_mmap = obs_mmap[:current_idx]
        action_mmap = action_mmap[:current_idx]
        reward_mmap = reward_mmap[:current_idx]
        done_mmap = done_mmap[:current_idx]
        
        # Save to NPZ format
        save_path = os.path.join(OUTPUT_DIR, SAVE_FILENAME_NPZ)
        print(f"Saving to {save_path}...")
        np.savez_compressed(
            save_path,
            obs=obs_mmap,
            action=action_mmap,
            reward=reward_mmap,
            done=done_mmap
        )
        
        print(f"\n--- NPZ Data Collection Complete ---")
        print(f"Total frames collected: {current_idx}")
        print(f"Observations shape: {obs_mmap.shape}")
        print(f"Actions shape: {action_mmap.shape}")
        print(f"Successfully saved all data to {save_path}")
        
    except Exception as e:
        print(f"\nError during NPZ collection: {e}")
        raise
    
    finally:
        env.close()
        
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for temp_file in [temp_obs_path, temp_action_path, temp_reward_path, temp_done_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Warning: Could not remove {temp_file}: {e}")
        
        # Explicitly delete memory-mapped arrays to close files
        if obs_mmap is not None:
            del obs_mmap
        if action_mmap is not None:
            del action_mmap
        if reward_mmap is not None:
            del reward_mmap
        if done_mmap is not None:
            del done_mmap

# --- HDF5 Implementation (Native Incremental) ---

def collect_rollouts_hdf5():
    """
    Collect rollouts and write incrementally to HDF5 format.
    HDF5 natively supports incremental writes with chunking and compression.
    """
    # Initialize environment
    print(f"Initializing environment: {ENV_NAME}")
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, SAVE_FILENAME_HDF5)
    print(f"Saving HDF5 dataset to: {save_path}")
    
    # Counter for current position
    current_idx = 0
    
    print(f"Starting data collection for {NUM_EPISODES} episodes...")
    print("Using HDF5 native incremental writing with chunking and compression.")
    
    try:
        # Create HDF5 file with resizable datasets
        with h5py.File(save_path, 'w') as f:
            # Create datasets with chunking for efficient incremental writes
            # maxshape=(None, ...) allows dynamic resizing
            obs_dset = f.create_dataset(
                'obs',
                shape=(MAX_TOTAL_FRAMES, TARGET_SIZE[0], TARGET_SIZE[1], 3),
                maxshape=(None, TARGET_SIZE[0], TARGET_SIZE[1], 3),
                chunks=(100, TARGET_SIZE[0], TARGET_SIZE[1], 3),  # Chunk size for efficient I/O
                dtype=np.float32,
                compression='gzip',  # Compression for efficient storage
                compression_opts=4  # Compression level (0-9, 4 is a good balance)
            )
            action_dset = f.create_dataset(
                'action',
                shape=(MAX_TOTAL_FRAMES, 3),
                maxshape=(None, 3),
                chunks=(1000, 3),
                dtype=np.float32,
                compression='gzip',
                compression_opts=4
            )
            reward_dset = f.create_dataset(
                'reward',
                shape=(MAX_TOTAL_FRAMES,),
                maxshape=(None,),
                chunks=(10000,),
                dtype=np.float32,
                compression='gzip',
                compression_opts=4
            )
            done_dset = f.create_dataset(
                'done',
                shape=(MAX_TOTAL_FRAMES,),
                maxshape=(None,),
                chunks=(10000,),
                dtype=np.bool_,
                compression='gzip',
                compression_opts=4
            )
            
            for episode_num in tqdm(range(NUM_EPISODES), desc="Collecting Episodes (HDF5)"):
                # Reset environment for a new random track
                observation, info = env.reset(seed=episode_num)
                
                # Pre-process the initial frame
                current_frame = process_frame(observation)
                
                terminated = False
                truncated = False
                step = 0

                while not terminated and not truncated and step < MAX_FRAMES_PER_EPISODE:
                    # Check if we need to resize datasets
                    if current_idx >= MAX_TOTAL_FRAMES:
                        print(f"\nWarning: Reached maximum pre-allocated size at frame {current_idx}")
                        # Resize datasets if needed (though we pre-allocated enough)
                        new_size = current_idx + MAX_FRAMES_PER_EPISODE
                        obs_dset.resize((new_size, TARGET_SIZE[0], TARGET_SIZE[1], 3))
                        action_dset.resize((new_size, 3))
                        reward_dset.resize((new_size,))
                        done_dset.resize((new_size,))
                    
                    # 1. Choose action
                    action = get_action(step)
                    
                    # 2. Take step
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    
                    # 3. Pre-process the next frame
                    next_frame = process_frame(next_observation)
                    
                    # 4. Write directly to HDF5 datasets (incremental write)
                    obs_dset[current_idx] = current_frame
                    action_dset[current_idx] = action
                    reward_dset[current_idx] = reward
                    done_dset[current_idx] = terminated or truncated
                    
                    # Update for next loop iteration
                    current_frame = next_frame
                    current_idx += 1
                    step += 1
            
            # Truncate datasets to actual size collected
            print(f"\nTruncating datasets to actual size: {current_idx} frames")
            obs_dset.resize((current_idx, TARGET_SIZE[0], TARGET_SIZE[1], 3))
            action_dset.resize((current_idx, 3))
            reward_dset.resize((current_idx,))
            done_dset.resize((current_idx,))
            
            print(f"\n--- HDF5 Data Collection Complete ---")
            print(f"Total frames collected: {current_idx}")
            print(f"Observations shape: {obs_dset.shape}")
            print(f"Actions shape: {action_dset.shape}")
            print(f"Successfully saved all data to {save_path}")
    
    except Exception as e:
        print(f"\nError during HDF5 collection: {e}")
        # Clean up partial file on error
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
                print(f"Removed partial file: {save_path}")
            except:
                pass
        raise
    
    finally:
        env.close()

# --- Main Function ---

def collect_rollouts(format_type: str = "both"):
    """
    Main function to collect rollouts using the specified format(s).
    
    Args:
        format_type: One of "npz", "hdf5", or "both"
    """
    if format_type == "npz":
        collect_rollouts_npz()
    elif format_type == "hdf5":
        collect_rollouts_hdf5()
    elif format_type == "both":
        print("=" * 70)
        print("Collecting data in both NPZ and HDF5 formats")
        print("=" * 70)
        print("\n[1/2] Collecting NPZ format...")
        collect_rollouts_npz()
        print("\n[2/2] Collecting HDF5 format...")
        collect_rollouts_hdf5()
        print("\n" + "=" * 70)
        print("Both formats collected successfully!")
        print("=" * 70)
    else:
        raise ValueError(f"Unknown format type: {format_type}. Use 'npz', 'hdf5', or 'both'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Collect CarRacing rollouts with incremental writing (NPZ and/or HDF5)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='both',
        choices=['npz', 'hdf5', 'both'],
        help='Output format: npz (memory-mapped), hdf5 (native incremental), or both (default: both)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=NUM_EPISODES,
        help=f'Number of episodes to collect (default: {NUM_EPISODES})'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=MAX_FRAMES_PER_EPISODE,
        help=f'Maximum frames per episode (default: {MAX_FRAMES_PER_EPISODE})'
    )
    
    args = parser.parse_args()
    
    # Update global config if arguments provided
    global NUM_EPISODES, MAX_FRAMES_PER_EPISODE, MAX_TOTAL_FRAMES
    if args.episodes != NUM_EPISODES:
        NUM_EPISODES = args.episodes
    
    if args.max_frames != MAX_FRAMES_PER_EPISODE:
        MAX_FRAMES_PER_EPISODE = args.max_frames
    
    # Recalculate max total frames
    MAX_TOTAL_FRAMES = NUM_EPISODES * MAX_FRAMES_PER_EPISODE
    
    # Note: Requires the 'gymnasium' library and 'Pillow' (PIL) for image handling.
    # To run this script, ensure you have the required packages installed:
    # pip install gymnasium[box2d] numpy pillow tqdm h5py
    collect_rollouts(args.format)

