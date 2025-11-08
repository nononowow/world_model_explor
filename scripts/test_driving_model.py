"""
Test script for driving world model.

Usage:
    python scripts/test_driving_model.py --model_path data/driving_world_model.pth
"""

import argparse
import torch
from src.driving.driving_world_model import DrivingWorldModel
from src.driving.evaluator import DrivingEvaluator
from src.environments.driving_env import SimpleDrivingEnv


def main():
    parser = argparse.ArgumentParser(description='Test Driving World Model')
    parser.add_argument('--model_path', type=str, default='data/driving_world_model.pth')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Initialize world model
    world_model = DrivingWorldModel(
        latent_dim=32,
        action_dim=3,
        state_dim=4,
        device=args.device
    )
    
    # Load weights
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=args.device)
        world_model.vae.load_state_dict(checkpoint['vae'])
        world_model.dynamics.load_state_dict(checkpoint['dynamics'])
        world_model.controller.load_state_dict(checkpoint['controller'])
        print(f"Loaded model from {args.model_path}")
    
    # Create environment
    env = SimpleDrivingEnv(render_mode='human' if args.render else 'rgb_array')
    
    # Evaluate
    evaluator = DrivingEvaluator(env, world_model, device=args.device)
    results = evaluator.evaluate(
        num_episodes=args.num_episodes,
        max_steps=1000,
        render=args.render
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print("=" * 50)
    
    evaluator.close()


if __name__ == '__main__':
    main()

