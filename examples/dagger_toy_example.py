"""
DAgger Algorithm - Toy Example with PyTorch

This example demonstrates the DAgger (Dataset Aggregation) algorithm for imitation learning.
We use a simple 2D navigation task where an agent must reach a goal while avoiding obstacles.

Key Concepts Demonstrated:
1. Behavioral Cloning (BC) - baseline method
2. DAgger - iterative dataset aggregation
3. Distribution shift problem
4. How DAgger addresses distribution shift
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
from typing import List, Tuple, Dict


# ============================================================================
# Environment: Simple 2D Navigation Task
# ============================================================================

class SimpleNavigationEnv:
    """
    Simple 2D navigation environment.
    Agent starts at (0, 0) and must reach goal at (10, 10).
    There's an obstacle at (5, 5) that must be avoided.
    """
    
    def __init__(self, goal=(10.0, 10.0), obstacle=(5.0, 5.0), obstacle_radius=1.5):
        self.goal = np.array(goal)
        self.obstacle = np.array(obstacle)
        self.obstacle_radius = obstacle_radius
        self.state_dim = 2  # (x, y) position
        self.action_dim = 2  # (dx, dy) velocity
        
        # Action space: 4 directions + stay
        self.actions = np.array([
            [0.5, 0.0],   # Right
            [0.0, 0.5],   # Up
            [-0.5, 0.0],  # Left
            [0.0, -0.5],  # Down
            [0.0, 0.0]    # Stay
        ])
        
    def reset(self):
        """Reset environment to initial state."""
        self.position = np.array([0.0, 0.0])
        return self.position.copy()
    
    def step(self, action_idx: int):
        """
        Execute action and return next state, reward, done.
        
        Args:
            action_idx: Index of action to take (0-4)
        
        Returns:
            next_state: New position
            reward: Reward for this step
            done: Whether episode is finished
        """
        action = self.actions[action_idx]
        new_position = self.position + action
        
        # Check collision with obstacle
        dist_to_obstacle = np.linalg.norm(new_position - self.obstacle)
        if dist_to_obstacle < self.obstacle_radius:
            # Bounce back if hitting obstacle
            new_position = self.position
        
        self.position = new_position
        
        # Calculate reward
        dist_to_goal = np.linalg.norm(self.position - self.goal)
        reward = -dist_to_goal  # Negative distance (closer = better)
        
        # Check if reached goal
        done = dist_to_goal < 0.5
        
        if done:
            reward += 100  # Bonus for reaching goal
        
        return self.position.copy(), reward, done
    
    def get_state(self):
        """Get current state."""
        return self.position.copy()


# ============================================================================
# Expert Policy: Rule-based policy that navigates to goal
# ============================================================================

class ExpertPolicy:
    """
    Expert policy that uses simple rules to navigate to goal.
    This represents the "teacher" we want to imitate.
    """
    
    def __init__(self, env: SimpleNavigationEnv):
        self.env = env
        self.goal = env.goal
        self.obstacle = env.obstacle
        self.obstacle_radius = env.obstacle_radius
    
    def get_action(self, state: np.ndarray) -> int:
        """
        Get expert action for given state.
        
        Strategy:
        1. Move towards goal
        2. Avoid obstacle by going around it
        """
        position = state
        goal_dir = self.goal - position
        dist_to_goal = np.linalg.norm(goal_dir)
        
        if dist_to_goal < 0.5:
            return 4  # Stay (reached goal)
        
        # Check if moving towards goal would hit obstacle
        # If so, go around it
        dist_to_obstacle = np.linalg.norm(position - self.obstacle)
        
        if dist_to_obstacle < self.obstacle_radius + 1.0:
            # Too close to obstacle, go around
            # Prefer going right and up to avoid obstacle
            if goal_dir[0] > 0:
                return 0  # Right
            else:
                return 1  # Up
        else:
            # Safe to move towards goal
            if abs(goal_dir[0]) > abs(goal_dir[1]):
                return 0 if goal_dir[0] > 0 else 2  # Right or Left
            else:
                return 1 if goal_dir[1] > 0 else 3  # Up or Down


# ============================================================================
# Learnable Policy: Neural Network
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Simple neural network policy.
    Takes state (x, y) and outputs action probabilities.
    """
    
    def __init__(self, state_dim=2, action_dim=5, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, state):
        """Forward pass: state -> action probabilities."""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
    def get_action(self, state, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            state: Current state (numpy array or torch tensor)
            deterministic: If True, return most likely action
        
        Returns:
            action: Action index (0-4)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        with torch.no_grad():
            probs = self.forward(state)
            if deterministic:
                return torch.argmax(probs).item()
            else:
                return torch.multinomial(probs, 1).item()


# ============================================================================
# Behavioral Cloning (Baseline)
# ============================================================================

def behavioral_cloning(env, expert, num_demos=50, epochs=100, lr=0.001):
    """
    Train policy using behavioral cloning (supervised learning on expert data).
    
    This is the baseline method that suffers from distribution shift.
    """
    print("\n" + "="*60)
    print("Training with Behavioral Cloning (BC)")
    print("="*60)
    
    # Collect expert demonstrations
    print(f"Collecting {num_demos} expert demonstrations...")
    expert_data = []
    
    for _ in range(num_demos):
        state = env.reset()
        done = False
        while not done:
            action = expert.get_action(state)
            expert_data.append((state.copy(), action))
            state, _, done = env.step(action)
            if len(expert_data) > 200:  # Limit episode length
                break
    
    print(f"Collected {len(expert_data)} (state, action) pairs")
    
    # Prepare training data
    states = torch.FloatTensor([s for s, _ in expert_data])
    actions = torch.LongTensor([a for _, a in expert_data])
    
    # Initialize policy
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train policy
    print(f"Training policy for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        probs = policy(states)
        loss = criterion(probs, actions)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return policy, expert_data


# ============================================================================
# DAgger Algorithm
# ============================================================================

def dagger(env, expert, num_iterations=10, rollouts_per_iter=5, 
           epochs_per_iter=50, lr=0.001, max_steps=100):
    """
    Train policy using DAgger algorithm.
    
    Args:
        env: Environment
        expert: Expert policy
        num_iterations: Number of DAgger iterations
        rollouts_per_iter: Number of rollouts per iteration
        epochs_per_iter: Training epochs per iteration
        lr: Learning rate
        max_steps: Maximum steps per rollout
    
    Returns:
        policy: Trained policy
        all_data: All collected data across iterations
        performance_history: Performance at each iteration
    """
    print("\n" + "="*60)
    print("Training with DAgger Algorithm")
    print("="*60)
    
    # Initialize dataset (start with some expert demonstrations)
    print("Initializing with expert demonstrations...")
    dataset = []
    
    # Collect initial expert demonstrations
    for _ in range(5):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = expert.get_action(state)
            dataset.append((state.copy(), action))
            state, _, done = env.step(action)
            steps += 1
    
    print(f"Initial dataset size: {len(dataset)}")
    
    # Initialize policy
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    performance_history = []
    all_data = [dataset.copy()]
    
    # DAgger iterations
    for iteration in range(num_iterations):
        print(f"\n--- DAgger Iteration {iteration + 1}/{num_iterations} ---")
        
        # Step 1: Run current policy and collect states
        print(f"Running policy for {rollouts_per_iter} rollouts...")
        new_states = []
        
        for rollout in range(rollouts_per_iter):
            state = env.reset()
            done = False
            steps = 0
            trajectory_states = []
            
            while not done and steps < max_steps:
                trajectory_states.append(state.copy())
                action = policy.get_action(state, deterministic=False)
                state, _, done = env.step(action)
                steps += 1
                
                if done:
                    break
            
            new_states.extend(trajectory_states)
        
        print(f"Collected {len(new_states)} new states from policy rollouts")
        
        # Step 2: Query expert for actions in these states
        print("Querying expert for actions...")
        new_data = []
        for state in new_states:
            expert_action = expert.get_action(state)
            new_data.append((state, expert_action))
        
        # Step 3: Aggregate dataset
        dataset.extend(new_data)
        all_data.append(new_data.copy())
        print(f"Total dataset size: {len(dataset)}")
        
        # Step 4: Train policy on aggregated dataset
        print(f"Training policy for {epochs_per_iter} epochs...")
        states = torch.FloatTensor([s for s, _ in dataset])
        actions = torch.LongTensor([a for _, a in dataset])
        
        for epoch in range(epochs_per_iter):
            optimizer.zero_grad()
            probs = policy(states)
            loss = criterion(probs, actions)
            loss.backward()
            optimizer.step()
        
        print(f"Training loss: {loss.item():.4f}")
        
        # Step 5: Evaluate performance
        avg_reward = evaluate_policy(env, policy, num_episodes=10)
        performance_history.append(avg_reward)
        print(f"Average reward: {avg_reward:.2f}")
    
    return policy, all_data, performance_history


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_policy(env, policy, num_episodes=10, max_steps=200):
    """
    Evaluate policy performance.
    
    Returns:
        average_reward: Average total reward per episode
    """
    total_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action = policy.get_action(state, deterministic=True)
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
        
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards)


def visualize_trajectory(env, policy, title="Trajectory", ax=None):
    """
    Visualize a trajectory taken by the policy.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    state = env.reset()
    trajectory = [state.copy()]
    done = False
    steps = 0
    
    while not done and steps < 200:
        action = policy.get_action(state, deterministic=True)
        state, _, done = env.step(action)
        trajectory.append(state.copy())
        steps += 1
    
    trajectory = np.array(trajectory)
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Path')
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
    
    # Plot goal
    ax.plot(env.goal[0], env.goal[1], 'g*', markersize=20, label='Goal')
    
    # Plot obstacle
    circle = plt.Circle(env.obstacle, env.obstacle_radius, color='red', alpha=0.3)
    ax.add_patch(circle)
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    return ax


# ============================================================================
# Main: Run Comparison
# ============================================================================

def main():
    """Main function to demonstrate DAgger vs Behavioral Cloning."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create environment and expert
    env = SimpleNavigationEnv()
    expert = ExpertPolicy(env)
    
    print("="*60)
    print("DAgger Algorithm - Toy Example")
    print("="*60)
    print("\nEnvironment: 2D Navigation")
    print("Goal: Reach (10, 10) from (0, 0)")
    print("Obstacle: Circle at (5, 5) with radius 1.5")
    print("\nWe'll compare:")
    print("1. Behavioral Cloning (BC) - baseline")
    print("2. DAgger - iterative dataset aggregation")
    
    # ========================================================================
    # 1. Behavioral Cloning
    # ========================================================================
    bc_policy, bc_data = behavioral_cloning(
        env, expert, 
        num_demos=30, 
        epochs=100, 
        lr=0.001
    )
    
    bc_reward = evaluate_policy(env, bc_policy, num_episodes=20)
    print(f"\nBC Final Performance: {bc_reward:.2f}")
    
    # ========================================================================
    # 2. DAgger
    # ========================================================================
    dagger_policy, dagger_data, performance_history = dagger(
        env, expert,
        num_iterations=8,
        rollouts_per_iter=5,
        epochs_per_iter=50,
        lr=0.001,
        max_steps=100
    )
    
    dagger_reward = evaluate_policy(env, dagger_policy, num_episodes=20)
    print(f"\nDAgger Final Performance: {dagger_reward:.2f}")
    
    # ========================================================================
    # 3. Visualization
    # ========================================================================
    print("\n" + "="*60)
    print("Creating Visualizations...")
    print("="*60)
    
    fig = plt.figure(figsize=(16, 5))
    
    # Expert trajectory
    ax1 = plt.subplot(1, 3, 1)
    visualize_trajectory(env, expert, "Expert Policy", ax1)
    
    # BC trajectory
    ax2 = plt.subplot(1, 3, 2)
    visualize_trajectory(env, bc_policy, f"Behavioral Cloning\n(Reward: {bc_reward:.1f})", ax2)
    
    # DAgger trajectory
    ax3 = plt.subplot(1, 3, 3)
    visualize_trajectory(env, dagger_policy, f"DAgger\n(Reward: {dagger_reward:.1f})", ax3)
    
    plt.tight_layout()
    plt.savefig('dagger_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'dagger_comparison.png'")
    
    # Performance over iterations
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(1, len(performance_history) + 1), performance_history, 
            'b-o', linewidth=2, markersize=8)
    ax.axhline(y=bc_reward, color='r', linestyle='--', 
               label=f'BC Performance ({bc_reward:.1f})')
    ax.set_xlabel('DAgger Iteration')
    ax.set_ylabel('Average Reward')
    ax.set_title('DAgger Performance Over Iterations')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('dagger_performance.png', dpi=150, bbox_inches='tight')
    print("Saved performance plot to 'dagger_performance.png'")
    
    # Dataset size growth
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    dataset_sizes = [len(data) for data in dagger_data]
    cumulative_sizes = np.cumsum(dataset_sizes)
    ax.plot(range(len(cumulative_sizes)), cumulative_sizes, 'g-o', 
            linewidth=2, markersize=8)
    ax.set_xlabel('DAgger Iteration')
    ax.set_ylabel('Cumulative Dataset Size')
    ax.set_title('Dataset Growth in DAgger')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dagger_dataset_growth.png', dpi=150, bbox_inches='tight')
    print("Saved dataset growth plot to 'dagger_dataset_growth.png'")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Behavioral Cloning Reward: {bc_reward:.2f}")
    print(f"DAgger Reward: {dagger_reward:.2f}")
    print(f"Improvement: {dagger_reward - bc_reward:.2f} ({((dagger_reward - bc_reward) / abs(bc_reward) * 100):.1f}%)")
    print("\nKey Takeaways:")
    print("1. BC only sees expert states → fails on distribution shift")
    print("2. DAgger collects data from policy states → handles distribution shift")
    print("3. DAgger iteratively improves by seeing more diverse states")
    print("4. Dataset grows over iterations, improving policy coverage")
    
    plt.show()


if __name__ == "__main__":
    main()

