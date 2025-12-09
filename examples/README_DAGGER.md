# DAgger Algorithm - Quick Start Guide

## Overview

This directory contains a toy example demonstrating the **DAgger (Dataset Aggregation)** algorithm for imitation learning. The example compares DAgger with Behavioral Cloning to show how DAgger addresses the distribution shift problem.

## Files

- `dagger_toy_example.py`: Complete implementation with visualization

## What is DAgger?

**DAgger** solves a key problem in imitation learning: **distribution shift**.

- **Problem**: Behavioral Cloning trains on expert states, but at test time, the policy visits different states (due to early mistakes)
- **Solution**: DAgger iteratively collects data from states the policy actually visits, then queries the expert for correct actions in those states

## Running the Example

### Prerequisites

Make sure you have the required dependencies:
```bash
pip install torch numpy matplotlib
```

### Run the Example

```bash
python examples/dagger_toy_example.py
```

### What You'll See

1. **Training Process**:
   - Behavioral Cloning training (baseline)
   - DAgger training (iterative dataset aggregation)
   - Performance metrics for both

2. **Visualizations** (saved as PNG files):
   - `dagger_comparison.png`: Trajectories of expert, BC, and DAgger policies
   - `dagger_performance.png`: Performance improvement over DAgger iterations
   - `dagger_dataset_growth.png`: How the dataset grows over iterations

## Understanding the Code

### Key Components

1. **Environment** (`SimpleNavigationEnv`):
   - 2D navigation task: reach goal at (10, 10) from (0, 0)
   - Obstacle at (5, 5) that must be avoided
   - 5 discrete actions: right, up, left, down, stay

2. **Expert Policy** (`ExpertPolicy`):
   - Rule-based policy that navigates to goal while avoiding obstacle
   - Represents the "teacher" we want to imitate

3. **Policy Network** (`PolicyNetwork`):
   - Simple neural network (2-layer MLP)
   - Takes state (x, y) → outputs action probabilities

4. **Behavioral Cloning** (`behavioral_cloning`):
   - Collects expert demonstrations once
   - Trains policy using supervised learning
   - **Problem**: Only sees expert states, fails on distribution shift

5. **DAgger** (`dagger`):
   - Iteratively:
     a. Run current policy to collect states
     b. Query expert for actions in those states
     c. Aggregate dataset
     d. Retrain policy
   - **Solution**: Sees states policy actually visits

## Key Concepts Demonstrated

### 1. Distribution Shift

**Behavioral Cloning**:
```
Training:  Expert states → Expert actions
Test:      Policy states → ??? (no training data!)
```

**DAgger**:
```
Iteration 1: Expert states → Expert actions
Iteration 2: Expert states + Policy states → Expert actions
Iteration 3: Expert states + Policy states (more diverse) → Expert actions
...
```

### 2. Iterative Improvement

- Each DAgger iteration adds more diverse states
- Policy sees training data for states it will actually visit
- Performance improves as dataset grows

### 3. Dataset Aggregation

- Dataset grows over time: `D_i = D_{i-1} ∪ new_data`
- All data is labeled by expert
- Policy learns from both expert and policy states

## Expected Results

You should see:
- **BC Performance**: Lower (suffers from distribution shift)
- **DAgger Performance**: Higher (addresses distribution shift)
- **Improvement**: DAgger typically outperforms BC by 20-50%

## Customization

You can modify:
- `num_iterations`: Number of DAgger iterations (default: 8)
- `rollouts_per_iter`: Rollouts per iteration (default: 5)
- `epochs_per_iter`: Training epochs per iteration (default: 50)
- Environment parameters (goal, obstacle, etc.)

## Learning More

For detailed explanation of DAgger, see:
- `docs/imitation_learning_dagger.md`: Comprehensive guide to DAgger algorithm

## References

- **Original Paper**: Ross, S., Gordon, G., & Bagnell, D. (2011). "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"

