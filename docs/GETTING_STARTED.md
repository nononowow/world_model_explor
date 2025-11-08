# Getting Started

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd world_model_explor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run Basic Examples

```bash
python examples/basic_usage.py
```

This will demonstrate:
- VAE encoding and decoding
- RNN sequence prediction
- Controller action generation
- Complete world model workflow

### 2. Train a VAE

Train a VAE on MNIST:
```bash
python scripts/train_vae.py --dataset mnist --epochs 50
```

Train on CIFAR-10:
```bash
python scripts/train_vae.py --dataset cifar10 --epochs 50
```

### 3. Train World Model

Train a complete world model on CarRacing-v0:
```bash
python scripts/train_world_model.py --env CarRacing-v0 --collect_episodes 1000
```

This will:
1. Collect random rollouts
2. Train VAE on collected frames
3. Train RNN on latent sequences
4. Train controller using evolutionary strategies

### 4. Train Driving Model

Train a driving-specific world model:
```bash
python scripts/train_driving_model.py --env SimpleDrivingEnv
```

### 5. Test Model

Test a trained model:
```bash
python scripts/test_driving_model.py --model_path data/driving_world_model.pth --render
```

## Project Structure

```
world_model_explor/
├── src/                    # Source code
│   ├── vae/               # VAE implementations
│   ├── rnn/               # RNN/Dynamics models
│   ├── controller/        # Controllers
│   ├── world_model/       # Classic World Models
│   ├── planning/          # Planning algorithms
│   ├── driving/           # Driving-specific code
│   └── utils/             # Utilities
├── scripts/               # Training scripts
├── examples/              # Example code
├── docs/                  # Documentation
└── data/                  # Data and models
```

## Learning Path

Follow this progression to master world models:

1. **Phase 1**: Understand VAE, RNN, and Controller
   - Study `src/vae/basic_vae.py`
   - Study `src/rnn/mdn_rnn.py`
   - Study `src/controller/simple_controller.py`
   - Run `examples/basic_usage.py`

2. **Phase 2**: Train World Model
   - Run `scripts/train_world_model.py`
   - Understand the training pipeline in `src/world_model/trainer.py`
   - Experiment with hyperparameters

3. **Phase 3**: Advanced Techniques
   - Try β-VAE (`src/vae/improved_vae.py`)
   - Experiment with Transformer dynamics (`src/rnn/transformer_dynamics.py`)
   - Use planning algorithms (`src/planning/mpc_planner.py`)

4. **Phase 4**: Autonomous Driving
   - Train driving model (`scripts/train_driving_model.py`)
   - Understand multi-modal inputs (`src/driving/driving_vae.py`)
   - Explore safety features (`src/driving/safe_planner.py`)

5. **Phase 5**: Integration and Optimization
   - Integrate all components
   - Optimize for real-time performance
   - Document experiments

## Next Steps

- Read `docs/ARCHITECTURE.md` for detailed architecture
- Read `docs/TRAINING_GUIDE.md` for training tips
- Read `docs/API.md` for API reference
- Experiment with different environments
- Try different hyperparameters
- Implement your own improvements

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size
   - Use smaller models
   - Use CPU: `--device cpu`

2. **Import errors**
   - Make sure you're in the project root
   - Install dependencies: `pip install -r requirements.txt`

3. **Environment not found**
   - Install gymnasium: `pip install gymnasium[box2d]`
   - For Atari: `pip install gymnasium[atari]`

## Resources

- **Paper**: [World Models](https://arxiv.org/abs/1803.10122)
- **Paper**: [Dreamer](https://arxiv.org/abs/1912.01603)
- **Paper**: [PlaNet](https://arxiv.org/abs/1811.04551)

## Support

For questions or issues, please refer to the documentation or create an issue in the repository.

