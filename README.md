# World Model Exploration

A comprehensive implementation of world models from scratch, progressing from foundational concepts to autonomous driving applications.

## Overview

This project implements world models following a structured learning path:
1. **Phase 1**: Foundation - VAE, MDN-RNN, and basic controllers
2. **Phase 2**: Classic World Models - Full implementation for CarRacing-v0
3. **Phase 3**: Advanced Techniques - Improved architectures and planning
4. **Phase 4**: Autonomous Driving - Driving-specific world models
5. **Phase 5**: Integration - Complete system with optimization

## Project Structure

```
world_model_explor/
├── src/
│   ├── vae/              # Variational Autoencoder implementations
│   ├── rnn/              # RNN/Dynamics model implementations
│   ├── controller/       # Policy networks
│   ├── world_model/      # Classic World Models implementation
│   ├── planning/         # Planning algorithms
│   ├── driving/          # Autonomous driving specific code
│   ├── environments/     # Environment wrappers
│   ├── utils/            # Utilities and helpers
│   └── cpp/              # C++ implementations
├── scripts/              # Training and evaluation scripts
├── experiments/          # Experiment configs and results
├── data/                 # Data storage
├── docs/                 # Documentation
└── tests/                # Unit tests
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Phase 1: Foundation
```bash
# Train basic VAE
python scripts/train_vae.py --dataset mnist

# Train MDN-RNN
python scripts/train_mdn_rnn.py

# Test controller
python scripts/test_controller.py
```

### Phase 2: World Models
```bash
# Collect data
python scripts/collect_data.py --env CarRacing-v0

# Train world model
python scripts/train_world_model.py --env CarRacing-v0
```

## References

- World Models (Ha & Schmidhuber, 2018)
- Dreamer (Hafner et al., 2019)
- PlaNet (Hafner et al., 2019)

## License

MIT

