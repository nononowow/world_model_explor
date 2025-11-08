# API Reference

## VAE

### `VAE`

Basic Variational Autoencoder.

```python
from src.vae.basic_vae import VAE

vae = VAE(input_channels=3, latent_dim=32)
recon_x, mu, logvar, z = vae(x)
```

### `BetaVAE`

β-VAE for disentangled representations.

```python
from src.vae.improved_vae import BetaVAE

vae = BetaVAE(input_channels=3, latent_dim=32, beta=4.0)
```

## RNN

### `MDNRNN`

Mixture Density Network RNN.

```python
from src.rnn.mdn_rnn import MDNRNN

rnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256)
pi, mu, logvar, hidden = rnn(latent_states, actions)
```

## Controller

### `SimpleController`

Basic policy network.

```python
from src.controller.simple_controller import SimpleController

controller = SimpleController(latent_dim=32, action_dim=3)
action = controller.act(latent_state)
```

## World Model

### `WorldModelTrainer`

Complete training pipeline.

```python
from src.world_model.trainer import WorldModelTrainer

trainer = WorldModelTrainer(env_name='CarRacing-v0')
trainer.collect_data(num_episodes=1000)
trainer.train_vae()
trainer.train_rnn()
trainer.train_controller()
```

## Planning

### `CEMPlanner`

Cross Entropy Method planner.

```python
from src.planning.mpc_planner import CEMPlanner

planner = CEMPlanner(action_dim=3, horizon=12)
action = planner.plan(world_model, initial_latent, reward_fn)
```

## Driving

### `DrivingWorldModel`

Complete driving world model.

```python
from src.driving.driving_world_model import DrivingWorldModel

model = DrivingWorldModel(latent_dim=32, action_dim=3, state_dim=4)
latent = model.encode(image, state)
action = model.act(latent)
```

