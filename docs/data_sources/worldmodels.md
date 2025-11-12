# World Models Dataset Sources

This project follows the original **World Models** paper setup by David Ha and Jürgen Schmidhuber. The authors released pre-recorded rollouts for both the Gym `CarRacing-v0` task and the VizDoom `take_cover` scenario. These archives contain the raw observations, control signals, and rewards that the paper used to train the VAE (Vision), MDN-RNN (Memory), and Controller components.

The data lives in a public Google Cloud Storage (GCS) bucket. We do **not** check the data into the repository. Instead, download it when you need it and place the files under `data/raw/worldmodels/` as described below.

## Download URLs

All files are public – no authentication is required. You can fetch them with `wget`, `curl`, or `gsutil`. The file sizes are roughly 1.3 GB per environment once uncompressed.

### CarRacing (`CarRacing-v0`)

| File | Description |
| --- | --- |
| `https://storage.googleapis.com/worldmodels/dataset/carracing_obs.zip` | RGB observations (64×64×3) for every timestep |
| `https://storage.googleapis.com/worldmodels/dataset/carracing_act.zip` | Continuous control actions (steer, gas, brake) |
| `https://storage.googleapis.com/worldmodels/dataset/carracing_reward.zip` | Scalar rewards |
| `https://storage.googleapis.com/worldmodels/dataset/carracing_terminal.zip` | Episode termination flags (`True` on the last step) |

### Doom (`VizDoomTakeCover-v0`)

| File | Description |
| --- | --- |
| `https://storage.googleapis.com/worldmodels/dataset/doom_obs.zip` | Grayscale observations (64×64×1) captured from VizDoom |
| `https://storage.googleapis.com/worldmodels/dataset/doom_act.zip` | Discrete action indices (three dodge actions) |
| `https://storage.googleapis.com/worldmodels/dataset/doom_reward.zip` | Rewards per timestep |
| `https://storage.googleapis.com/worldmodels/dataset/doom_terminal.zip` | Episode termination flags |

> ℹ️ If you cannot access the HTTPS URLs (TLS handshake issues), try the equivalent `gs://worldmodels/dataset/<file>` addresses with the [`gsutil`](https://cloud.google.com/storage/docs/gsutil) CLI or configure `curl`/`wget` with an explicit TLS version: `curl --tls-max 1.2 -O <url>`.

## Recommended Directory Layout

After downloading and extracting the archives, arrange the files like this:

```
data/
  raw/
    worldmodels/
      carracing/
        observations.npy
        actions.npy
        rewards.npy
        terminals.npy
      doom_take_cover/
        observations.npy
        actions.npy
        rewards.npy
        terminals.npy
```

- Each `.zip` archive contains a single `.npy` array. Unzip with `unzip carracing_obs.zip` (a `gunzip` equivalent also works) and rename the output to match the structure above for consistency.
- The arrays are stored in rollout order: time progresses along axis 0; you can reshape into episodes using the `terminals` mask.

## Quick Verification

Once the files are in place run the small pytest below to verify shapes. It expects CarRacing observations as RGB (HWC) and Doom observations as grayscale.

```bash
pytest tests/data/test_worldmodels_layout.py
```

If you notice mismatched shapes, double-check the unzip step – some shells auto-extract into directories named after the archive.

## Usage in Code

The loaders under `src/utils/worldmodels_dataset.py` expect the layout above. They take care of:

- Converting image arrays to PyTorch tensors (optionally normalised to `[0, 1]`).
- Reconstructing episode boundaries from the `terminals` vector.
- Returning tuples of `(observations, actions, rewards, dones)` per episode ready for the V-M-C pipeline.

See `examples/basic_usage.py` for end-to-end usage once the datasets are in place.

