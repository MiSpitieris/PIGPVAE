# PIGPVAE – Physics‑Informed Gaussian Process Variational Autoencoders

> **What is it?**  
> PIGPVAE marries physics‑based simulators with deep latent‑variable models, letting you generate *realistic* synthetic time‑series even when only a handful of measurements are available.  
> The method extends a Variational Auto‑Encoder (VAE) with
>
> * a **physics decoder** (any ODE, PDE, or PINN) that captures known system dynamics, and
> * a **latent Gaussian‑Process (GP) discrepancy** that learns the missing physics.
>
> The result is an interpretable generative model that stays faithful to first principles *and* outperforms purely data‑driven baselines under small‑data and distribution‑shift scenarios.

If you use this code, **please cite the paper** (see the *Citation* section).

---

## Table of contents
1. [Installation](#installation)
2. [Project layout](#project-layout)
3. [Quick start](#quick-start)
4. [Reproducing the paper](#reproducing-the-paper)
5. [Datasets](#datasets)
6. [Citation](#citation)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)

---

## Installation

```bash
# Clone
git clone https://github.com/MiSpitieris/PIGPVAE.git
cd PIGPVAE

# (Recommended) create a virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt        # installs torch, gpytorch, torchdiffeq, etc.

# Run a quick test
python - <<'PY'
from physics import NewtonsLaw
import torch, matplotlib.pyplot as plt
model = NewtonsLaw(); t = torch.linspace(0,20,100)
print(model(T0=30.0, t=t, Ts=20.0, k=0.1)[:5])
PY
```

---

## Project layout

```text
PIGPVAE/
├── Data/                    # Pre‑processed HVAC heating/cooling curves & pendulum trajectories
├── Models/                  # PyTorch modules
├── Notebooks_R4/            # Real‑data experiments (full distribution)
├── Notebooks_R4_limited/    # Real‑data experiments (OOD generation)
├── Notebooks_Pendulum/      # Simulated pendulum study
├── GP.py                    # GP‑based latent module (uses GPyTorch)
├── physics.py               # Physics decoders (Newton’s law, simple pendulum, ...)
├── VAE_utils.py             # Encoder/decoder helpers, losses, annealing schedule
├── metrics.py               # MMD, MDD, correlation difference, etc.
└── ...
```

---

## Quick start

The **fastest way** to see PIGPVAE in action is to open the demo notebooks:

```bash
# Launch Jupyter Lab / Notebook
jupyter lab
# then open Notebooks_R4/_PIGPVAE_cooling.ipynb.ipynb (in‑distribution) or
#      Notebooks_R4_limited/_PIGPVAE_cooling.ipynb.ipynb (out‑of‑distribution)
```

### Minimal Python snippet

```python
import torch
from physics import NewtonsLaw
from GP import GP_inference
from VAE_utils import q_net, Decoder

# Define encoder/decoder networks (tiny example)
enc = q_net(input_size=24, hidden_layers=[64,32], latent_dim=1, activation='relu')
dec_phy = NewtonsLaw()             # physics decoder: Newton’s law of cooling
# Assume GP_inference is wrapped inside a PIGPVAE model… see notebooks for full code
```

---

## Reproducing the paper

| Experiment | Notebook | Command |
|------------|----------|---------|
| HVAC heating/cooling (in‑distribution) | `Notebooks_R4/_PIGPVAE_cooling.ipynb.ipynb` | *Run all cells* |
| HVAC OOD generation | `Notebooks_R4_limited/_PIGPVAE_cooling.ipynb.ipynb` | *Run all cells* |
| Pendulum synthetic study | `Notebooks_Pendulum/PIGPVAE_Pendulum.ipynb` | *Run all cells* |

Each notebook folder trains PIGPVAE *and* baseline models (PIVAE and GPVAE), reproducing the figures & metrics reported in the paper in the realted metrics notebooks.

---

## Datasets

* **RICO HVAC curves** – 29 heating and 28 cooling trajectories collected in a climate‑controlled test room. Pre‑processed splits are included under `Data/`. The raw dataset is available on Zenodo: <https://zenodo.org/record/14871584>.
* **Pendulum simulations** – Generated on‑the‑fly by `physics.SimplePendulumSolver` with added damping & forcing terms (see paper, Section 5).

---

## Citation

```bibtex
@article{Spitieris2025PIGPVAE,
  title   = {PIGPVAE: Physics-Informed Gaussian Process Variational Autoencoders},
  author  = {Spitieris, Michail and Ruocco, Massimiliano and Murad, Abdulmajid and Nocente, Alessandro},
  journal = {Applied Intelligence},
  year    = {2025},
  note    = {In press}
}
```

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.
 

