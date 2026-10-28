# Physics-Informed Neural Network for Cardiovascular Hemodynamics

A small PINN that infers blood **velocity and flow rate from pressure alone**, using the Hagen–Poiseuille relation
`Q = π r⁴ ΔP / (8 μ L)` and the flow–velocity relation `v = Q / A` as physics constraints. PyTorch, 3×64 tanh MLP.

* `Cardiovascular_Hemodynamics.ipynb` – original exploratory notebook (synthetic data, training curves).
* `pinn_cardio.py` – controlled benchmark with held-out validation (`python pinn_cardio.py`, ~4 min on CPU).

## Validation (`pinn_cardio.py`)

Ground truth is a physically consistent synthetic waveform (2 cardiac cycles, 500 time points). 100 random points get 8 % Gaussian
noise and are used for training; the other 400 are held out. Relative RMSE on held-out times, mean ± std over 5 seeds
(lower is better):

| Model | Observed | Pressure | Velocity | Flow |
|---|---|---|---|---|
| MLP | pressure only | 0.058 ± 0.010 | 0.992 ± 0.096 | 1.022 ± 0.059 |
| **PINN** | pressure only | 0.067 ± 0.005 | **0.065 ± 0.005** | **0.066 ± 0.005** |
| MLP | pressure + velocity + flow | 0.061 ± 0.010 | 0.070 ± 0.009 | 0.059 ± 0.006 |
| **PINN** | pressure + velocity + flow | **0.027 ± 0.005** | **0.027 ± 0.003** | **0.026 ± 0.004** |

* With pressure only, a plain MLP cannot recover flow or velocity (error ≈ 100 %); the PINN recovers both to ≈ 6.5 % from the physics terms.
* With all three signals observed, the physics terms cut held-out error by about 2–2.7× under noise and sparse data.

## Limitations (please read)

* **Synthetic data only.** No clinical or measured data. The ground truth is generated from the same equations used as the physics loss,
  so this shows the method works when the physics is right, not that it works on real patients.
* The constraints are **algebraic** (Hagen–Poiseuille, `v = Q/A`) for a single 1-D vessel segment. There is no Navier–Stokes / PDE
  residual and no spatial dimension. Pressure is used in mmHg directly in the Q formula (no unit conversion), following the original notebook.
* Not compared against a numerical solver or a clinical reference.
