"""Hagen-Poiseuille PINN for hemodynamics, with held-out validation.
Ground truth is physically consistent: p(t) waveform, Q = pi r^4 (p-80)/(8 mu L), v = Q/A.
Only pressure is observed (noisy, sparse) in the 'non-invasive' setting; flow and velocity are inferred.
Metrics: relative RMSE on held-out time points, 5 seeds. Units follow the original notebook (mmHg used directly in Q formula)."""
import numpy as np, torch, torch.nn as nn
R, MU, L, P0 = 0.5, 0.04, 10.0, 80.0
A = np.pi * R ** 2

def truth(t):
    p = P0 + 40 * (0.5 * (1 + np.sin(2 * np.pi * t - np.pi / 2)) + 0.2 * np.sin(4 * np.pi * t))
    q = np.pi * R ** 4 * (p - P0) / (8 * MU * L)
    return np.stack([p, q / A, q], 1)          # pressure, velocity, flow

def feats(t):
    t = torch.as_tensor(t, dtype=torch.float32).reshape(-1, 1)
    return torch.cat([torch.sin(2 * np.pi * t), torch.cos(2 * np.pi * t), torch.sin(4 * np.pi * t), torch.cos(4 * np.pi * t), t / 2], 1)

def run(arm, seed, noise=0.08, n_train=100, epochs=4000, lam=1.0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    t_all = np.linspace(0, 2, 500)
    idx = rng.permutation(500); tr, te = np.sort(idx[:n_train]), np.sort(idx[n_train:])   # random train / held-out times
    y_all = truth(t_all); mu, sd = y_all.mean(0), y_all.std(0)
    y_tr = y_all[tr] + rng.normal(0, 1, (n_train, 3)) * noise * sd                         # noisy observations
    yn = torch.tensor((y_tr - mu) / sd, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(5, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))
    opt = torch.optim.Adam(net.parameters(), 2e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ft, fc = feats(t_all[tr]), feats(rng.uniform(0, 2, 400))                               # fc: unlabeled collocation times
    mu_t, sd_t = torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)
    observed = [0] if arm in ('mlp_p', 'pinn_p') else [0, 1, 2]
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(ft)[:, observed] - yn[:, observed]) ** 2).mean()
        if arm.startswith('pinn'):
            y = net(fc) * sd_t + mu_t                                                      # back to physical units
            q_hp = np.pi * R ** 4 * (y[:, 0] - P0) / (8 * MU * L)
            phys = (((y[:, 2] - q_hp) / sd[2]) ** 2).mean() + (((y[:, 1] - y[:, 2] / A) / sd[1]) ** 2).mean()
            loss = loss + lam * phys
        loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        pred = net(feats(t_all[te])).numpy() * sd + mu
    return np.sqrt(((pred - y_all[te]) ** 2).mean(0)) / sd                                 # relative RMSE per output

if __name__ == '__main__':
    arms = {'mlp_all': 'MLP, p+v+Q observed', 'mlp_p': 'MLP, pressure only', 'pinn_p': 'PINN, pressure only', 'pinn_all': 'PINN, p+v+Q observed'}
    for k, name in arms.items():
        r = np.array([run(k, s) for s in range(5)])
        m, s = r.mean(0), r.std(0)
        print(f'{name:24s} relRMSE  pressure {m[0]:.3f}±{s[0]:.3f} | velocity {m[1]:.3f}±{s[1]:.3f} | flow {m[2]:.3f}±{s[2]:.3f}', flush=True)
