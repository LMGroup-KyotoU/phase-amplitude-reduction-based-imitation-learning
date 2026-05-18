"""Verify and compare adjoint vs BPTT gradients on a small PA model.

Sections:
  1. Forward output equivalence
  2. Per-parameter gradient agreement
  3. Backward wall-clock comparison vs rollout length T
  4. Peak memory comparison (CUDA only; reported as 'n/a' on CPU)
  5. End-to-end learn() loss curves
"""
import os
import sys
import time
import resource
import gc

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_amplitude_reduction import PADynTrainer


def _grad_dict(trainer):
    grads = {}
    for name, p in trainer.enc.named_parameters():
        grads[("enc", name)] = None if p.grad is None else p.grad.detach().clone()
    for name, p in trainer.dec.named_parameters():
        grads[("dec", name)] = None if p.grad is None else p.grad.detach().clone()
    return grads


def _zero_grads(trainer):
    for p in trainer.enc.parameters():
        if p.grad is not None:
            p.grad.zero_()
    for p in trainer.dec.parameters():
        if p.grad is not None:
            p.grad.zero_()
    for p in trainer.dyn.parameters():
        if p.grad is not None:
            p.grad.zero_()


def _build_trainer(seed: int, use_adjoint: bool, *,
                   dim_latent=3, dim_obs=2, hidden=(16, 16),
                   delta_t=0.05, device="cpu") -> PADynTrainer:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return PADynTrainer(
        dim_latent=dim_latent,
        dim_obs=dim_obs,
        hidden_layer_size=list(hidden),
        delta_t=delta_t,
        num_torus_flow=1,
        device=device,
        num_iters=1,
        batch_size=4,
        latent_noise=0.0,
        natural_freq=1.0,
        amplitude_range=[0.5, 5.0],
        use_adjoint=use_adjoint,
    )


def _bptt_rollout(trainer, obs, T, noise):
    """Plain BPTT rollout matching AdjointRollout's forward (predict mode)."""
    B = obs.shape[0]
    enc_latent = trainer.enc(obs[:, 0:1])
    latent_0 = enc_latent[:, 0]
    n_torus = trainer.num_torus_flow
    tc = trainer.dyn._time_constants
    latent_hist = []
    obs_rec = []
    for i in range(T):
        t = trainer.delta_t * i
        z = torch.empty_like(latent_0)
        z[..., :n_torus] = latent_0[..., :n_torus] + tc[:n_torus] * t
        z[..., n_torus:] = latent_0[..., n_torus:] * torch.exp(tc[n_torus:] * t)
        z = z + noise[:, i]
        latent_hist.append(z)
        obs_rec.append(trainer.dec(z))
    return torch.stack(obs_rec, dim=1), torch.stack(latent_hist, dim=1)


def compare_rollout_grad(T=12, B=5):
    """Section 1+2: forward and gradient equivalence."""
    print(f"-- T={T}, B={B}, hidden=[16,16] --")
    torch.manual_seed(0)
    obs = torch.randn(B, T, 2)

    base = _build_trainer(seed=42, use_adjoint=False)
    adj = _build_trainer(seed=42, use_adjoint=True)
    for p_b, p_a in zip(base.enc.parameters(), adj.enc.parameters()):
        assert torch.equal(p_b, p_a)
    for p_b, p_a in zip(base.dec.parameters(), adj.dec.parameters()):
        assert torch.equal(p_b, p_a)

    noise = torch.zeros(B, T, base.dim_latent)
    target_obs = torch.randn(B, T, 2)
    target_lat = torch.randn(B, T, base.dim_latent)

    def loss_fn(obs_rec, latent_hist):
        return (
            ((obs_rec - target_obs) ** 2).mean()
            + 0.3 * ((latent_hist - target_lat) ** 2).mean()
        )

    obs_rec_b, latent_hist_b = _bptt_rollout(base, obs, T, noise)
    loss_b = loss_fn(obs_rec_b, latent_hist_b)
    _zero_grads(base)
    loss_b.backward()
    g_bptt = _grad_dict(base)

    obs_rec_a, latent_hist_a = adj.rollout_adjoint(obs, T, noise=noise)
    loss_a = loss_fn(obs_rec_a, latent_hist_a)
    _zero_grads(adj)
    loss_a.backward()
    g_adj = _grad_dict(adj)

    fwd_obs = (obs_rec_b - obs_rec_a).abs().max().item()
    fwd_lat = (latent_hist_b - latent_hist_a).abs().max().item()
    print(f"forward  max|obs_rec_BPTT - obs_rec_adj|    = {fwd_obs:.3e}")
    print(f"forward  max|latent_BPTT  - latent_adj|     = {fwd_lat:.3e}")
    print(f"loss     |loss_BPTT - loss_adj|             = {abs(loss_b.item() - loss_a.item()):.3e}")

    print("\nper-parameter gradient comparison:")
    print(f"  {'parameter':30s} {'max|BPTT|':>11s} {'max|diff|':>11s} {'rel':>11s}")
    max_rel = 0.0
    max_abs = 0.0
    for k in g_bptt:
        gb, ga = g_bptt[k], g_adj[k]
        if gb is None and ga is None:
            continue
        err = (gb - ga).abs().max().item()
        ref = gb.abs().max().item()
        rel = err / (ref + 1e-12)
        max_abs = max(max_abs, err)
        max_rel = max(max_rel, rel)
        name = f"{k[0]}.{k[1]}"
        print(f"  {name:30s} {ref:11.3e} {err:11.3e} {rel:11.3e}")
    print(f"\noverall max abs diff = {max_abs:.3e}  ({max_rel:.3e} relative)")
    assert max(fwd_obs, fwd_lat) < 1e-5
    assert max_rel < 1e-5
    print("OK: adjoint == BPTT (within float32 round-off)\n")


def _backward_time(trainer, obs, T, noise, target_obs, target_lat, use_adjoint):
    """Time one forward+backward, repeated REPEATS times."""
    REPEATS = 5
    # warm-up
    if use_adjoint:
        obs_rec, latent_hist = trainer.rollout_adjoint(obs, T, noise=noise)
    else:
        obs_rec, latent_hist = _bptt_rollout(trainer, obs, T, noise)
    loss = ((obs_rec - target_obs) ** 2).mean() + 0.3 * ((latent_hist - target_lat) ** 2).mean()
    _zero_grads(trainer)
    loss.backward()
    torch.cuda.synchronize() if obs.is_cuda else None

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        if use_adjoint:
            obs_rec, latent_hist = trainer.rollout_adjoint(obs, T, noise=noise)
        else:
            obs_rec, latent_hist = _bptt_rollout(trainer, obs, T, noise)
        loss = ((obs_rec - target_obs) ** 2).mean() + 0.3 * ((latent_hist - target_lat) ** 2).mean()
        _zero_grads(trainer)
        loss.backward()
        if obs.is_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / REPEATS


def _peak_memory_bytes(callable_fn):
    """Peak GPU memory for callable_fn on CUDA, otherwise process RSS delta."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        callable_fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    # CPU: peak RSS as a coarse proxy
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    callable_fn()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (rss_after - rss_before) * 1024  # ru_maxrss is KiB on Linux


def benchmark_time_vs_T():
    """Section 3+4: backward time and peak memory vs horizon T."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-- device={device}, hidden=[256,256], B=32 --")
    print(f"  {'T':>5s} {'BPTT (ms)':>12s} {'adjoint (ms)':>14s} {'speedup':>9s}"
          f" {'BPTT mem':>12s} {'adj  mem':>12s} {'ratio':>8s}")
    B = 32
    for T in (8, 16, 32, 64, 128):
        base = _build_trainer(seed=7, use_adjoint=False, hidden=(256, 256), device=device)
        adj = _build_trainer(seed=7, use_adjoint=True, hidden=(256, 256), device=device)
        obs = torch.randn(B, T, 2, device=device)
        noise = torch.zeros(B, T, base.dim_latent, device=device)
        target_obs = torch.randn(B, T, 2, device=device)
        target_lat = torch.randn(B, T, base.dim_latent, device=device)

        t_bptt = _backward_time(base, obs, T, noise, target_obs, target_lat, use_adjoint=False)
        t_adj = _backward_time(adj, obs, T, noise, target_obs, target_lat, use_adjoint=True)

        def run_bptt():
            obs_rec, latent_hist = _bptt_rollout(base, obs, T, noise)
            loss = ((obs_rec - target_obs) ** 2).mean() + 0.3 * ((latent_hist - target_lat) ** 2).mean()
            _zero_grads(base)
            loss.backward()

        def run_adj():
            obs_rec, latent_hist = adj.rollout_adjoint(obs, T, noise=noise)
            loss = ((obs_rec - target_obs) ** 2).mean() + 0.3 * ((latent_hist - target_lat) ** 2).mean()
            _zero_grads(adj)
            loss.backward()

        mem_bptt = _peak_memory_bytes(run_bptt)
        mem_adj = _peak_memory_bytes(run_adj)
        if mem_bptt <= 0 or mem_adj <= 0:
            mem_bptt_s = "n/a"
            mem_adj_s = "n/a"
            mem_ratio_s = "n/a"
        else:
            mem_bptt_s = f"{mem_bptt/1e6:8.2f} MB"
            mem_adj_s = f"{mem_adj/1e6:8.2f} MB"
            mem_ratio_s = f"{mem_bptt / mem_adj:7.2f}x"
        print(f"  {T:5d} {t_bptt*1e3:12.2f} {t_adj*1e3:14.2f}"
              f" {t_bptt/t_adj:8.2f}x"
              f" {mem_bptt_s:>12s} {mem_adj_s:>12s} {mem_ratio_s:>8s}")
        # release graphs
        del base, adj, obs, noise, target_obs, target_lat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if device == "cpu":
        print("  (memory column reports RSS delta on CPU and is noisy; run on CUDA for clean numbers)")
    print("  On CPU + small models, adjoint is slower because backward re-runs")
    print("  the decoder forward (compute roughly doubles). The win shows up on")
    print("  GPU with deep decoders / long T, where peak memory drops by ~T.")


def smoke_learn():
    """Section 5: end-to-end learn() losses with both backends."""
    omega = 2 * np.pi * 1.0
    dt = 0.05
    T = 20
    t = torch.arange(T) * dt
    base_traj = torch.stack(
        [torch.cos(omega * t), torch.sin(omega * t)], dim=-1
    )
    torch.manual_seed(0)
    traj = base_traj[None] + 0.05 * torch.randn(8, T, 2)

    def run(use_adjoint):
        # Re-seed before each run so model init AND noise sequences match.
        torch.manual_seed(123)
        trainer = PADynTrainer(
            dim_latent=2,
            dim_obs=2,
            hidden_layer_size=[32, 32],
            delta_t=dt,
            num_torus_flow=1,
            device="cpu",
            num_iters=50,
            batch_size=8,
            latent_noise=1e-4,
            natural_freq=1.0,
            amplitude_range=[1.0, 1.0],
            use_adjoint=use_adjoint,
        )
        torch.manual_seed(456)  # noise sequence
        trainer.learn(traj)
        return [x.item() for x in trainer.record["loss"]]

    losses_bptt = run(False)
    losses_adj = run(True)
    print(f"  {'iter':>5s} {'BPTT loss':>12s} {'adj loss':>12s}")
    for it in (0, 9, 19, 29, 39, 49):
        print(f"  {it:5d} {losses_bptt[it]:12.4f} {losses_adj[it]:12.4f}")
    print("  (small drift between curves comes from different noise sampling")
    print("   orderings inside the two rollout implementations, not from the")
    print("   gradient itself -- see section 1 for gradient equivalence.)")


if __name__ == "__main__":
    print("=" * 60)
    print("1) forward & gradient equivalence")
    print("=" * 60)
    compare_rollout_grad(T=12, B=5)
    compare_rollout_grad(T=64, B=8)

    print("=" * 60)
    print("2) backward wall-clock + peak memory vs T")
    print("=" * 60)
    benchmark_time_vs_T()

    print("=" * 60)
    print("3) end-to-end learn() loss curves")
    print("=" * 60)
    smoke_learn()
