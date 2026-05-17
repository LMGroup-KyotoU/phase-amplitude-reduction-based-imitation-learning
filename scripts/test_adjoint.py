"""Verify adjoint-method gradients match BPTT on a small PA model."""
import os
import sys

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


def _build_trainer(seed: int, use_adjoint: bool) -> PADynTrainer:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return PADynTrainer(
        dim_latent=3,
        dim_obs=2,
        hidden_layer_size=[16, 16],
        delta_t=0.05,
        num_torus_flow=1,
        device="cpu",
        num_iters=1,
        batch_size=4,
        latent_noise=0.0,
        natural_freq=1.0,
        amplitude_range=[0.5, 5.0],
        use_adjoint=use_adjoint,
    )


def compare_rollout_grad(T: int = 12, B: int = 5):
    """grad through the (rollout -> loss(obs_rec) + loss(latent_hist)) chain
    must match between BPTT and the adjoint implementation."""
    torch.manual_seed(0)
    obs = torch.randn(B, T, 2)

    base = _build_trainer(seed=42, use_adjoint=False)
    adj = _build_trainer(seed=42, use_adjoint=True)
    # weights are identical because the same seed was used inside _build_trainer
    for p_b, p_a in zip(base.enc.parameters(), adj.enc.parameters()):
        assert torch.equal(p_b, p_a)
    for p_b, p_a in zip(base.dec.parameters(), adj.dec.parameters()):
        assert torch.equal(p_b, p_a)

    noise = torch.zeros(B, T, base.dim_latent)  # deterministic
    target_obs = torch.randn(B, T, 2)
    target_lat = torch.randn(B, T, base.dim_latent)

    def loss_fn(obs_rec, latent_hist):
        # something that exercises both outputs
        return (
            ((obs_rec - target_obs) ** 2).mean()
            + 0.3 * ((latent_hist - target_lat) ** 2).mean()
        )

    # --- BPTT path ---
    enc_latent = base.enc(obs[:, 0:1])
    latent_0 = enc_latent[:, 0]
    latent_hist_b = torch.empty(B, T, base.dim_latent)
    obs_rec_b = torch.empty(B, T, 2)
    n_torus = base.num_torus_flow
    tc = base.dyn._time_constants
    for i in range(T):
        t = base.delta_t * i
        z = torch.empty_like(latent_0)
        z[..., :n_torus] = latent_0[..., :n_torus] + tc[:n_torus] * t
        z[..., n_torus:] = latent_0[..., n_torus:] * torch.exp(tc[n_torus:] * t)
        z = z + noise[:, i]
        latent_hist_b[:, i] = z
        obs_rec_b[:, i] = base.dec(z)
    loss_b = loss_fn(obs_rec_b, latent_hist_b)
    _zero_grads(base)
    loss_b.backward()
    g_bptt = _grad_dict(base)

    # --- adjoint path ---
    obs_rec_a, latent_hist_a = adj.rollout_adjoint(obs, T, noise=noise)
    loss_a = loss_fn(obs_rec_a, latent_hist_a)
    _zero_grads(adj)
    loss_a.backward()
    g_adj = _grad_dict(adj)

    # forward outputs match
    max_fwd_err = max(
        (obs_rec_b - obs_rec_a).abs().max().item(),
        (latent_hist_b - latent_hist_a).abs().max().item(),
    )
    print(f"forward max|diff| = {max_fwd_err:.3e}")

    # gradients match
    max_grad_err = 0.0
    for k in g_bptt:
        gb = g_bptt[k]
        ga = g_adj[k]
        if gb is None and ga is None:
            continue
        if gb is None or ga is None:
            raise AssertionError(f"grad mismatch (one is None) at {k}")
        err = (gb - ga).abs().max().item()
        ref = gb.abs().max().item()
        rel = err / (ref + 1e-12)
        print(f"  {k[0]}.{k[1]:<20s}  max|diff|={err:.3e}  rel={rel:.3e}")
        max_grad_err = max(max_grad_err, err)
    print(f"\noverall grad max|diff| = {max_grad_err:.3e}")
    assert max_fwd_err < 1e-5, "forward outputs differ"
    assert max_grad_err < 1e-4, "gradients differ"
    print("OK: adjoint matches BPTT")


def smoke_learn(use_adjoint: bool):
    """Run a few learn() iterations end-to-end and report final loss."""
    torch.manual_seed(0)
    omega = 2 * np.pi * 1.0
    dt = 0.05
    T = 20
    t = torch.arange(T) * dt
    base_traj = torch.stack(
        [torch.cos(omega * t), torch.sin(omega * t)], dim=-1
    )
    traj = base_traj[None] + 0.05 * torch.randn(8, T, 2)

    trainer = PADynTrainer(
        dim_latent=2,
        dim_obs=2,
        hidden_layer_size=[32, 32],
        delta_t=dt,
        num_torus_flow=1,
        device="cpu",
        num_iters=20,
        batch_size=8,
        latent_noise=1e-4,
        natural_freq=1.0,
        amplitude_range=[1.0, 1.0],
        use_adjoint=use_adjoint,
    )
    trainer.learn(traj)
    losses = [x.item() for x in trainer.record["loss"]]
    print(f"use_adjoint={use_adjoint}  loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}")
    return losses


if __name__ == "__main__":
    print("== gradient equivalence ==")
    compare_rollout_grad()
    print("\n== smoke learn() ==")
    smoke_learn(use_adjoint=False)
    smoke_learn(use_adjoint=True)
