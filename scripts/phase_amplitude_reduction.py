from itertools import chain
from collections import OrderedDict
from typing import Union, List, Optional, Sequence, Callable, Any

import numpy as np
import torch
from torch import nn
from torch import optim


class PADynTrainer:

    def __init__(
        self,
        dim_latent: int,
        dim_obs: int,
        hidden_layer_size: List[int],
        delta_t: float,
        num_torus_flow: int=1,
        device: str="cpu",
        discount_gamma: float=0.99,
        learning_rate: float=1e-4,
        num_iters: int=1000,
        batch_size: int=255,
        latent_noise: Union[float, List[float]]=1e-3,
        natural_freq: Union[float, List[float]]=1.0,
        amplitude_range: Optional[Sequence]=None,
        recon_loss_coef: Optional[Sequence]=None,
        enable_loss: Optional[Sequence]=None,
    ) -> None:
        self.dim_obs = dim_obs
        self.dim_latent = dim_latent
        self.num_torus_flow = num_torus_flow
        self.delta_t = delta_t
        if amplitude_range is None:
            if isinstance(natural_freq, float):  
                amplitude_range = (natural_freq / 2, 1 / delta_t / 2)
            else:
                amplitude_range = (natural_freq[0] / 2, 1 / delta_t / 2)

        self.enc = Encoder(dim_latent, dim_obs, num_torus_flow, hidden_layer_size)
        self.dec = Decoder(dim_latent, dim_obs, num_torus_flow, hidden_layer_size)
        self.dyn = PADynamics(
            dim_latent,
            delta_t,
            natural_freq=natural_freq,
            amplitude_range=amplitude_range,
            num_torus_flow=num_torus_flow,
        )
        self.natural_freq = natural_freq

        self.enc = self.enc.to(device)
        self.dec = self.dec.to(device)
        self.dyn = self.dyn.to(device)
        self.device = device

        self.num_iters = num_iters
        self.batch_size = batch_size
        self._optimizer = optim.Adam(
            chain(
                self.enc.parameters(),
                self.dec.parameters(),
                # self.dyn.parameters(),
            ),
            lr=learning_rate,
        )
        self._optimizer_params = dict(lr=learning_rate)

        self.gamma = discount_gamma
        self.latent_noise = latent_noise
        if recon_loss_coef is None:
            self.recon_loss_coef = torch.ones(dim_obs)
        else:
            if isinstance(recon_loss_coef, torch.Tensor):
                self.recon_loss_coef = recon_loss_coef
            else:
                self.recon_loss_coef = torch.Tensor(recon_loss_coef)

        if enable_loss is None:
            self.enable_loss = {
                "recon": True,
                "rec_diff": True,
                "dec": True,
                "dec_diff": True,
                "enc": True,
                "lat": True,
            }
        else:
            self.enable_loss = enable_loss

        self.record = {
            "loss": [],
            "loss_recon": [],
            "loss_rec_diff": [],
            "loss_dec": [],
            "loss_dec_diff": [],
            "loss_enc": [],
            "loss_lat": [],
            "enc_grad": [],
            "dec_grad": [],
            "dyn_grad": [],
        }  # type: ignore

        self.params = dict(
            dim_latent=dim_latent,
            dim_obs=dim_obs,
            num_torus_flow=num_torus_flow,
            hidden_layer_size=hidden_layer_size,
            delta_t=delta_t,
            device=device,
            discount_gamma=discount_gamma,
            learning_rate=learning_rate,
            num_iters=num_iters,
            batch_size=batch_size,
            latent_noise=latent_noise,
            natural_freq=natural_freq,
            amplitude_range=amplitude_range,
            recon_loss_coef=recon_loss_coef,
            enable_loss=enable_loss,
        )

    def learn(self, trajectory: torch.Tensor):
        """
        Learning model from trajectory data.

        trajectory: torch.tensor, (batch_size, time_length, dim_obs)
        """
        if trajectory.ndim != 3:
            raise ValueError(f"trajectory dim size:{trajectory.ndim} must be 3.")

        n_torus = self.num_torus_flow
        time_length = trajectory.shape[1]
        # gamma weight series
        series = self.gamma**torch.arange(time_length, device=self.device)
        series *= 1 - self.gamma
        series = series[None, :]

        # beta weight series for encoder loss
        scale = torch.exp(self.dyn._time_constants[self.num_torus_flow:].detach() * self.dyn._delta_t)
        scale = scale.reshape(-1, 1)
        beta_series = scale**torch.arange(1, time_length, device=self.device)
        beta_series *= (1 - self.gamma * scale) / (1 - self.gamma)
        beta_series = torch.swapaxes(beta_series, 0, 1)

        for itr in range(self.num_iters):
            idx = torch.randint(len(trajectory), size=(self.batch_size,), device=trajectory.device)
            batch = trajectory[idx]

            with torch.no_grad():
                input_batch = batch

            obs_rec, latent_hist = self.rollout(input_batch, time_length, deterministic=False, predict=True)
            # detach computation graph from variation distributions
            # latent_hist = latent_hist.detach()

            latent_pa_dyn = self.dyn.predict(self.delta_t, latent_hist[:, :-1])
            rec_diff = torch.diff(obs_rec, dim=1) / self.delta_t

            # detach computation graph from variation distributions
            latent_direct = self.enc(input_batch)
            latent_direct_noise = torch.randn_like(latent_direct) * self.latent_noise
            direct_rec = self.dec(latent_direct + latent_direct_noise)
            target_diff = torch.diff(batch, dim=1) / self.delta_t

            # Reconstructed Diff Loss: |d * x_k - d * h_inv * A^k h(x0)|
            weight = self.recon_loss_coef

            if self.enable_loss["rec_diff"]:
                loss_rec_diff = torch.abs((rec_diff - target_diff) * weight).mean(dim=-1) * np.sqrt(self.delta_t)
                loss_rec_diff = ((loss_rec_diff * series[:, :-1]).sum(dim=-1).mean())  # average over batch
            else:
                loss_rec_diff = torch.zeros(1, device=self.device)

            # Reconstructed Loss: |x_k - h_inv * A^k * h(x0)|
            loss_recon = torch.abs((obs_rec - batch) * weight).mean(dim=-1)
            loss_recon = (loss_recon * series).sum(dim=-1).mean()  # average over batch

            # Enc Loss: gain * |h(x_k+1) - A * h(x_k)|
            if self.enable_loss["enc"]:
                error = torch.empty_like(latent_pa_dyn)
                error[..., :n_torus] = (latent_pa_dyn[..., :n_torus] - unwrap_func(latent_direct[:, 1:, :n_torus], dim=1))  # phase component
                error[..., n_torus:] = (latent_pa_dyn[..., n_torus:] - latent_direct[:, 1:, n_torus:]) * beta_series  # amplitude components
                error *= self.dyn.gain

                loss_enc = torch.abs(error).mean(dim=-1)
                assert loss_enc.shape == (self.batch_size, time_length - 1)
                loss_enc = (loss_enc * series[:, 1:]).sum(dim=-1).mean()
            else:
                loss_enc = torch.zeros(1, device=self.device)

            # Dec Loss: |x - h_inv * h(x)|
            if self.enable_loss["dec"]:
                weight = self.recon_loss_coef
                loss_dec = torch.abs((direct_rec - batch) * weight).mean(dim=-1)
                assert loss_dec.shape == (self.batch_size, time_length)
                loss_dec = loss_dec.mean()
                loss_dec = (loss_dec * series).sum(dim=-1).mean()
            else:
                loss_dec = torch.zeros(1, device=self.device)

            # Dec Diff Loss: |d * s - d * h_inv * h(s)|
            if self.enable_loss["dec_diff"]:
                weight = self.recon_loss_coef
                direct_diff = torch.diff(direct_rec, dim=1) / self.delta_t
                err_dec_diff = (direct_diff - target_diff) * weight
                loss_dec_diff = torch.abs(err_dec_diff).mean(dim=-1) * np.sqrt(self.delta_t)
                loss_dec_diff = ((loss_rec_diff * series[:, :-1]).sum(dim=-1).mean())  # average over batch
            else:
                loss_dec_diff = torch.zeros(1, device=self.device)

            # Loss Latent: |h(x_k+1) - A * h(x_k)|
            if self.enable_loss["lat"]:
                latent_direct_pred = self.dyn.predict(self.delta_t, (latent_direct + latent_direct_noise)[:, :-1])
                error = torch.empty_like(latent_direct_pred)  # phase and amp
                error[..., :n_torus] = (unwrap_func(latent_direct_pred[..., :n_torus], dim=1) - unwrap_func(latent_direct[..., 1:, :n_torus], dim=1))  # phase component
                error[..., n_torus:] = (latent_direct_pred[..., n_torus:] - latent_direct[:, 1:, n_torus:]) * beta_series  # amplitude components
                error *= (2 - self.dyn.gain)
                loss_lat = torch.abs(error).mean(dim=-1)
                assert loss_lat.shape == (self.batch_size, time_length - 1)
                loss_lat = (loss_lat * series[:, 1:]).sum(dim=-1).mean()
            else:
                loss_lat = torch.zeros(1, device=self.device)

            loss = (
                # loss_recon + loss_enc + loss_dec + loss_lat
                loss_rec_diff + loss_dec_diff + loss_recon + loss_enc + loss_dec + loss_lat  # + loss_struct_recon
            )

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            with torch.no_grad():
                self.record["loss"].append(loss.detach())
                self.record["loss_recon"].append(loss_recon.detach())
                self.record["loss_rec_diff"].append(loss_rec_diff.detach())
                self.record["loss_dec"].append(loss_dec.detach())
                self.record["loss_dec_diff"].append(loss_dec_diff.detach())
                self.record["loss_enc"].append(loss_enc.detach())
                self.record["loss_lat"].append(loss_lat.detach())

                self.record["enc_grad"].append(
                    np.mean([p.grad.norm().detach().cpu() for p in self.enc.parameters() if p.grad is not None]))
                self.record["dec_grad"].append(
                    np.mean([p.grad.norm().detach().cpu() for p in self.dec.parameters() if p.grad is not None]))
                self.record["dyn_grad"].append(
                    np.mean([p.grad.norm().detach().cpu() for p in self.dyn.parameters() if p.grad is not None]))

                # print("step:", itr, ",loss:", loss.detach().cpu().item())

    def logging(self,):
        return self.record

    def rollout(self, obs, episode_length, deterministic=True, predict=True):
        batch_size = obs.shape[0]
        dim_obs = obs.shape[-1]
        obs_rec = torch.empty(
            (batch_size, episode_length, dim_obs),
            dtype=obs.dtype,
            device=obs.device,
        )
        dim_latent = self.dim_latent
        latent_hist = torch.empty(
            (batch_size, episode_length, dim_latent),
            dtype=obs.dtype,
            device=obs.device,
        )
        if deterministic:
            self.enc.eval()
            self.dec.eval()
            self.dyn.eval()
        enc_latent = self.enc(obs)
        latent_0 = enc_latent[:, 0:1].clone()
        if not deterministic:
            latent = latent_0 + torch.randn_like(latent_0) * self.latent_noise
        else:
            latent = latent_0
        rec = obs[:, 0:1]
        for i in range(episode_length):
            latent_hist[:, i:i + 1] = latent
            rec = self.dec(latent)
            obs_rec[:, i:i + 1] = rec

            # predict
            t = self.delta_t * (i + 1)
            if predict:
                latent = self.dyn(t, latent_0)
            elif i == episode_length - 1:
                latent = self.dyn(self.delta_t, latent)
            else:
                latent = self.dyn(self.delta_t, latent, ref_latent=enc_latent[:, i + 1:i + 2])
            if not deterministic:
                latent = latent + torch.randn_like(latent) * self.latent_noise

        if deterministic:
            self.enc.train()
            self.dec.train()
            self.dyn.train()
            obs_rec = obs_rec.detach()
            latent_hist = latent_hist.detach()
        return obs_rec, latent_hist

    def save(self, path):
        torch.save(
            dict(
                optimizer_state_dict=self._optimizer.state_dict(),
                encoder_state_dict=self.enc.state_dict(),
                decoder_state_dict=self.dec.state_dict(),
                dynamics_state_dict=self.dyn.state_dict(),
                params=self.params,
                record=self.record,
            ),
            path,
        )

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=False)

        params = checkpoint["params"]
        params["device"] = "cpu"
        self = cls(**params)
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.enc.load_state_dict(checkpoint["encoder_state_dict"])
        self.dec.load_state_dict(checkpoint["decoder_state_dict"])
        self.dyn.load_state_dict(checkpoint["dynamics_state_dict"])
        self.record = checkpoint["record"]
        return self

    def to(self, device):
        self.enc = self.enc.to(device)
        self.dec = self.dec.to(device)
        self.dyn = self.dyn.to(device)


class Decoder(nn.Module):

    def __init__(self, 
                 dim_latent: int, 
                 dim_obs: int, 
                 num_torus_flow: int,
                 hidden_layer_size: List[int]
                 ):
        super().__init__()
        relu = nn.ReLU
        self.net = make_mlp(
            [
                dim_latent + num_torus_flow,
            ] + hidden_layer_size + [
                dim_obs,
            ],
            relu,
            is_norm_layer=False,
        )
        self.register_parameter("phase_rescale", nn.Parameter(torch.ones(1)))

        self.dim_latent = dim_latent
        self.dim_obs = dim_obs
        self.num_torus_flow = num_torus_flow

        self.params = dict(
            dim_latent=dim_latent,
            dim_obs=dim_obs,
            num_torus_flow=num_torus_flow,
            hidden_layer_size=hidden_layer_size,
        )

    def forward(self, latent):
        is_reshape = False
        if latent.ndim == 3:  # for batch norm
            batch_size, length = latent.shape[0:2]
            latent = latent.reshape(-1, latent.shape[-1])
            is_reshape = True
        torus_latent = []  # type: List[torch.Tensor]
        for i in range(self.num_torus_flow):
            cos_x = torch.cos(self.phase_rescale * latent[..., i:i + 1])
            sin_x = torch.sin(self.phase_rescale * latent[..., i:i + 1])
            torus_latent.append(cos_x)
            torus_latent.append(sin_x)
        torus_latent.append(latent[..., self.num_torus_flow:])
        latent = torch.cat(torus_latent, dim=-1)
        out = self.net(latent)
        if is_reshape:  # for batch norm
            out = out.reshape(batch_size, length, -1)
        return out


class Encoder(nn.Module):

    def __init__(self, 
                 dim_latent: int, 
                 dim_obs: int, 
                 num_torus_flow: int,
                 hidden_layer_size: List[int],
                ):
        super().__init__()
        self.num_torus_flow = num_torus_flow
        relu = nn.ReLU
        self.net = make_mlp(
            [
                dim_obs,
            ] + hidden_layer_size + [
                dim_latent + num_torus_flow,
            ],
            relu,
            is_norm_layer=False,
        )
        self.register_parameter("phase_scale", nn.Parameter(torch.ones(1)))
        self.params = dict(
            dim_latent=dim_latent,
            dim_obs=dim_obs,
            num_torus_flow=num_torus_flow,
            hidden_layer_size=hidden_layer_size,
        )

    def forward(self, obs):
        is_reshape = False
        n_torus = self.num_torus_flow
        if obs.ndim == 3:  # for batch norm
            batch_size, length = obs.shape[0:2]
            obs = obs.reshape(-1, obs.shape[-1])
            is_reshape = True
        latent = self.net(obs)
        latent_variables = []
        for i in range(n_torus):
            base_phi = torch.atan2(latent[..., 2 * i + 1:2 * i + 2], latent[..., 2 * i:2 * i + 1])
            phi = self.phase_scale * base_phi
            latent_variables.append(phi)
        z = latent[..., 2 * n_torus:]
        latent_variables.append(z)
        latent = torch.cat(latent_variables, dim=-1)
        if is_reshape:  # for batch norm
            latent = latent.reshape(batch_size, length, -1)
        return latent


class PADynamics(nn.Module):

    def __init__(
        self,
        dim_latent: int = 10,
        delta_t: float = 0.1,
        natural_freq: Union[float, List[float]] = 1.0,
        amplitude_range: Optional[Sequence]=None,
        num_torus_flow: int = 1,
    ):
        super().__init__()
        if dim_latent < 2:
            raise ValueError(f"dim_latent:{dim_latent} must be greater than 1.")
        self._dim_latent = dim_latent

        # time constants
        _time_constants = torch.empty(dim_latent)
        # for linear flow on the N-dim torus
        if num_torus_flow > dim_latent:
            raise ValueError(f"num_torus_flow:{num_torus_flow} must be less than dim_latent:{dim_latent}.")
        if num_torus_flow == 1:
            if isinstance(natural_freq, list):
                natural_freq = natural_freq[0] 
            _time_constants[0] = 2 * np.pi * natural_freq  # rad / s
        elif num_torus_flow < 0 or not isinstance(natural_freq, list):
            raise ValueError(f"num_torus_flow:{num_torus_flow} must be positive integer over natural_freq num.:{natural_freq}.")
        else:
            for i in range(num_torus_flow):
                _time_constants[i] = 2 * np.pi * natural_freq[i]
        # end time constants

        # numerical stable range: 0 < time_constants < 2 / delta_t
        if amplitude_range is None:
            amplitude_range = [0.01, 25.0]
        _time_constants[num_torus_flow:] = -torch.logspace(
            np.log10(amplitude_range[0]),
            np.log10(amplitude_range[1]),
            dim_latent - num_torus_flow,
        )
        self.register_buffer("_time_constants", _time_constants)
        self._logit_gain = torch.nn.Parameter(torch.ones(dim_latent) * (-0.0))

        self._delta_t = delta_t
        self.num_torus_flow = num_torus_flow # type: int
        self.params = dict(
            dim_latent=dim_latent,
            delta_t=delta_t,
            natural_freq=natural_freq,
            amplitude_range=amplitude_range,
            num_torus_flow=num_torus_flow,
        )

    def predict(self, t, latent):
        n_torus = self.num_torus_flow
        next_latent = torch.empty_like(latent)
        next_latent[..., :n_torus] = latent[..., :n_torus] + self._time_constants[:n_torus] * t
        next_latent[..., n_torus:] = latent[..., n_torus:] * torch.exp(self._time_constants[n_torus:] * t)
        return next_latent

    def filter_process(self, latent, ref_latent):
        k = self.gain
        n_torus = self.num_torus_flow
        latent[..., :n_torus] = latent[..., :n_torus] + short_path(latent[..., :n_torus], ref_latent[..., :n_torus], scale=k[:n_torus])
        latent[..., n_torus:] = latent[..., n_torus:] + k[n_torus:] * (ref_latent[..., n_torus:] - latent[..., n_torus:])
        return latent

    def forward(self, time, latent_0, ref_latent=None):
        if latent_0.ndim == 1:
            latent_0 = latent_0.unsqueeze(dim=0)

        next_latent = self.predict(time, latent_0)

        if ref_latent is not None:
            if ref_latent.ndim == 1:
                ref_latent = ref_latent.unsqueeze(dim=0)
            self.filter_process(next_latent, ref_latent)
        return next_latent

    @property
    def gain(self,):
        return torch.sigmoid(self._logit_gain)


def wrap_func(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def short_path(th1, th2, scale=0.5):
    """
    Interpolate between two angles with the shortest path.

    th = th1 + s * (th2 - th1)
    if |th2 - th1| > pi, select shorter path
    """
    th1 = wrap_func(th1)
    th2 = wrap_func(th2)

    delta = th2 - th1
    idx_1 = delta > np.pi
    idx_2 = delta < -np.pi
    delta[idx_1] -= 2 * np.pi
    delta[idx_2] += 2 * np.pi
    return scale * delta


def unwrap_func(theta: torch.Tensor, dim: int=0) -> torch.Tensor:
    """unwrap phase tensor.

    Parameters
    ----------
    theta : torch.Tensor
    dim : int, optional
        dimension to unwrap, by default 0

    Returns
    -------
    torch.Tensor: unwrapped phase tensor
    """
    phi = torch.diff(theta, dim=dim)
    idx = torch.abs(phi) > np.pi
    phi[idx] = wrap_func(phi[idx])
    phi = torch.cumsum(phi, dim=dim)
    init = torch.select(theta, dim=dim, index=0)
    init = init.unsqueeze(dim)
    phi = torch.cat([init, init + phi], dim=dim)
    return phi


def make_mlp(
        layer_size: List[int],
        activate_function: Callable,
        last_activate_function: Optional[Callable] = None,
        is_norm_layer: bool = False,
        ) -> nn.Module:
    layers = OrderedDict([])  # type: OrderedDict[str, Any]
    if len(layer_size) < 2:
        return nn.Sequential(layers)
    dim_input = layer_size[0]
    for i, h in enumerate(layer_size[1:-1]):
        layers[f"layer_{i}"] = nn.Linear(dim_input, h, bias=True)
        activate_func_name = str(activate_function()).lower().rstrip("()")
        nn.init.xavier_uniform_(layers[f"layer_{i}"].weight.data, gain=nn.init.calculate_gain(activate_func_name))
        layers[f"layer_{i}"].bias.data.fill_(0)
        if is_norm_layer:
            layers[f"norm_{i}"] = nn.BatchNorm1d(h)
        layers[f"ActivateFunc_{i}"] = activate_function()
        dim_input = h
    layers["layer_last"] = nn.Linear(dim_input, layer_size[-1], bias=True)
    if last_activate_function is None:
        last_activate_function = nn.Identity
    else:
        activate_func_name = str(last_activate_function()).lower().rstrip("()")
        nn.init.xavier_uniform_(layers["layer_last"].weight.data, gain=nn.init.calculate_gain(activate_func_name))
    layers["layer_last"].bias.data.fill_(0)
    layers["ActiveFunc_last"] = last_activate_function()
    return nn.Sequential(layers)
