#  Example of the Van-der-pol Equation.
import logging
from utils import *
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
from Data_get import Get_van_der_pol_data
import torchsde
from Myplots import Myplot1

class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.LSTM(x)
        x = self.Linear(x)
        return x


class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size):
        super(LatentSDE, self).__init__()
        # Encoder.
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_size),
        )
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid(),
                )
                for _ in range(latent_size)
            ]
        )
        self.projector = nn.Linear(latent_size, data_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))
        self._ctx = None

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, dt, noise_std, adjoint=False, method="euler"):
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=dt, logqp=True, method=method)
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=dt, logqp=True, method=method)

        _xs = self.projector(zs)
        xs_dist = Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        return log_pxs, logqp0 + logqp_path

    @torch.no_grad()
    def sample(self, batch_size, ts, dt, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=dt, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)
        return _xs


def main(
        batch_size=1000,
        show_size=1,
        latent_size=68,
        context_size=128,
        hidden_size=128,
        lr_init=1.0e-2,
        t0=0.,
        t1=1.,
        lr_gamma=1,
        num_iters=1000,
        kl_anneal_iters=200,
        pause_every=50,
        noise_std=0.005,
        adjoint=False,
        method="euler",
        seeds=123):
    # Set seed.
    manual_seed(seeds)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Calculate initial input data by drift-theta scheme.
    data = Get_van_der_pol_data(batch_size=batch_size, theta=1.0)
    t, u, urefs, ufore = data.data()
    N_epi = u.shape[-1]
    dt1 = (t1 - t0) / (N_epi - 1)
    ts = torch.FloatTensor(t)
    xs = torch.FloatTensor(u)
    xs = xs.permute(2, 0, 1)
    ts, xs = ts.to(device), xs.to(device)

    latent_sde = LatentSDE(
        data_size=2,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
    ).to(device)
    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

    # Fix the same Brownian motion for visualization.
    bm_vis = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")

    colors = Mycolor()  # colors
    sample = np.random.randint(batch_size, size=show_size).tolist()  # Pick samples to plot.

    #   Train.
    for global_step in tqdm.tqdm(range(1, num_iters + 1)):
        latent_sde.zero_grad()
        log_pxs, log_ratio = latent_sde(xs, ts, dt1, noise_std, adjoint, method)
        # latent_sde Output：（time，batch—size，solve-dim）
        loss = -log_pxs + log_ratio * kl_scheduler.val
        loss.backward()
        optimizer.step()
        scheduler.step()
        kl_scheduler.step()
        if global_step % pause_every == 0:
            plt.figure()
            plt.subplot(231)
            for i, sam in enumerate(sample):
                plt.plot(t, u[sam, 0, :], color=colors[i])
            plt.title('u1_ture')

            plt.subplot(232)
            for i, sam in enumerate(sample):
                plt.plot(t, u[sam, 1, :], color=colors[i])
            plt.title('u2_ture')

            plt.subplot(233)
            for i, sam in enumerate(sample):
                plt.plot(u[sam, 0, :], u[sam, 1, :], color=colors[i])
            plt.title('(u1,u2)')

            xs_ = latent_sde.sample(batch_size=batch_size, ts=ts, dt=dt1, bm=bm_vis).cpu().numpy()
            # z1, z2 = np.split(xs_, indices_or_sections=2, axis=-1)

            plt.subplot(234)
            for i, sam in enumerate(sample):
                plt.plot(t, xs_[:, sam, 0], color=colors[i])
            plt.title('u1_pre')

            plt.subplot(235)
            for i, sam in enumerate(sample):
                plt.plot(t, xs_[:, sam, 1], color=colors[i])
            plt.title('u2_pre')

            plt.subplot(236)
            for i, sam in enumerate(sample):
                plt.plot(xs_[:, sam, 0], xs_[:, sam, 1], color=colors[i])
            plt.xlabel('lSDE_u(1)=Q')
            plt.ylabel('lSDE_u(2)=P')
            plt.title('(u1.u2)_pre')
            plt.tight_layout()
            plt.show()
            lr_now = optimizer.param_groups[0]['lr']
            logging.warning(
                f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
            )

    # Compute Weak-error and Strong-error.
    bm_fit = torchsde.BrownianInterval(
        t0=0, t1=1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")

    t_fit = np.linspace(0, 1, N_epi)
    t_fit_= torch.FloatTensor(t_fit).to(device)
    zs_fit = latent_sde.sample(batch_size=batch_size, ts=t_fit_, dt=t_fit_[1] - t_fit_[0], bm=bm_fit).cpu().numpy()

    LSW = Str_Weak_err(types="VDP", epi=batch_size, train_iter=num_iters, info="1")
    LSW.CompussAndSave(urefs.transpose(2, 0, 1), zs_fit)

    # ==========================Show results==========================
    # Fitting and Generalization test.
    t_fit = np.linspace(0, 1, len(ts))
    t_gen = np.linspace(0, 2, 2 * (len(ts)) - 1)

    t_fit_, t_gen_ = torch.FloatTensor(t_fit).to(device), torch.FloatTensor(t_gen).to(device)
    # set Bm
    bm_fit = torchsde.BrownianInterval(
        t0=0, t1=1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")
    bm_gen = torchsde.BrownianInterval(
        t0=0, t1=2, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")
    zs_fit = latent_sde.sample(batch_size=batch_size, ts=t_fit_, dt=t_fit_[1]-t_fit_[0], bm=bm_fit).cpu().numpy()
    zs_gen = latent_sde.sample(batch_size=batch_size, ts=t_gen_, dt=t_gen_[1]-t_gen_[0], bm=bm_gen).cpu().numpy()
    zs_fit, zs_gen = zs_fit.transpose(1, 0, 2), zs_gen.transpose(1, 0, 2)
    Myplot1(zs_fit, urefs, zs_gen, ufore)
    print("Van-der-pol is over!!!")


if __name__ == "__main__":
    Sample_size = [100, 200, 500, 800, 1000]
    Loop =tqdm.tqdm(Sample_size)
    for S in Loop:
        main(
            batch_size=S,
            show_size=1,
            latent_size=68,
            context_size=68,
            hidden_size=32,
            lr_init=2.0e-2,
            t0=0.,
            t1=1.,
            lr_gamma=.999,
            num_iters=500,
            kl_anneal_iters=400,
            pause_every=50,
            noise_std=0.004,
            adjoint=False,
            method="euler",
            seeds=123)
