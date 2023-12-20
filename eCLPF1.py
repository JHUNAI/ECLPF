# Example of the Ginzburg-Landau Equation.
import logging
import math
from typing import Optional, Union
import tqdm
from torch import distributions, nn, optim
import torchsde
from Data_get import Get_3c_data
from utils import *
from Myplots import Myplot2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rtol = 1e-3
atol = 1e-3


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


class EMAMetric(object):
    def __init__(self, gamma: Optional[float] = .99):
        super(EMAMetric, self).__init__()
        self._val = 0.
        self._gamma = gamma

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        self._val = self._gamma * self._val + (1 - self._gamma) * x
        return self._val

    @property
    def val(self):
        return self._val


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


class eCLPF(torchsde.SDEIto):
    def __init__(self, theta=1.0, mu=0.0, sigma=0.5):
        super(eCLPF, self).__init__(noise_type="diagonal")
        logvar = math.log(sigma ** 2 / (2. * theta))

        # Prior drift.
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("mu", torch.tensor([[mu]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))

        # p(y0).
        self.register_buffer("py0_mean", torch.tensor([[mu]]))
        self.register_buffer("py0_logvar", torch.tensor([[logvar]]))

        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        self.net = nn.Sequential(
            nn.Linear(1, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        )
        # Initialization trick from Glow.
        self.net[-1].weight.data.fill_(0.)
        self.net[-1].bias.data.fill_(0.)

        # q(y0).
        self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=True)

    def f(self, t, y):  # Approximate posterior drift.
        if t.dim() == 0:
            t = torch.full_like(y, fill_value=t)
        # Positional encoding in transformers for time-inhomogeneous posterior.
        return self.net(y)

    def g(self, t, y):  # Shared diffusion.
        return self.sigma.repeat(y.size(0), 1)

    def h(self, t, y):  # Prior drift.
        return self.theta * (self.mu - y)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, 0:1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, 0:1]
        g = self.g(t, y)
        g_logqp = torch.zeros_like(y)
        return torch.cat([g, g_logqp], dim=1)

    def forward(self, ts, batch_size, eps=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_std) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).

        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)
        aug_ys = torchsde.sdeint(
            sde=self,
            y0=aug_y0,
            ts=ts,
            method='euler',
            dt=ts[1] - ts[0],
            adaptive=False,
            rtol=rtol,
            atol=atol,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        ys, logqp_path = aug_ys[:, :, 0:1], aug_ys[-1, :, 1]
        logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        return ys, logqp

    def sample_p(self, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.py0_mean) if eps is None else eps
        y0 = self.py0_mean + eps * self.py0_std
        return torchsde.sdeint(self, y0, ts, bm=bm, method='srk', dt=ts[1] - ts[0], names={'drift': 'h'})

    def sample_q(self, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_mean) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        return torchsde.sdeint(self, y0, ts, bm=bm, method='srk', dt=ts[1] - ts[0])

    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)


def main(
        Sample_size=100,
        kl_anneal_iters=1000,
        train_iters=500,
        pause_iters=50,
        img_path="",
        seeds=123,
):
    manual_seed(seeds)
    # Calculate initial input data by drift-theta scheme.
    data1 = Get_3c_data(Sample_size, theta2=1.0)
    ts_ext_, ys_, yrefs, yfore = data1.data()

    ts_vis_ = np.linspace(0, 2, 300)
    ts_, ys_ = ts_ext_, ys_.T
    ts, ts_ext, ts_vis, ys = \
        torch.FloatTensor(ts_).to(device), torch.FloatTensor(ts_ext_).to(device), \
            torch.FloatTensor(ts_vis_).to(device), torch.FloatTensor(ys_).to(device)

    vis_idx = np.random.choice(Sample_size)

    eps = torch.randn(1, 1).to(device)  # Fix seed for the random draws used in the plots.
    bm = torchsde.BrownianInterval(
        t0=ts_vis[0],
        t1=ts_vis[-1],
        size=(1, 1),
        device=device,
        levy_area_approximation='space-time'
    )

    # Model.
    model = eCLPF().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.997)
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

    logpy_metric = EMAMetric()
    kl_metric = EMAMetric()
    loss_metric = EMAMetric()

    for global_step in tqdm.tqdm(range(train_iters)):
        # Plot and save.
        if global_step % pause_iters == 0:
            with torch.no_grad():
                zs = model.sample_q(ts=ts_vis, batch_size=1, eps=eps, bm=bm).squeeze()
                # samples = zs[:, vis_idx]
                ts_vis_, zs_, = ts_vis.cpu().numpy(), zs.cpu().numpy()
                plt.subplot(frameon=False)

                # plt.scatter(ts_, ys_[:, 0], marker='x', zorder=3, color='k', s=35)  # Data.
                plt.plot(ts_vis_, zs_, '-', color='blue', alpha=.5)
                plt.plot(ts_, ys_[:, 0], '-', color='r', alpha=.5)
                plt.axvline(x=ts_vis_[int(.5 * len(ts_vis_))], color='k', linestyle='--')
                plt.xlabel('$t$')
                plt.ylabel('$Y_t$')
                plt.tight_layout()
                # plt.savefig(img_path, dpi=300)
                plt.show()
                # logging.info(f'Saved figure at: {img_path}')

        # Training.
        optimizer.zero_grad()
        zs, kl = model(ts=ts_ext, batch_size=Sample_size)
        zs = zs.squeeze()
        zs = zs
        likelihood_constructor = distributions.Normal
        likelihood = likelihood_constructor(loc=zs, scale=0.05)
        logpy = likelihood.log_prob(ys).sum(dim=0).mean(dim=0)

        loss = -logpy + kl * kl_scheduler.val
        loss.backward()

        optimizer.step()
        scheduler.step()
        kl_scheduler.step()

        logpy_metric.step(logpy)
        kl_metric.step(kl)
        loss_metric.step(loss)

        logging.info(
            f'global_step: {global_step}, '
            f'logpy: {logpy_metric.val:.4f}, '
            f'kl: {kl_metric.val:.4f}, '
            f'loss: {loss_metric.val:.4f}'
        )

    # Compute Weak-error and Strong-error.
    SWE = Str_Weak_err(types="LD", epi=Sample_size, train_iter=train_iters, info="1")
    T_f = np.linspace(0, 2, len(yfore.T))
    T_fore = torch.FloatTensor(T_f).to(device)
    (zs, _), (zsfore, _) = (model(ts=ts_ext, batch_size=Sample_size),
                            model(ts=T_fore.to(device), batch_size=Sample_size))
    zs_, zsfore_ = zs.squeeze().cpu().detach().numpy(), zsfore.squeeze().cpu().detach().numpy()
    SWE.CompussAndSave(ys=zs_, zs=zsfore_[:2000, :])
    # =====================Show the results========================
    # Fitting and Generalization test.
    FCLPF_t = ts_
    (FCLPF_y, _) = (model(ts=torch.FloatTensor(FCLPF_t).to(device), batch_size=Sample_size))
    FCLPF_y = FCLPF_y.squeeze().cpu().detach().numpy()  #

    GCLPF_t = np.linspace(0, 2, 2 * (len(ts_) - 1) + 1)
    (GCLPF_y, _) = (model(ts=torch.FloatTensor(GCLPF_t).to(device), batch_size=Sample_size))
    GCLPF_y = GCLPF_y.squeeze().cpu().detach().numpy()
    Myplot2(FCLPF_y.T, yrefs, GCLPF_y, yfore.T)
    # Finish.
    print(f"{'=' * 10}Ginzburg-Landau is Over！{'=' * 10}")


if __name__ == '__main__':
    Sample_size = [100, 200, 500, 800, 1000]
    Loop = tqdm.tqdm(Sample_size)
    for S in Loop:
        main(Sample_size=S,
             kl_anneal_iters=1000,
             train_iters=500,
             pause_iters=50,
             img_path="",
             seeds=123)
