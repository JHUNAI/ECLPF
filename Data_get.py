# Get data
"""
Fun Define：
    Get_3c_data： Generate Ginzburg-Landau equation
    Get_van_der_pol_data： Generate Van-Der-Pol equation
"""
from scipy.optimize import fsolve
import tqdm
from utils import *
import matplotlib.pyplot as plt


class Get_3c_data():
    #   theta-scheme to generate the primal data (Ginzburg-Landau equation)
    def __init__(self, batch_num, theta2=1.0):
        u0 = 1
        # r = -8
        # sigma = 3
        T = 1
        N = 100
        d = 1
        m = 1

        # Functions
        def fhandle(u):
            return 100 * u * (1 - u**2)

        def ghandle(u):
            return 0.5 * u

        def dghandle(u):
            return 0.5

        # 模拟
        def EulerMilsteinTheta(u0, T0, T, N, d, m, fhandle, ghandle, dghandle, theta):
            dt = T / N
            t = np.linspace(T0, T, N + 1)
            u = np.zeros(N + 1)
            u[0] = u0
            for i in range(N):
                dW = np.random.normal(0, np.sqrt(dt))
                u[i + 1] = u[i] + fhandle(u[i]) * dt + theta * ghandle(u[i]) * dW + 0.5 * theta * dghandle(u[i]) * (dW**2 - dt)
            return t, u

        # self.t1, self.u1 = EulerMilsteinTheta(u0, T, N, d, m, fhandle, ghandle, dghandle, theta1)
        self.t = np.linspace(0, T, N + 1)
        self.u1 = np.zeros((batch_num, N + 1))  # Numerical solutions
        self.urefs = np.zeros((batch_num, N * 20 + 1))  # Reference solutions
        self.ufore = np.zeros((batch_num, N * 40 + 1))

        for i in tqdm.tqdm(range(batch_num)):
            np.random.seed(i)       # Set seed
            _, self.u1[i, :] = EulerMilsteinTheta(u0, 0, T, N, d, m, fhandle, ghandle, dghandle, theta2)
            _, self.urefs[i, :] = EulerMilsteinTheta(u0, 0, T, 20 * N, d, m, fhandle, ghandle, dghandle, theta2)
            _, self.ufore[i, :] = EulerMilsteinTheta(u0, 0, 2 * T, 40 * N, d, m, fhandle, ghandle, dghandle, theta2)

    def data(self):
        # Return: time ,Numerical solutions, Reference solutions, Reference solutions
        return self.t, self.u1, self.urefs, self.ufore


class Get_van_der_pol_data():
    #   theta-scheme to generate the primal data (Van_Der_Pol_ equation)
    def __init__(self, T=1, N=100, batch_size=1000, theta=1.0):
        super().__init__()
        lambda_val = 1
        alpha = 1
        sigma = 1

        u0 = np.array([0.5, 0])

        def fhandle(u):
            return np.array([u[1], -u[1] * (lambda_val + u[0] ** 2) + alpha * u[0] - u[0] ** 3])

        def ghandle(u):
            return np.array([sigma * u[1], sigma * u[0]])

        def dghandle(u):
            return np.array([0, 0])

        def EulerMaruyamaTheta_VDP(u0, t0, t1, N, d, m, fhandle, ghandle, dghandle, theta):

            Dt = T / N  # time interval
            u = np.zeros((d, N + 1))  #
            t = np.linspace(t0, t1, N + 1)  # solve dim
            sqrtDt = np.sqrt(Dt)

            u[:, 0] = u0
            u_n = u0

            for n in range(1, N + 1):
                dw = sqrtDt * np.random.randn(m)
                u_explicit = u_n + Dt * fhandle(u_n) + ghandle(u_n) * dw + 0.5 * (dghandle(u_n) * ghandle(u_n)) * (
                            dw ** 2 - Dt)

                if theta > 0:
                    v = u_n + (1 - theta) * Dt * fhandle(u_n) + ghandle(u_n) * dw + 0.5 * (
                                dghandle(u_n) * ghandle(u_n)) * (
                                dw ** 2 - Dt)
                    u_new = fsolve(lambda u: -u + v + theta * fhandle(u) * Dt, u_explicit)
                else:
                    u_new = u_explicit

                u[:, n] = u_new
                u_n = u_new
            return t, u
        # self.t, self.u = euler_milstein_theta(u0, T, N, 2, 1, fhandle, ghandle, dghandle, theta)
        self.t = np.linspace(0, T, N + 1)
        self.u1 = np.zeros((batch_size, 2, N + 1))
        self.urefs = np.zeros((batch_size, 2, N * 20 + 1))
        self.ufore = np.zeros((batch_size, 2, 40 * N + 1))

        for i in tqdm.tqdm(range(batch_size)):
            np.random.seed(i)
            self.t, self.u1[i, :, :] = EulerMaruyamaTheta_VDP(u0, 0, 1, N, 2, 1, fhandle, ghandle, dghandle, theta)
            _, self.urefs[i, :, :] = EulerMaruyamaTheta_VDP(u0, 0, 1, 20 * N, 2, 1, fhandle, ghandle, dghandle, theta)
            _, self.ufore[i, :, :] = EulerMaruyamaTheta_VDP(u0, 0, 2, 40 * N, 2, 1, fhandle, ghandle, dghandle, theta)

    def plot(self):
        ys_ = self.urefs
        errs = [np.abs(ys_[i, 0, -1] - ys_[i, 0, 0]) + np.abs(ys_[i, 1, -1]
                 - ys_[i, 0, 0]) for i in range(ys_.shape[0])]
        indx = np.where(errs == np.min(errs))[0]
        plt.plot(ys_[indx, 0, :].T, ys_[indx, 1, :].T)
        plt.show()
        plt.figure(figsize=(10, 5))
        plt.subplot(311)
        plt.plot(self.t, self.u1[indx, 0, :].T, label="u0")

        plt.subplot(312)
        plt.plot(self.t, self.u1[indx, 1, :].T, label="u1")

        plt.subplot(313)
        plt.plot(self.u1[indx, 0, :].T, self.u1[indx, 1, :].T, label="u0_u1")
        plt.legend()
        plt.show()

    def data(self):
        return self.t, self.u1, self.urefs, self.ufore



