# Plots
from utils import *
def Myplot1(FCLPF_y, Fref_y, GCLPF_y, Gref_y):
    # Plots for Van-der-pol Equation.
    create_directory_if_not_exists("./Plots")
    t1 = np.linspace(0, 1, FCLPF_y.shape[1])
    t2 = np.linspace(0, 1, Fref_y.shape[2])

    GCLPF_y, Gref_y = GCLPF_y[:, 100:, ], Gref_y[:, :, 2000:]
    t3 = np.linspace(1, 2, GCLPF_y.shape[1])
    t4 = np.linspace(1, 2, Gref_y.shape[2])

    indices = np.where(np.isin(t2, t1))[0]  # Find the indices of the same time points
    RMSEs = np.zeros((FCLPF_y.shape[0], Fref_y.shape[0]))
    for i in range(FCLPF_y.shape[0]):
        for j in range(Fref_y.shape[0]):
            RMSEs[i, j] = np.mean((Fref_y[j, :, indices] - FCLPF_y[i, :, :]) ** 2)
    i, j = np.where(RMSEs == RMSEs.min())   # j is Ref;i is CLPF。

    sigsizes = (12, 6)
    fsize = 26
    labelsize = 20
    # Plot Fit
    plt.figure(figsize=sigsizes)
    plt.plot(t1, FCLPF_y[i, :, 0].T, c='blue', label="Extended CLPF solution")
    plt.plot(t2, Fref_y[j, 0, :].T, c='red', label="Reference solution")
    plt.legend(loc='lower left', prop={'size': labelsize})
    plt.xlabel("t", fontsize=fsize)
    plt.ylabel("u\u2081", fontsize=fsize)
    plt.tight_layout()
    plt.savefig("./Plots/VDPfit_u1.eps", dpi=600)
    plt.show()

    plt.figure(figsize=sigsizes)
    plt.plot(t1, FCLPF_y[i, :, 1].T, c='blue', label="Extended CLPF solution")
    plt.plot(t2, Fref_y[j, 1, :].  T, c='red', label="Reference solution")
    plt.legend(loc='lower right', prop={'size': labelsize})
    plt.xlabel("t", fontsize=fsize)
    plt.ylabel("u\u2082", fontsize=fsize)
    plt.tight_layout()
    plt.savefig("./Plots/VDPfit_u2.eps", dpi=600)
    plt.show()

    # Plot Generate
    indices = np.where(np.isin(t2, t1))[0]  # Find the indices of the same time points
    RMSEs = np.zeros((GCLPF_y.shape[0], Gref_y.shape[0]))
    for i in range(GCLPF_y.shape[0]):
        for j in range(Gref_y.shape[0]):
            RMSEs[i, j] = np.mean((Gref_y[j, :, indices] - GCLPF_y[i, :, :]) ** 2)
    Is, Js = MinstN(RMSEs, 10)
    uCLPF = np.mean(GCLPF_y[Is, :, :], axis=0).T
    uref = np.mean(Gref_y[Js, :, :], axis=0)

    # Compute
    indices = np.where(np.isin(t4, t3))[0]  # Find the indices of the same time points
    detaT = 1 / (len(indices)-1)
    Gen_err1 = (np.sum((uCLPF[0, :] - uref[0, indices])**2 + (uCLPF[1, :] - uref[1, indices])**2) * detaT) ** .5
    print(f"-Gen-Err:{Gen_err1} ")

    plt.figure(figsize=sigsizes)
    plt.plot(t3, uCLPF[0, :], c='blue', label="Sample mean of the extended CLPF solution")
    plt.plot(t4, uref[0, :], c='red', label="Sample mean of the reference solution")
    plt.legend(loc='upper right', prop={'size': labelsize})
    plt.xlabel("t", fontsize=fsize)
    plt.ylabel("u\u2081", fontsize=fsize)
    plt.tight_layout()
    plt.savefig("./Plots/VDPGen_u1.eps", dpi=600)

    plt.figure(figsize=sigsizes)
    plt.plot(t3, uCLPF[1, :], c='blue', label="Sample mean of the extended CLPF solution")
    plt.plot(t4, uref[1, :], c='red', label="Sample mean of the reference solution")
    plt.legend(loc='upper right', prop={'size': labelsize})
    plt.xlabel("t", fontsize=fsize)
    plt.ylabel("u\u2082", fontsize=fsize)
    plt.tight_layout()
    plt.savefig("./Plots/VDPGen_u2.eps", dpi=600)
    plt.show()

def Myplot2(FCLPF_y, Fref_y, GCLPF_y, Gref_y):
    # Plots for Ginzburg-Landau Equation.
    create_directory_if_not_exists("./Plots")
    t1 = np.linspace(0, 1, len(FCLPF_y.T))
    t2 = np.linspace(0, 1, len(Fref_y.T))

    idxs = np.where(np.isin(t2, t1))[0]
    ref_y = Fref_y[0, idxs]
    RMSEs = [np.sum((ref_y-FCLPF_y[i, :])**2) for i in range(len(Fref_y))]

    I = np.argmin(RMSEs)
    plt.plot(t1, FCLPF_y[I, :], c='blue', label="Extended CLPF solution")
    plt.plot(t2, Fref_y[0, :], c='red', label="Reference solution")
    plt.xlabel("t", fontsize=12)
    plt.ylabel("u", fontsize=12)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("./Plots/LDfit.eps", dpi=600)
    plt.show()


    GCLPF_y, Gref_y = GCLPF_y[100:], Gref_y[2000:]
    t3 = np.linspace(1, 2, len(GCLPF_y))
    t4 = np.linspace(1, 2, len(Gref_y))

    idxs = np.where(np.isin(t4, t3))[0]
    RMSEs = np.zeros((len(Gref_y.T), len(GCLPF_y.T)))
    for i in range(len(Gref_y.T)):
        for j in range(len(GCLPF_y.T)):
            RMSEs[i, j] = np.sum((Gref_y[idxs, i] - GCLPF_y[:, j]) ** 2)
    I, J = MinstN(RMSEs, 10)
    ref_y = np.mean(Gref_y[:, I], axis=1)
    gen_y = np.mean(GCLPF_y[:, J], axis=1)

    plt.figure()
    plt.plot(t3, gen_y, c='blue', label="Sample mean of the extended CLPF solution")
    plt.plot(t4, ref_y, c='red', label="Sample mean of the reference solution")
    plt.legend(loc='upper left')
    plt.xlabel("t", fontsize=12)
    plt.ylabel("u", fontsize=12)
    plt.tight_layout()
    plt.savefig("./Plots/LDgen.eps", dpi=600)
    plt.show()

    indices = np.where(np.isin(t4, t3))[0]
    detaT = 1 / (len(indices) - 1)
    Gen_err = (np.sum((ref_y[indices] - gen_y) ** 2) * detaT) ** .5
    print(f"G_Value（Ginzburg-Landau Eq）:{Gen_err} ")
