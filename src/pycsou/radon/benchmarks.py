import time

import numpy as np
import tqdm
from disk_func import Disk
from gaussian_func import TruncatedGaussian
from matplotlib import pyplot as plt
from radon_op import RadonOp


def plot_benchmark(time_steps, run_start, run_end, run_step, title):
    xs = np.arange(run_start, run_end, run_step)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.scatter(xs, time_steps)
    plt.title(label=title)
    fig.savefig("../../../outputs/" + title + ".png")


def time_run(radon_op, alphas, time_steps, k):
    start = time.time()
    _ = radon_op.applyF(alphas)
    end = time.time()
    time_steps[k] = end - start


def benchmark(delta, psi, n, t, eps, alphas, n_runs, t_runs, title_prefix, **kwargs):
    uniform_bool = type(delta) == dict

    n_runstart, n_runend, n_step = n_runs["start"], n_runs["stop"], n_runs["step"]
    t_runstart, t_runend, t_step = t_runs["start"], t_runs["stop"], t_runs["step"]

    n_timesteps = np.zeros(int((n_runend - n_runstart) / n_step))
    k = 0
    for run in tqdm.trange(n_runstart, n_runend, n_step):
        if uniform_bool:
            thetas = {"start": n["start"], "stop": n["stop"], "num": run}
            radon_op = RadonOp.uniform(delta, psi, thetas, t, eps=eps, **kwargs)
        else:
            thetas = np.linspace(start=np.min(n), stop=np.max(n), num=run)
            radon_op = RadonOp.nonuniform(delta, psi, thetas, t, eps=eps, **kwargs)
        time_run(radon_op, alphas, n_timesteps, k)
        k += 1

    t_timesteps = np.zeros(int((t_runend - t_runstart) / t_step))
    k = 0
    for run in tqdm.trange(t_runstart, t_runend, t_step):
        ts = {"start": t["start"], "stop": t["stop"], "num": run}
        generator = RadonOp.uniform if uniform_bool else RadonOp.nonuniform
        radon_op = generator(delta, psi, n, ts, eps=eps, **kwargs)
        time_run(radon_op, alphas, t_timesteps, k)
        k += 1

    title_prefix += "_uniform" if uniform_bool else "_nonuniform"
    plot_benchmark(n_timesteps, n_runstart, n_runend, n_step, title_prefix + "_incr_n")
    plot_benchmark(t_timesteps, t_runstart, t_runend, t_step, title_prefix + "_incr_t")


sigma_gaussian = 0.3
radius_disk = 1.0

gaussian = TruncatedGaussian(sigma_gaussian)
disk = Disk(radius_disk)

epsilons = [1e-3, 1e-6]

sparse_deltas_x = 0.3 * np.array([1, 0.5, -0.5])
sparse_deltas_y = 0.3 * np.array([0, 0.25, -0.5])
sparse_deltas = np.array(list(zip(sparse_deltas_x, sparse_deltas_y)))

grid_deltas = {"start": [-1024, -1024], "stop": [1024, 1024], "num": [2048, 2048]}

sparse_n = np.linspace(start=0, stop=2 * np.pi, num=20)

grid_n = {"start": 0, "stop": 2 * np.pi, "num": 2000}

n_run = {"start": 1500, "stop": 2500, "step": 20}

t = {"start": -1024, "stop": 1024, "num": 2000}

t_run = {"start": 1500, "stop": 2500, "step": 20}

gridded_alphas = np.random.randn(np.prod(grid_deltas["num"]))
sparse_alphas = np.ones(len(sparse_deltas))

if __name__ == "__main__":
    for eps in epsilons:

        # benchmark(
        #     grid_deltas,
        #     disk,
        #     grid_n,
        #     t,
        #     eps,
        #     gridded_alphas,
        #     n_run,
        #     t_run,
        #     f"disk_eps={eps}"
        # )
        #
        # benchmark(
        #     sparse_deltas,
        #     disk,
        #     sparse_n,
        #     t,
        #     eps,
        #     sparse_alphas,
        #     n_run,
        #     t_run,
        #     f"disk_eps={eps}"
        # )

        print(f"\nGaussian uniform, eps={eps}")
        benchmark(grid_deltas, gaussian, grid_n, t, eps, gridded_alphas, n_run, t_run, f"gaussian_eps={eps}")

        # print(f"\nGaussian non uniform, eps={eps}")
        # benchmark(sparse_deltas, gaussian, sparse_n, t, eps, sparse_alphas, n_run, t_run, f"gaussian_eps={eps}")
