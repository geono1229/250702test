
"""
read_h5_and_plot.py
-------------------
Utilities to read the HDF5 produced by postprocess_grid_to_h5.py and
(1) extract/plot line profiles at fixed x or y for selected timesteps,
(2) plot 2D colormaps of a scalar field.

Usage examples:
  # Plot l(y) at x=0 for t=1.0, 5.0
  python read_h5_and_plot.py --h5 path/to/file.h5 --var l --line x=0.0 --times 1.0 5.0

  # Plot u_mag(y) at x=-10 and x=10 for t nearest to 2.35
  python read_h5_and_plot.py --h5 path/to/file.h5 --var u_mag --line x=-10 10 --times 2.35

  # Colormap of l at t=4.0 with PNG output
  python read_h5_and_plot.py --h5 path/to/file.h5 --var l --colormap --times 4.0 --save fig_l_t4.png
"""

"""
25/10/27 edit by Geono Kim
  1. colormap for multiple time inputs
    new parameter savename define, save plot with different names for each time input
  2. plot_error function
    for all y index iy, calculate relative error of data[iy][ix] and data[iy][ix+1]
    error - x plot, if relative error < tolerance, put converging x_coordinate
    use: python read_h5_and_plot.py --h5 path/to/file.h5 --var l --error --times 0 10 20
  3. plot_converge function
    use absolute tolerance(atol), relative tolerance(rtol) for checking convergence
    tolerance = atol + rtol * data[iy][ix]
    abs_error < tolerance --> converge
    xconverge - t plot
    use: python read_h5_and_plot.py --h5 path/to/file.h5 --var l --converge --times 0 5 10 15 20
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def nearest_index(arr, value):
    arr = np.asarray(arr)
    return int(np.argmin(np.abs(arr - value)))

def load_h5(path):
    f = h5py.File(path, "r")
    time = f["/time"][...]
    x    = f["/coordinates/x_grid"][...]
    y    = f["/coordinates/y_grid"][...]
    return f, time, x, y

def get_field_group(f, var):
    # Map short names to dataset paths
    if var in ("u", "u_mag", "umag"):
        return f["/velocity/u_mag"]
    if var in ("ux", "u_x"):
        return f["/velocity/u_x"]
    if var in ("uy", "u_y"):
        return f["/velocity/u_y"]
    if var in ("p", "pressure"):
        return f["/pressure/p"]
    if var in ("shr", "shear", "shear_rate"):
        return f["/shear_rate/shr"]
    if var in ("l", "lambda"):
        return f["/structure/l"]
    raise ValueError(f"Unknown variable '{var}'")

def plot_lines(h5path, var, line_spec, times, save=None, show=True, dpi=150):
    f, t_arr, x, y = load_h5(h5path)
    data = get_field_group(f, var)

    # Determine line axis
    if line_spec.startswith("x="):
        # single x value
        xs = [float(line_spec.split("=")[1])]
    elif line_spec.startswith("y="):
        ys = [float(line_spec.split("=")[1])]
    else:
        # allow multiple values like: x=-10 0 10  OR y=0.5 1.0
        if line_spec.startswith("x"):
            xs = list(map(float, line_spec[1:].split()))
        elif line_spec.startswith("y"):
            ys = list(map(float, line_spec[1:].split()))
        else:
            raise ValueError("line_spec must start with 'x=' or 'y=' or 'x'/'y' followed by values")

    plt.figure()
    for t in times:
        it = nearest_index(t_arr, t)
        if 'xs' in locals():
            for xv in xs:
                ix = nearest_index(x, xv)
                prof = data[it, :, ix]  # Ny
                plt.plot(y, prof, label=f"t={t_arr[it]:.3f}, x={x[ix]:.3f}")
            plt.xlabel("y")
        else:
            for yv in ys:
                iy = nearest_index(y, yv)
                prof = data[it, iy, :]  # Nx
                plt.plot(x, prof, label=f"t={t_arr[it]:.3f}, y={y[iy]:.3f}")
            plt.xlabel("x")
    plt.ylabel(var)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=dpi)
    if show:
        plt.show()
    f.close()

def plot_colormap(h5path, var, times, save=None, show=True, dpi=150):
    f, t_arr, x, y = load_h5(h5path)
    data = get_field_group(f, var)
    for t in times:
        it = nearest_index(t_arr, t)
        Z = data[it, :, :]  # (Ny, Nx)

        extent = [x.min(), x.max(), y.min(), y.max()]
        plt.figure()
        im = plt.imshow(Z, extent=extent, origin="lower", aspect="auto")
        plt.colorbar(im, label=var)
        plt.xlabel("x"); plt.ylabel("y")
        plt.title(f"{var} at t≈{t_arr[it]:.3f}")
        plt.tight_layout()
        if save:
            savename = f"{var}_colormap_t{t_arr[it]}.png"
            plt.savefig(savename, dpi=dpi)
        if show:
            plt.show()
    f.close()

def plot_error(h5path, var, times, save=None, show=True, dpi=150):
    f, t_arr, x, y = load_h5(h5path)
    data = get_field_group(f, var)
    nx = len(x)
    ny = len(y)
    tot_error = np.zeros_like(x)
    threshold = 1e-3
    plt.figure()
    for t in times:
        it = nearest_index(t_arr, t)
        for ix in range(1, nx):
            curr = data[it, :, ix]
            prev = data[it, :, ix-1]
            abs_error = np.abs(curr - prev)
            denominator = np.where(
                np.abs(prev) == 0,
                1e-12,
                np.abs(prev)
            )
            rel_error = abs_error / denominator
            tot_error[ix] = np.sum(rel_error) / ny
            if tot_error[ix] > threshold:
                x_converge = x[ix]
        if x_converge != 25:
            plt.plot(x, tot_error, label=f"t={t_arr[it]:.3f}, converge at x={x_converge:.3f}")
        else:
            plt.plot(x, tot_error, label=f"t={t_arr[it]:.3f}, don't converge")
    plt.ylabel("Total error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=dpi)
    if show:
        plt.show()
    f.close()

def plot_converge(h5path, var, times, save=None, show = True, dpi=150):
    f, t_arr, x, y = load_h5(h5path)
    data = get_field_group(f, var)
    nx = len(x)
    xc = np.zeros(len(times), dtype = float)
    rtol = 1e-3
    atol = 1e-6
    plt.figure()
    for i, t in enumerate(times):
        it = nearest_index(t_arr, t)
        x_converge = np.nan
        for ix in range(1, nx):
            curr = data[it, :, ix]
            prev = data[it, :, ix-1]
            # 1: all y coordinate should converge
            """
            abs_error = np.abs(curr - prev)
            tolerance = atol + rtol * np.abs(prev)
            is_converged = np.all(abs_error <= tolerance)
            """
            # 2: use vector norm to check convergence
            diff_norm = np.linalg.norm(curr-prev)
            prev_norm = np.linalg.norm(prev)
            tolerance = atol + rtol * prev_norm
            is_converged = (diff_norm <= tolerance)
            if is_converged:
                x_converge = x[ix]
                break
        xc[i] = x_converge

    plt.plot(times, xc, marker = 'o', linestyle = 'None')
    plt.ylabel("Converging x coordinate")
    plt.xlabel("time")
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=dpi)
    if show:
        plt.show()
    f.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Path to HDF5 produced by postprocess_grid_to_h5.py")
    ap.add_argument("--var", default="u_mag", help="One of: u_mag, u_x, u_y, p, shr, l")
    ap.add_argument("--line", default=None, help="Line spec: e.g., 'x=0.0' or 'y=1.25'")
    ap.add_argument("--colormap", action="store_true", help="Plot 2D colormap instead of line(s)")
    ap.add_argument("--error", action="store_true", help="Plot total relative error through x-dir")
    ap.add_argument("--converge", action="store_true", help="Check converging x-dir coordinate")
    ap.add_argument("--times", type=float, nargs="+", required=True, help="One or more target times")
    ap.add_argument("--save", default=None, help="Optional output image path")
    ap.add_argument("--no-show", action="store_true", help="Do not display interactive window")
    args = ap.parse_args()

    if args.colormap:
        plot_colormap(args.h5, args.var, args.times, save=args.save, show=(not args.no_show))
    elif args.error:
        plot_error(args.h5, args.var, args.times, save=args.save, show=(not args.no_show))
    elif args.converge:
        plot_converge(args.h5, args.var, args.times)
    else:
        if args.line is None:
            raise ValueError("Provide --line for line plots (e.g., --line 'x=0.0')")
        plot_lines(args.h5, args.var, args.line, args.times, save=args.save, show=(not args.no_show))

if __name__ == "__main__":
    main()
