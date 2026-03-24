"""
Interactive viewer for ptychography phase data from abtem_sto HDF5 output.

Usage:
    python ptycho_viewer.py                          # default file
    python ptycho_viewer.py abtem_sto_results.h5     # custom file
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RadioButtons, Button


def load_data(h5path):
    """Load phase images and probe amplitudes from the HDF5 file."""
    data = {}
    with h5py.File(h5path, "r") as h5:
        for tag in ("iam", "qe"):
            prefix = f"ptycho_rpie/{tag}"
            if prefix not in h5:
                continue
            grp = h5[prefix]
            entry = {}
            if "object_phase" in grp:
                entry["phase"] = grp["object_phase"][()]
            elif "object" in grp:
                entry["phase"] = np.angle(grp["object"][()])
            if "probe" in grp:
                entry["probe"] = np.abs(grp["probe"][()])
            if "sse" in grp:
                entry["sse"] = grp["sse"][()]
            if entry:
                data[tag] = entry

        # Also grab HAADF if present
        for tag in ("iam", "qe"):
            key = f"haadf/{tag}"
            if key in h5:
                data.setdefault(tag, {})["haadf"] = h5[key][()]
    return data


def launch_viewer(h5path):
    data = load_data(h5path)
    if not data:
        print(f"No ptychography data found in {h5path}")
        return

    tags = sorted(data.keys())
    has_phase = [t for t in tags if "phase" in data[t]]
    has_probe = [t for t in tags if "probe" in data[t]]

    if not has_phase:
        print("No phase data found.")
        return

    # Check if we can compute a difference
    has_diff = len(has_phase) >= 2

    # Compute global phase range across all tags
    all_phases = [data[t]["phase"] for t in has_phase]
    global_vmin = min(p.min() for p in all_phases)
    global_vmax = max(p.max() for p in all_phases)

    # Compute phase difference if both IAM and QE exist
    phase_diff = None
    if has_diff:
        p0 = data[has_phase[0]]["phase"].squeeze()
        p1 = data[has_phase[1]]["phase"].squeeze()
        # Crop to common shape if needed
        common = tuple(min(a, b) for a, b in zip(p0.shape, p1.shape))
        phase_diff = p0[:common[0], :common[1]] - p1[:common[0], :common[1]]

    ncols = len(has_phase) + (1 if has_diff else 0)
    nrows = 2 if has_probe else 1

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 5 * nrows + 1.8),
        squeeze=False,
    )
    fig.subplots_adjust(bottom=0.22, hspace=0.3, wspace=0.35)

    # --- Phase images ---
    phase_ims = []
    for col, tag in enumerate(has_phase):
        phase = data[tag]["phase"].squeeze()
        im = axes[0, col].imshow(
            phase, cmap="twilight", vmin=global_vmin, vmax=global_vmax,
            origin="lower",
        )
        axes[0, col].set_title(f"Phase – {tag.upper()}")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
        phase_ims.append(im)

    # --- Phase difference ---
    diff_im = None
    if has_diff and phase_diff is not None:
        diff_col = len(has_phase)
        dlim = max(abs(phase_diff.min()), abs(phase_diff.max()))
        diff_im = axes[0, diff_col].imshow(
            phase_diff, cmap="RdBu_r", vmin=-dlim, vmax=dlim,
            origin="lower",
        )
        axes[0, diff_col].set_title(
            f"Δ Phase ({has_phase[0].upper()} − {has_phase[1].upper()})"
        )
        fig.colorbar(diff_im, ax=axes[0, diff_col], fraction=0.046, pad=0.04)

    # --- Probe images ---
    if has_probe:
        for col, tag in enumerate(has_probe):
            probe = data[tag]["probe"].squeeze()
            im = axes[1, col].imshow(
                probe, cmap="inferno", origin="lower",
            )
            axes[1, col].set_title(f"Probe |ψ| – {tag.upper()}")
            fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        # Hide unused axes in probe row
        for col in range(len(has_probe), ncols):
            axes[1, col].set_visible(False)

    fig.suptitle(f"Ptychography Viewer — {Path(h5path).name}", fontsize=13)

    # --- Sliders for vmin / vmax ---
    margin = (global_vmax - global_vmin) * 0.5
    slider_lo = global_vmin - margin
    slider_hi = global_vmax + margin

    ax_vmin = fig.add_axes([0.15, 0.10, 0.55, 0.025])
    ax_vmax = fig.add_axes([0.15, 0.06, 0.55, 0.025])

    s_vmin = Slider(
        ax_vmin, "Phase vmin", slider_lo, slider_hi,
        valinit=global_vmin, valstep=0.01,
    )
    s_vmax = Slider(
        ax_vmax, "Phase vmax", slider_lo, slider_hi,
        valinit=global_vmax, valstep=0.01,
    )

    def update(_val):
        vmin = s_vmin.val
        vmax = s_vmax.val
        for im in phase_ims:
            im.set_clim(vmin=vmin, vmax=vmax)
        fig.canvas.draw_idle()

    s_vmin.on_changed(update)
    s_vmax.on_changed(update)

    # --- Reset button ---
    ax_reset = fig.add_axes([0.78, 0.06, 0.1, 0.04])
    btn_reset = Button(ax_reset, "Reset")

    def reset(_event):
        s_vmin.reset()
        s_vmax.reset()

    btn_reset.on_clicked(reset)

    # --- Colormap selector ---
    ax_cmap = fig.add_axes([0.78, 0.12, 0.12, 0.08])
    cmap_choices = ("twilight", "RdBu_r", "coolwarm", "hsv", "gray")
    radio_cmap = RadioButtons(ax_cmap, cmap_choices, active=0)

    def set_cmap(label):
        for im in phase_ims:
            im.set_cmap(label)
        fig.canvas.draw_idle()

    radio_cmap.on_clicked(set_cmap)

    plt.show()


if __name__ == "__main__":
    default = "abtem_sto_results.h5"
    h5file = sys.argv[1] if len(sys.argv) > 1 else default
    if not Path(h5file).is_file():
        print(f"File not found: {h5file}")
        sys.exit(1)
    launch_viewer(h5file)
