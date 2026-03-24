"""
SrTiO3 HAADF-STEM + Ptychography (rPIE) comparison:
  Lobato IAM  vs.  Quantum ESPRESSO multislice.

Outputs are written to a single HDF5 file (``--output``) containing:
  /haadf/iam                 – HAADF image (IAM)
  /haadf/qe                  – HAADF image (QE)
  /haadf/difference          – |IAM − QE|
  /ptycho_4dstem/iam         – 4D-STEM dataset (IAM)
  /ptycho_4dstem/qe          – 4D-STEM dataset (QE)
  /ptycho_rpie/iam/object    – reconstructed phase (IAM)
  /ptycho_rpie/iam/probe     – reconstructed probe (IAM)
  /ptycho_rpie/qe/object     – reconstructed phase (QE)
  /ptycho_rpie/qe/probe      – reconstructed probe (QE)

Usage
-----
    python abtem_sto.py                        # full comparison
    python abtem_sto.py --iam-only             # skip QE
    python abtem_sto.py --skip-ptycho          # HAADF only
    python abtem_sto.py --output results.h5    # custom output path
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

import abtem
from abtem.detectors import AnnularDetector, PixelatedDetector
from abtem.reconstruct import RegularizedPtychographicOperator
from abtem.scan import GridScan
from abtem.waves import Probe

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
CIF_PATH = DATA_DIR / "srtio3_100.cif"

# ---------------------------------------------------------------------------
# QE configuration — adapt paths to your environment
# ---------------------------------------------------------------------------
QE_PW_COMMAND = "/apps/nvhpc/25.3/openmpi/5.0.7/espresso/7.5.0_b200/bin/pw.x"
QE_PP_COMMAND = (
    QE_PW_COMMAND.replace("pw.x", "pp.x")
    if os.path.isfile(QE_PW_COMMAND.replace("pw.x", "pp.x"))
    else "pp.x"
)
PSEUDO_DIR = "/blue/ds.kim/shared/pseudo/paw-sr-11_pbesol_standard_upf/"

PSEUDOPOTENTIALS = {
    "Sr": "Sr.upf",
    "Ti": "Ti.upf",
    "O": "O.upf",
}

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
ENERGY = 200e3           # eV
SAMPLING = 0.05          # Å  (potential grid)
SLICE_THICKNESS = 1.0    # Å
SUPERCELL_REP = (2, 2, 6)

# STEM probe
SEMIANGLE_CUTOFF = 21.4  # mrad  (convergence semi-angle)

# HAADF detector angles (mrad)
HAADF_INNER = 68.0
HAADF_OUTER = 200.0

# STEM scan step size
SCAN_STEP = 0.33  # Å

# Ptychography
PTYCHO_SCAN_STEP = 0.33     # Å  (overlap-rich scan for ptycho)
PTYCHO_MAX_ANGLE = 5.0      # × semiangle_cutoff
PTYCHO_MAX_ITER = 50
PTYCHO_ROI_SHAPE = (128, 128)


# ===================================================================
# Helpers
# ===================================================================

def build_supercell():
    srtio3 = read(str(CIF_PATH))
    repeated = srtio3 * SUPERCELL_REP
    return srtio3, repeated


def _potential_iam(atoms):
    return abtem.Potential(
        atoms,
        sampling=SAMPLING,
        parametrization="lobato",
        slice_thickness=SLICE_THICKNESS,
        projection="finite",
    )


def _potential_qe(workdir):
    from abtem.potentials.qe import QEPotential

    return QEPotential(
        calculators=workdir,
        sampling=SAMPLING,
        slice_thickness=SLICE_THICKNESS,
        pp_command=QE_PP_COMMAND,
    )


# ===================================================================
# HAADF-STEM
# ===================================================================

def run_haadf(potential, label=""):
    """Run a HAADF-STEM scan and return the image (Images object)."""
    probe = Probe(
        energy=ENERGY,
        semiangle_cutoff=SEMIANGLE_CUTOFF,
        sampling=SAMPLING,
    )

    detector = AnnularDetector(inner=HAADF_INNER, outer=HAADF_OUTER)

    cell = potential.box if hasattr(potential, "box") and potential.box else None
    # Build a grid scan covering the full supercell
    if cell is not None:
        end = (cell[0], cell[1])
    else:
        end = (potential.extent[0], potential.extent[1])

    scan = GridScan(
        start=(0, 0),
        end=end,
        sampling=SCAN_STEP,
    )

    print(f"  [{label}] Running HAADF scan ({scan.gpts} grid) …")
    measurement = probe.scan(
        potential,
        scan=scan,
        detectors=detector,
    )

    image = measurement.compute()
    print(f"  [{label}] HAADF range: [{float(image.array.min()):.4f}, "
          f"{float(image.array.max()):.4f}]")
    return image


# ===================================================================
# 4D-STEM + Ptychography (rPIE)
# ===================================================================

def run_4dstem(potential, label=""):
    """Collect a 4D-STEM dataset suitable for ptychographic reconstruction."""
    probe = Probe(
        energy=ENERGY,
        semiangle_cutoff=SEMIANGLE_CUTOFF,
        sampling=SAMPLING,
    )

    max_angle_mrad = PTYCHO_MAX_ANGLE * SEMIANGLE_CUTOFF
    detector = PixelatedDetector(
        max_angle=max_angle_mrad,
        resample="uniform",
    )

    cell = potential.box if hasattr(potential, "box") and potential.box else None
    if cell is not None:
        end = (cell[0], cell[1])
    else:
        end = (potential.extent[0], potential.extent[1])

    scan = GridScan(
        start=(0, 0),
        end=end,
        sampling=PTYCHO_SCAN_STEP,
    )

    print(f"  [{label}] Collecting 4D-STEM ({scan.gpts} grid) …")
    measurement = probe.scan(
        potential,
        scan=scan,
        detectors=detector,
    )

    dataset_4d = measurement.compute()
    print(f"  [{label}] 4D-STEM shape: {dataset_4d.array.shape}")
    return dataset_4d


def run_rpie(dataset_4d, label=""):
    """Reconstruct the phase object from a 4D-STEM dataset using rPIE."""
    print(f"  [{label}] Running rPIE ({PTYCHO_MAX_ITER} iterations) …")

    ptycho = RegularizedPtychographicOperator(
        diffraction_patterns=dataset_4d,
        energy=ENERGY,
        semiangle_cutoff=SEMIANGLE_CUTOFF,
        region_of_interest_shape=PTYCHO_ROI_SHAPE,
        preprocess=True,
        device="cpu",
    )

    objects, probes, positions, sse = ptycho.reconstruct(
        max_iterations=PTYCHO_MAX_ITER,
        return_iterations=True,
        fix_com=True,
    )

    # Take the last iteration
    final_object = objects[-1]
    final_probe = probes[-1]
    final_sse = sse[-1] if hasattr(sse[-1], "__float__") else sse[-1]

    print(f"  [{label}] Final SSE: {float(final_sse):.6e}")

    return final_object, final_probe, positions[-1], sse


# ===================================================================
# QE SCF
# ===================================================================

def run_qe_scf(atoms, workdir="qe_sto_scf"):
    from ase.calculators.espresso import Espresso, EspressoProfile

    os.makedirs(workdir, exist_ok=True)

    profile = EspressoProfile(
        command=QE_PW_COMMAND,
        pseudo_dir=PSEUDO_DIR,
    )

    input_data = {
        "control": {
            "calculation": "scf",
            "restart_mode": "from_scratch",
            "tprnfor": True,
            "tstress": True,
            "etot_conv_thr": 1.0e-8,
            "forc_conv_thr": 1.0e-7,
            "verbosity": "low",
            "disk_io": "low",
        },
        "system": {
            "ibrav": 0,
            "ecutwfc": 80.0,
            "occupations": "fixed",
        },
        "electrons": {
            "conv_thr": 1.0e-12,
            "electron_maxstep": 300,
            "mixing_beta": 0.5,
            "diagonalization": "david",
        },
    }

    kpts = (1, 1, 1)

    calc = Espresso(
        profile=profile,
        input_data=input_data,
        pseudopotentials=PSEUDOPOTENTIALS,
        kpts=kpts,
        directory=workdir,
    )

    atoms = atoms.copy()
    atoms.calc = calc
    atoms.get_potential_energy()
    return atoms


# ===================================================================
# HDF5 output
# ===================================================================

def _write_array(h5, path, data, attrs=None):
    """Write a numpy array (or abTEM measurement) to an HDF5 dataset."""
    arr = data.array if hasattr(data, "array") else np.asarray(data)
    ds = h5.create_dataset(path, data=arr.squeeze(), compression="gzip")
    if attrs:
        for k, v in attrs.items():
            ds.attrs[k] = v


def save_results(output_path, results):
    """
    Save all results to a single HDF5 file.

    Parameters
    ----------
    output_path : str
        Path to the output ``.h5`` file.
    results : dict
        Nested dict of results produced by main().
    """
    with h5py.File(output_path, "w") as h5:
        # --- HAADF ---
        if "haadf" in results:
            grp = h5.create_group("haadf")
            grp.attrs["inner_mrad"] = HAADF_INNER
            grp.attrs["outer_mrad"] = HAADF_OUTER
            grp.attrs["scan_step_angstrom"] = SCAN_STEP

            for tag, img in results["haadf"].items():
                if img is not None:
                    _write_array(h5, f"haadf/{tag}", img)

            iam = results["haadf"].get("iam")
            qe = results["haadf"].get("qe")
            if iam is not None and qe is not None:
                diff = iam.array.squeeze() - qe.array.squeeze()
                h5.create_dataset("haadf/difference", data=diff, compression="gzip")

        # --- 4D-STEM ---
        if "4dstem" in results:
            grp = h5.create_group("ptycho_4dstem")
            grp.attrs["scan_step_angstrom"] = PTYCHO_SCAN_STEP
            grp.attrs["max_angle_mrad"] = PTYCHO_MAX_ANGLE * SEMIANGLE_CUTOFF

            for tag, ds4d in results["4dstem"].items():
                if ds4d is not None:
                    _write_array(h5, f"ptycho_4dstem/{tag}", ds4d)

        # --- rPIE reconstructions ---
        if "rpie" in results:
            for tag, rec in results["rpie"].items():
                if rec is None:
                    continue
                obj, prb, pos, sse = rec
                prefix = f"ptycho_rpie/{tag}"
                h5grp = h5.create_group(prefix)
                h5grp.attrs["max_iterations"] = PTYCHO_MAX_ITER
                h5grp.attrs["roi_shape"] = list(PTYCHO_ROI_SHAPE)

                obj_arr = obj.array if hasattr(obj, "array") else np.asarray(obj)
                prb_arr = prb.array if hasattr(prb, "array") else np.asarray(prb)
                pos_arr = np.asarray(pos)

                h5.create_dataset(
                    f"{prefix}/object", data=obj_arr.squeeze(), compression="gzip"
                )
                h5.create_dataset(
                    f"{prefix}/object_phase",
                    data=np.angle(obj_arr.squeeze()),
                    compression="gzip",
                )
                h5.create_dataset(
                    f"{prefix}/probe", data=prb_arr.squeeze(), compression="gzip"
                )
                h5.create_dataset(
                    f"{prefix}/positions", data=pos_arr, compression="gzip"
                )
                sse_arr = [float(s) for s in sse]
                h5.create_dataset(f"{prefix}/sse", data=sse_arr)

    print(f"\nAll results saved to {output_path}")


# ===================================================================
# Plotting
# ===================================================================

def plot_haadf(results, savefig="abtem_sto_haadf.png"):
    iam = results["haadf"].get("iam")
    qe = results["haadf"].get("qe")

    ncols = (1 if iam is not None else 0) + (1 if qe is not None else 0)
    if ncols == 0:
        return
    ncols += 1 if (iam is not None and qe is not None) else 0

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    idx = 0
    if iam is not None:
        im = axes[idx].imshow(iam.array.squeeze(), cmap="gray")
        axes[idx].set_title("HAADF – Lobato IAM")
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        idx += 1

    if qe is not None:
        im = axes[idx].imshow(qe.array.squeeze(), cmap="gray")
        axes[idx].set_title("HAADF – Quantum ESPRESSO")
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        idx += 1

    if iam is not None and qe is not None:
        diff = iam.array.squeeze() - qe.array.squeeze()
        im = axes[idx].imshow(diff, cmap="RdBu_r")
        axes[idx].set_title("IAM − QE")
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    fig.suptitle("SrTiO₃ HAADF-STEM", fontsize=14)
    fig.tight_layout()
    fig.savefig(savefig, dpi=200, bbox_inches="tight")
    print(f"  Figure saved to {savefig}")


def plot_ptycho(results, savefig="abtem_sto_ptycho.png"):
    rpie = results.get("rpie", {})
    tags = [t for t in ("iam", "qe") if rpie.get(t) is not None]
    if not tags:
        return

    fig, axes = plt.subplots(2, len(tags), figsize=(5 * len(tags), 9))
    if len(tags) == 1:
        axes = axes[:, np.newaxis]

    for col, tag in enumerate(tags):
        obj, prb, _, sse = rpie[tag]
        obj_arr = obj.array if hasattr(obj, "array") else np.asarray(obj)
        prb_arr = prb.array if hasattr(prb, "array") else np.asarray(prb)

        # Phase
        phase = np.angle(obj_arr.squeeze())
        im = axes[0, col].imshow(phase, cmap="twilight")
        axes[0, col].set_title(f"Phase – {tag.upper()}")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        # Probe amplitude
        im2 = axes[1, col].imshow(np.abs(prb_arr.squeeze()), cmap="inferno")
        axes[1, col].set_title(f"Probe |ψ| – {tag.upper()}")
        fig.colorbar(im2, ax=axes[1, col], fraction=0.046, pad=0.04)

    fig.suptitle("SrTiO₃ rPIE Ptychography", fontsize=14)
    fig.tight_layout()
    fig.savefig(savefig, dpi=200, bbox_inches="tight")
    print(f"  Figure saved to {savefig}")


# ===================================================================
# main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SrTiO3 HAADF + Ptychography: Lobato IAM vs QE."
    )
    parser.add_argument(
        "--iam-only", action="store_true",
        help="Skip the QE calculation.",
    )
    parser.add_argument(
        "--skip-ptycho", action="store_true",
        help="Skip ptychography (HAADF only).",
    )
    parser.add_argument(
        "--workdir", default="qe_sto_scf",
        help="Working directory for QE files (default: qe_sto_scf).",
    )
    parser.add_argument(
        "--output", default="abtem_sto_results.h5",
        help="Output HDF5 file (default: abtem_sto_results.h5).",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Suppress matplotlib figures.",
    )
    args = parser.parse_args()

    results = {"haadf": {}, "4dstem": {}, "rpie": {}}

    # --- Structure ---
    print("Reading SrTiO3 and building supercell …")
    prim, supercell = build_supercell()
    print(f"  Primitive cell : {len(prim)} atoms")
    print(f"  Supercell      : {len(supercell)} atoms  (rep = {SUPERCELL_REP})")

    # --- IAM potential ---
    iam_pot = _potential_iam(supercell)

    # ================= HAADF =================
    print("\n===== HAADF-STEM =====")
    print("  --- IAM (Lobato) ---")
    results["haadf"]["iam"] = run_haadf(iam_pot, label="IAM")

    qe_pot = None
    if not args.iam_only:
        print(f"\n  --- QE SCF (workdir={args.workdir}) ---")
        scf_atoms = run_qe_scf(supercell, workdir=args.workdir)
        print(f"  Total energy: {scf_atoms.get_potential_energy():.6f} eV")

        qe_pot = _potential_qe(args.workdir)
        print("  --- QE ---")
        results["haadf"]["qe"] = run_haadf(qe_pot, label="QE")

    # ================= PTYCHOGRAPHY =================
    if not args.skip_ptycho:
        print("\n===== 4D-STEM + Ptychography (rPIE) =====")

        print("  --- IAM ---")
        ds4d_iam = run_4dstem(iam_pot, label="IAM")
        results["4dstem"]["iam"] = ds4d_iam
        obj_iam, prb_iam, pos_iam, sse_iam = run_rpie(ds4d_iam, label="IAM")
        results["rpie"]["iam"] = (obj_iam, prb_iam, pos_iam, sse_iam)

        if qe_pot is not None:
            print("  --- QE ---")
            ds4d_qe = run_4dstem(qe_pot, label="QE")
            results["4dstem"]["qe"] = ds4d_qe
            obj_qe, prb_qe, pos_qe, sse_qe = run_rpie(ds4d_qe, label="QE")
            results["rpie"]["qe"] = (obj_qe, prb_qe, pos_qe, sse_qe)

    # ================= SAVE =================
    print("\n===== Saving results =====")
    save_results(args.output, results)

    # ================= PLOTS =================
    if not args.no_plots:
        print("\n===== Generating figures =====")
        plot_haadf(results)
        if not args.skip_ptycho:
            plot_ptycho(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
