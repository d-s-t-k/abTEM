"""Module to handle ab initio electrostatic potentials from Quantum ESPRESSO."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ase.calculators.espresso import Espresso as _EspressoType

import dask
import dask.array as da
import numpy as np
from ase import Atoms, units
from ase.io.cube import read_cube_data

from abtem.atoms import is_cell_orthogonal, plane_to_axes
from abtem.core.axes import AxisMetadata
from abtem.core.ensemble import _wrap_with_array
from abtem.core.fft import fft_crop
from abtem.core.utils import itemset
from abtem.inelastic.phonons import (
    BaseFrozenPhonons,
    DummyFrozenPhonons,
    FrozenPhonons,
)
from abtem.parametrizations import EwaldParametrization
from abtem.potentials.charge_density import _interpolate_slice
from abtem.potentials.iam import Potential, PotentialArray, _PotentialBuilder

try:
    from ase.calculators.espresso import Espresso, EspressoProfile
except ImportError:
    Espresso = None
    EspressoProfile = None


def _read_atoms_qe(calculator, prefix: str = "pwscf") -> Atoms:
    """Read atoms from a QE calculator or working directory."""
    if isinstance(calculator, Atoms):
        atoms = calculator
    elif isinstance(calculator, str):
        # Directory path – try QE output, then input
        pwo = os.path.join(calculator, "espresso.pwo")
        pwi = os.path.join(calculator, "espresso.pwi")
        xml = os.path.join(calculator, f"{prefix}.save", "data-file-schema.xml")
        if os.path.isfile(pwo):
            from ase.io import read
            atoms = read(pwo, format="espresso-out")
        elif os.path.isfile(xml):
            from ase.io import read
            atoms = read(xml, format="espresso-out")
        elif os.path.isfile(pwi):
            from ase.io import read
            atoms = read(pwi, format="espresso-in")
        else:
            raise FileNotFoundError(
                f"Could not find QE output in '{calculator}'. "
                "Expected espresso.pwo, espresso.pwi, or "
                f"{prefix}.save/data-file-schema.xml."
            )
    elif hasattr(calculator, "atoms"):
        atoms = calculator.atoms
    else:
        raise TypeError(
            f"Cannot read atoms from {type(calculator)}. "
            "Expected an Atoms object, a directory path, or a calculator."
        )
    atoms = atoms.copy()
    atoms.constraints = None
    atoms.calc = None
    return atoms


def _run_pp_x(
    pp_command: str,
    outdir: str,
    prefix: str,
    filplot: str,
    fileout: str,
    plot_num: int = 11,
):
    """
    Run Quantum ESPRESSO's ``pp.x`` post-processing tool to extract the
    electrostatic potential on a real-space grid and write it as a Gaussian
    cube file.

    Parameters
    ----------
    pp_command : str
        Command (or full path) to run ``pp.x``.
    outdir : str
        Directory containing the QE save folder (``<prefix>.save``).
    prefix : str
        QE ``prefix`` used during the SCF calculation.
    filplot : str
        Intermediate plot file produced by ``pp.x``.
    fileout : str
        Path to the output cube file.
    plot_num : int
        Quantity to extract.  ``11`` = the *total* local (electrostatic)
        potential V_bare + V_Hartree + V_xc  (in Ry).
    """
    pp_input = (
        f"&INPUTPP\n"
        f"  prefix = '{prefix}',\n"
        f"  outdir = '{outdir}',\n"
        f"  filplot = '{filplot}',\n"
        f"  plot_num = {plot_num},\n"
        f"/\n"
        f"&PLOT\n"
        f"  iflag = 3,\n"
        f"  output_format = 6,\n"
        f"  fileout = '{fileout}',\n"
        f"/\n"
    )

    result = subprocess.run(
        pp_command.split(),
        input=pp_input,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"pp.x failed (return code {result.returncode}):\n{result.stderr}"
        )


def _read_potential_cube(path: str) -> np.ndarray:
    """
    Read the electrostatic potential from a Gaussian cube file produced
    by ``pp.x`` and convert from Ry to eV.

    Parameters
    ----------
    path : str
        Path to the cube file.

    Returns
    -------
    potential : numpy.ndarray
        3-D real-space potential array in eV.
    """
    data, _ = read_cube_data(path)
    return data * units.Ry


def _extract_potential_from_qe(
    outdir: str,
    prefix: str,
    pp_command: str = "pp.x",
    plot_num: int = 11,
) -> np.ndarray:
    """
    Extract the electrostatic potential from a completed QE calculation.

    Parameters
    ----------
    outdir : str
        Directory containing the QE save data.
    prefix : str
        QE calculation prefix.
    pp_command : str
        Command to invoke ``pp.x``.
    plot_num : int
        Quantity selector for ``pp.x`` (default 11 = V_tot).

    Returns
    -------
    potential : numpy.ndarray
        3-D potential in eV.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filplot = os.path.join(tmpdir, "filplot")
        fileout = os.path.join(tmpdir, "potential.cube")

        _run_pp_x(
            pp_command=pp_command,
            outdir=outdir,
            prefix=prefix,
            filplot=filplot,
            fileout=fileout,
            plot_num=plot_num,
        )

        potential = _read_potential_cube(fileout)

    return potential


def integrate_slice(array, gpts, a, b, thickness):
    """Integrate potential array over a slab [a, b) along z."""
    dz = thickness / array.shape[2]
    na = int(np.floor(a / dz))
    nb = int(np.floor(b / dz))
    slice_array = np.sum(array[..., na:nb], axis=-1) * dz
    new_shape = (nb - na,) + gpts
    old_shape = (nb - na,) + slice_array.shape
    slice_array = np.fft.fftn(slice_array)
    slice_array = fft_crop(slice_array, gpts)
    slice_array = (
        np.fft.ifftn(slice_array).real * np.prod(new_shape) / np.prod(old_shape)
    )
    return slice_array


class _DummyParametrization:
    """Wrapper to present an interpolator as a parametrization."""

    def __init__(self, potential):
        self._potential = potential

    def potential(self, symbol):
        return self._potential

    @property
    def sigmas(self):
        return {}


@dataclass
class _DummyQE:
    """
    Lightweight container that stores the essential results of a converged
    Quantum ESPRESSO calculation so that the heavy calculator object does
    not have to be kept in memory.
    """

    atoms: Atoms
    electrostatic_potential: np.ndarray
    outdir: str
    prefix: str
    pseudo_dir: Optional[str] = None
    pseudopotentials: Optional[dict] = None

    @classmethod
    def from_espresso(
        cls,
        calculator,
        pp_command: str = "pp.x",
        plot_num: int = 11,
        lazy: bool = True,
    ):
        """Build from a converged ASE ``Espresso`` calculator."""
        atoms = calculator.atoms.copy()
        atoms.calc = None

        params = calculator.parameters
        outdir = params.get("outdir", calculator.directory or ".")
        prefix = params.get("prefix", "pwscf")

        potential = _extract_potential_from_qe(
            outdir=outdir,
            prefix=prefix,
            pp_command=pp_command,
            plot_num=plot_num,
        )

        return cls(
            atoms=atoms,
            electrostatic_potential=potential,
            outdir=outdir,
            prefix=prefix,
            pseudo_dir=params.get("pseudo_dir"),
            pseudopotentials=params.get("pseudopotentials"),
        )

    @classmethod
    def from_directory(
        cls,
        path: str,
        atoms: Atoms = None,
        prefix: str = "pwscf",
        pp_command: str = "pp.x",
        plot_num: int = 11,
        lazy: bool = True,
    ):
        """
        Build from a QE output directory that already contains converged
        results (i.e. ``<prefix>.save/``).

        If *atoms* is not given, the structure is read from the QE XML file
        inside the save directory.
        """
        if lazy:
            return dask.delayed(cls.from_directory)(
                path, atoms=atoms, prefix=prefix,
                pp_command=pp_command, plot_num=plot_num, lazy=False,
            )

        if atoms is None:
            atoms = _read_atoms_qe(path, prefix=prefix)

        potential = _extract_potential_from_qe(
            outdir=path,
            prefix=prefix,
            pp_command=pp_command,
            plot_num=plot_num,
        )

        return cls(
            atoms=atoms,
            electrostatic_potential=potential,
            outdir=path,
            prefix=prefix,
        )

    @classmethod
    def from_generic(cls, calculator, lazy: bool = True, **kwargs):
        if isinstance(calculator, str):
            return cls.from_directory(calculator, lazy=lazy, **kwargs)
        elif isinstance(calculator, cls):
            return calculator
        elif Espresso is not None and isinstance(calculator, Espresso):
            return cls.from_espresso(calculator, lazy=lazy, **kwargs)
        else:
            raise RuntimeError(
                f"Cannot build _DummyQE from {type(calculator)}. "
                "Expected an ASE Espresso calculator, a path to a QE "
                "output directory, or an existing _DummyQE instance."
            )


def _generate_slices(
    valence_potential,
    atoms,
    gpts,
    slice_thickness,
    ewald_potential,
    plane="xy",
    first_slice=0,
    last_slice=None,
):
    """
    Yield potential slices by combining the Ewald (nuclear) IAM contribution
    with the DFT valence electrostatic potential extracted from QE.
    """
    ewald_gen = ewald_potential.generate_slices()

    transform_valence_potential = None
    if ewald_potential.plane != "xy":
        if not is_cell_orthogonal(atoms.cell):
            raise NotImplementedError(
                "Non-orthogonal cells are not supported for non-xy planes."
            )
        axes = plane_to_axes(ewald_potential.plane)
        valence_potential = np.moveaxis(valence_potential, axes[:2], (0, 1))
        transform_valence_potential = False

    transformed_atoms = ewald_potential.get_transformed_atoms()
    if np.allclose(transformed_atoms.cell, atoms.cell):
        transform_valence_potential = False
    elif transform_valence_potential is None:
        transform_valence_potential = True

    if last_slice is None:
        last_slice = len(ewald_potential)

    for slice_idx in range(first_slice, last_slice):
        slic = next(ewald_gen)

        a, b = ewald_potential.get_sliced_atoms().slice_limits[slice_idx]

        if transform_valence_potential:
            slic.array[:] -= _interpolate_slice(
                valence_potential,
                atoms.cell,
                ewald_potential.gpts,
                ewald_potential.sampling,
                a,
                b,
            )
        else:
            slic.array[:] -= integrate_slice(
                valence_potential,
                ewald_potential.gpts,
                a,
                b,
                ewald_potential.thickness,
            )

        yield slic


class QEPotential(_PotentialBuilder):
    """
    Calculate the electrostatic potential from (a set of) converged Quantum
    ESPRESSO DFT calculation(s).  Frozen phonon configurations can be included
    by providing multiple calculators or by using the ``frozen_phonons``
    keyword.

    The implementation mirrors ``GPAWPotential`` but replaces all
    GPAW-specific data extraction with calls to QE's ``pp.x``
    post-processor (via a temporary cube file).

    Parameters
    ----------
    calculators : Espresso or str or list
        An ASE ``Espresso`` calculator, a path to a QE output directory, a
        ``_DummyQE`` container, or a list of any of these.  The atomic
        structure is read from the calculator; if a path is given the atoms
        must be discoverable (XML output) or provided via *frozen_phonons*.
    gpts : int or (int, int), optional
        Number of grid points for each slice.  Supply *either* ``gpts``
        *or* ``sampling``.
    sampling : float or (float, float), optional
        Grid spacing of each slice [Å].
    slice_thickness : float or sequence of float, optional
        Slice thickness in the propagation direction [Å] (default 1.0 Å).
    exit_planes : int or tuple of int, optional
        Indices (or spacing) for collecting exit-plane measurements.
    plane : str, optional
        Crystal plane mapped to the *xy* plane (default ``'xy'``).
    origin : (float, float, float), optional
        Origin shift applied to the atoms (default ``(0, 0, 0)``).
    box : (float, float, float), optional
        Explicit potential extent; determined from the cell if not given.
    periodic : bool, optional
        Whether to enforce periodicity (default ``True``).
    frozen_phonons : BaseFrozenPhonons, optional
        Frozen-phonon configurations.
    repetitions : (int, int, int), optional
        Super-cell repetitions applied before slicing (default ``(1,1,1)``).
    pp_command : str, optional
        Command used to invoke QE's ``pp.x`` (default ``'pp.x'``).
    plot_num : int, optional
        ``pp.x`` ``plot_num`` flag selecting which quantity to extract
        (default ``11`` = total local potential).
    device : str, optional
        ``'cpu'`` or ``'gpu'``.
    """

    def __init__(
        self,
        calculators: Union[_EspressoType, List[_EspressoType], List[str], str],
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        slice_thickness: float = 1.0,
        exit_planes: int = None,
        plane: str = "xy",
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Tuple[float, float, float] = None,
        periodic: bool = True,
        frozen_phonons: BaseFrozenPhonons = None,
        repetitions: Tuple[int, int, int] = (1, 1, 1),
        pp_command: str = "pp.x",
        plot_num: int = 11,
        device: str = None,
    ):
        if Espresso is None:
            raise RuntimeError(
                "This functionality of abTEM requires ASE's Quantum "
                "ESPRESSO interface.  Install ASE >= 3.22 and ensure "
                "Quantum ESPRESSO is available on the system."
            )

        self._pp_command = pp_command
        self._plot_num = plot_num
        qe_kwargs = dict(pp_command=pp_command, plot_num=plot_num)

        if isinstance(calculators, (tuple, list)):
            atoms = _read_atoms_qe(calculators[0])
            num_configs = len(calculators)

            if frozen_phonons is not None:
                raise ValueError(
                    "Cannot provide both a list of calculators and "
                    "frozen_phonons simultaneously."
                )

            calculators = [
                _DummyQE.from_generic(calc, **qe_kwargs)
                for calc in calculators
            ]

            frozen_phonons = DummyFrozenPhonons(atoms, num_configs=num_configs)

        else:
            atoms = _read_atoms_qe(calculators)

            calculators = _DummyQE.from_generic(calculators, **qe_kwargs)

            if frozen_phonons is None:
                frozen_phonons = DummyFrozenPhonons(atoms, num_configs=None)

        self._calculators = calculators
        self._frozen_phonons = frozen_phonons
        self._repetitions = repetitions

        cell = frozen_phonons.atoms.cell * repetitions
        frozen_phonons.atoms.calc = None

        super().__init__(
            array_object=PotentialArray,
            gpts=gpts,
            sampling=sampling,
            cell=cell,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
            plane=plane,
            origin=origin,
            box=box,
            periodic=periodic,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frozen_phonons(self):
        return self._frozen_phonons

    @property
    def num_configurations(self):
        return self.frozen_phonons.num_configs

    @property
    def repetitions(self):
        return self._repetitions

    @property
    def pp_command(self):
        return self._pp_command

    @property
    def plot_num(self):
        return self._plot_num

    @property
    def calculators(self):
        return self._calculators

    # ------------------------------------------------------------------
    # Ewald / IAM helper
    # ------------------------------------------------------------------

    def _get_ewald_potential(self, atoms):
        """
        Build an IAM Ewald potential for the given atoms so that the nuclear
        contribution can be subtracted from the QE all-electron potential.
        """
        ewald_parametrization = EwaldParametrization(width=3)

        return Potential(
            atoms=atoms,
            gpts=self.gpts,
            sampling=self.sampling,
            parametrization=ewald_parametrization,
            slice_thickness=self.slice_thickness,
            projection="finite",
            plane=self.plane,
            box=self.box,
            origin=self.origin,
            exit_planes=self.exit_planes,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Slice generation
    # ------------------------------------------------------------------

    def generate_slices(self, first_slice: int = 0, last_slice: int = None):
        """
        Generate potential slices.

        Parameters
        ----------
        first_slice : int, optional
            First slice index (default 0).
        last_slice : int, optional
            Last slice index (default: all slices).

        Yields
        ------
        PotentialArray
            One slice of the projected potential.
        """
        if last_slice is None:
            last_slice = len(self)

        calculator = (
            self.calculators[0]
            if isinstance(self.calculators, list)
            else self.calculators
        )

        try:
            calculator = calculator.compute()
        except AttributeError:
            pass

        calculator = _DummyQE.from_generic(calculator)

        atoms = self.frozen_phonons.atoms

        if self.repetitions != (1, 1, 1):
            atoms = atoms * self.repetitions

        random_atoms = self.frozen_phonons.randomize(atoms)

        ewald_potential = self._get_ewald_potential(random_atoms)

        for slic in _generate_slices(
            valence_potential=calculator.electrostatic_potential,
            atoms=random_atoms,
            gpts=self.gpts,
            slice_thickness=self.slice_thickness,
            ewald_potential=ewald_potential,
            plane=self.plane,
            first_slice=first_slice,
            last_slice=last_slice,
        ):
            yield slic

    # ------------------------------------------------------------------
    # Ensemble / partitioning machinery (mirrors GPAWPotential)
    # ------------------------------------------------------------------

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        return self._frozen_phonons.ensemble_axes_metadata

    @property
    def num_frozen_phonons(self):
        return len(self.calculators)

    @property
    def ensemble_shape(self):
        return self._frozen_phonons.ensemble_shape

    @staticmethod
    def _qe_potential(*args, frozen_phonons_partial, **kwargs):
        args = args[0]
        if hasattr(args, "item"):
            args = args.item()

        if args["frozen_phonons"] is not None:
            frozen_phonons = frozen_phonons_partial(args["frozen_phonons"])
        else:
            frozen_phonons = None

        calculators = args["calculators"]

        new_potential = QEPotential(
            calculators, frozen_phonons=frozen_phonons, **kwargs
        )
        return _wrap_with_array(new_potential)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(
            exclude=("calculators", "frozen_phonons")
        )

        frozen_phonons_partial = self.frozen_phonons._from_partitioned_args()

        return partial(
            self._qe_potential,
            frozen_phonons_partial=frozen_phonons_partial,
            **kwargs,
        )

    def _partition_args(self, chunks: int = 1, lazy: bool = True):
        chunks = self._validate_ensemble_chunks(chunks)

        def _pack(calculators, frozen_phonons):
            arr = np.zeros((1,), dtype=object)
            itemset(
                arr,
                0,
                {"calculators": calculators, "frozen_phonons": frozen_phonons},
            )
            return arr

        calculators = self.calculators

        if isinstance(self.frozen_phonons, FrozenPhonons):
            array = np.zeros(len(self.frozen_phonons), dtype=object)
            for i, fp in enumerate(
                self.frozen_phonons._partition_args(chunks, lazy=lazy)[0]
            ):
                if lazy:
                    block = dask.delayed(_pack)(calculators, fp)
                    itemset(
                        array, i, da.from_delayed(block, shape=(1,), dtype=object)
                    )
                else:
                    itemset(array, i, _pack(calculators, fp))

            if lazy:
                array = da.concatenate(list(array))

        else:
            if len(self.ensemble_shape) == 0 and lazy:
                block = dask.delayed(_pack)(calculators, None)
                array = da.from_delayed(block, shape=(), dtype=object)
            elif len(self.ensemble_shape) == 0 and not lazy:
                array = _pack(calculators, None)
                array = _wrap_with_array(array, ndims=0)
            else:
                array = np.zeros(self.ensemble_shape[0], dtype=object)
                for i, calculator in enumerate(calculators):
                    calculator = [calculator]

                    if lazy:
                        calculator = dask.delayed(calculator)
                        block = da.from_delayed(
                            dask.delayed(_pack)(calculator, None),
                            shape=(1,),
                            dtype=object,
                        )
                    else:
                        block = _pack(calculator, None)

                    itemset(array, i, block)

                if lazy:
                    array = da.concatenate(list(array))

        return (array,)
