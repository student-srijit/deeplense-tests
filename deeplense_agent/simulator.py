"""Thin wrapper around DeepLenseSim.

The call order for each model follows the scripts in DeepLenseSim (Model_I/sim_cdm.py,
Model_II/sim_axion.py, ...). Nothing here changes the physics; it only turns a validated
SimulationParams into images on disk plus metadata.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .schemas import ImageStats, SimulationParams, SimulationResult

# The vortex mass used in the DeepLenseSim axion scripts.
VORTEX_MASS = 3e10
# Model_I hard-codes its pixel size inside DeepLens.simple_sim (deltaPix = 0.05).
MODEL_I_PIXEL_SCALE = 0.05


def _make_lens(params: SimulationParams):
    # Imported here so that importing the package (and its schemas) does not need lenstronomy.
    from deeplense.lens import DeepLens

    lens = DeepLens(
        axion_mass=params.axion_mass if params.substructure == "vortex" else None,
        z_halo=params.z_halo,
        z_gal=params.z_gal,
    )
    lens.make_single_halo(params.halo_mass)

    if params.substructure == "no_sub":
        lens.make_no_sub()
    elif params.substructure == "cdm":
        lens.make_old_cdm()
    else:
        lens.make_vortex(VORTEX_MASS)

    if params.model_name == "Model_I":
        lens.make_source_light()
        lens.simple_sim()
        instrument, pixel_scale = "generic (Gaussian PSF)", MODEL_I_PIXEL_SCALE
    else:
        lens.set_instrument("Euclid")
        lens.make_source_light_mag()
        lens.simple_sim_2()
        instrument = "Euclid VIS"
        pixel_scale = float(lens.kwargs_single_band["pixel_scale"])
    return lens, instrument, pixel_scale


def run_simulation(params: SimulationParams, output_root: Path) -> SimulationResult:
    """Generate params.num_images images and save them under output_root/<run_id>/."""
    started = time.time()
    run_id = uuid.uuid4().hex[:8]
    out_dir = Path(output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    images, instrument, pixel_scale = [], "", 0.0
    for i in range(params.num_images):
        if params.seed is not None:
            # DeepLenseSim draws from both numpy and python's random module.
            np.random.seed(params.seed + i)
            random.seed(params.seed + i)
        lens, instrument, pixel_scale = _make_lens(params)
        images.append(np.asarray(lens.image_real, dtype=np.float32))

    files = []
    for i, img in enumerate(images):
        path = out_dir / f"image_{i:03d}.npy"
        np.save(path, img)
        files.append(str(path))

    grid_path = out_dir / "grid.png"
    _save_grid(images, params, grid_path)

    result = SimulationResult(
        run_id=run_id,
        params=params,
        image_shape=tuple(images[0].shape),
        pixel_scale_arcsec=pixel_scale,
        instrument=instrument,
        files=files,
        grid_png=str(grid_path),
        stats=[
            ImageStats(index=i, min=float(a.min()), max=float(a.max()), mean=float(a.mean()))
            for i, a in enumerate(images)
        ],
        elapsed_seconds=round(time.time() - started, 2),
    )
    (out_dir / "metadata.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def _save_grid(images: list[np.ndarray], params: SimulationParams, path: Path) -> None:
    cols = min(len(images), 5)
    rows = -(-len(images) // cols)
    # Figure (not pyplot) so importing this module never changes the caller's matplotlib backend.
    fig = Figure(figsize=(2.4 * cols, 2.4 * rows))
    axes = fig.subplots(rows, cols, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img in zip(axes.ravel(), images):
        ax.imshow(img)
    fig.suptitle(f"{params.model_name} / {params.substructure}", y=1.0)
    fig.savefig(path, dpi=80, bbox_inches="tight")


def load_metadata(run_dir: Path) -> SimulationResult:
    return SimulationResult.model_validate(json.loads((Path(run_dir) / "metadata.json").read_text()))
