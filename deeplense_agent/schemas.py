"""Pydantic models for the DeepLenseSim agent.

Everything the LLM can ask the simulator to do goes through SimulationParams, so bad
values (a negative redshift, 5000 images, an axion mass for a CDM lens) are rejected
here with a readable message instead of crashing inside the simulator.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Which DeepLenseSim configuration to use. These two are implemented in the wrapper.
#   Model_I  : 150x150 pixels, simple Gaussian-PSF set-up, source given by amplitude
#   Model_II : 64x64 pixels, Euclid VIS instrument, source given by magnitude
ModelName = Literal["Model_I", "Model_II"]

# Type of dark-matter substructure in the lens.
#   no_sub : smooth lens, no substructure
#   cdm    : cold dark matter subhalos
#   vortex : axion / superfluid dark matter vortex (needs an axion mass)
Substructure = Literal["no_sub", "cdm", "vortex"]

MAX_IMAGES = 25
AXION_MASS_RANGE = (1e-24, 1e-22)  # eV, the range used in the DeepLenseSim scripts


class SimulationParams(BaseModel):
    """Everything needed for one batch of simulated lensing images."""

    model_name: ModelName = Field(description="Which DeepLenseSim model: Model_I or Model_II.")
    substructure: Substructure = Field(description="no_sub, cdm or vortex.")
    num_images: int = Field(ge=1, le=MAX_IMAGES, description=f"How many images, 1 to {MAX_IMAGES}.")
    halo_mass: float = Field(
        default=1e12, ge=1e10, le=1e14, description="Main halo mass in solar masses."
    )
    axion_mass: float | None = Field(
        default=None,
        description="Axion mass in eV (1e-24 to 1e-22). Only for substructure='vortex'.",
    )
    z_halo: float = Field(default=0.5, gt=0, lt=5, description="Redshift of the lens halo.")
    z_gal: float = Field(default=1.0, gt=0, lt=10, description="Redshift of the source galaxy.")
    seed: int | None = Field(default=None, description="Random seed for reproducible images.")

    @model_validator(mode="after")
    def _check_consistency(self) -> "SimulationParams":
        if self.z_halo >= self.z_gal:
            raise ValueError(
                f"z_halo ({self.z_halo}) must be smaller than z_gal ({self.z_gal}): "
                "the lens has to be in front of the source."
            )
        if self.substructure == "vortex":
            if self.axion_mass is None:
                raise ValueError("axion_mass is required when substructure is 'vortex'.")
            low, high = AXION_MASS_RANGE
            if not low <= self.axion_mass <= high:
                raise ValueError(f"axion_mass must be between {low:g} and {high:g} eV.")
        elif self.axion_mass is not None:
            raise ValueError("axion_mass only applies to substructure='vortex'.")
        return self


class ImageStats(BaseModel):
    index: int
    min: float
    max: float
    mean: float


class SimulationResult(BaseModel):
    """What a finished simulation hands back. Built by code, never written by the LLM."""

    run_id: str
    params: SimulationParams
    image_shape: tuple[int, int]
    pixel_scale_arcsec: float
    instrument: str
    files: list[str]
    grid_png: str
    stats: list[ImageStats]
    elapsed_seconds: float


class AgentReply(BaseModel):
    """The agent's final answer to the user (text only; numbers come from SimulationResult)."""

    message: str = Field(description="Short plain-language answer for the user.")
