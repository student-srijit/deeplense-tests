# DeepLense evaluation tests (practice)

Practice work on the ML4SCI DeepLense GSoC evaluation tests (2026 version, used as practice for 2027):
Common Test I (lens classification) and Test II for the project
"Agentic AI for Autonomous Gravitational Lensing Simulation Workflows".

## Contents

- `notebooks/01_common_test_1_classification.ipynb` - Common Test I: 3-class lens classification (no substructure / sphere / vortex), ROC and AUC
- `notebooks/02_test_2_agentic_deeplensesim.ipynb` - Test II: a Pydantic AI agent that wraps DeepLenseSim
- `deeplense_agent/` - the agent package (schemas, simulator wrapper, UI, agent, LLM configs)
- `tests/test_agent.py` - 23 tests that need no LLM
- `figures/` - sample images

## Common Test I results

Metrics are on the 90:10 validation split (33,750 train / 3,750 validation images, balanced classes).
Each model was trained for 15 epochs on a Colab T4 GPU; numbers are from the last epoch.

| Experiment | Val AUC | Val accuracy |
|---|---|---|
| Baseline small CNN | 0.907 | 0.759 |
| + augmentation | 0.877 | 0.480 |
| + AdamW, cosine schedule, weight decay | 0.868 | 0.708 |
| augmentation + AdamW and the rest | 0.771 | 0.561 |
| **ResNet-18 (ImageNet weights) + augmentation + cosine** | **0.994** | **0.951** |

Final macro one-vs-rest AUC for the best model: **0.9937**.

Limits: one run per experiment (one seed), and the best model was chosen on the same
validation set the numbers are reported on, so the final figure is slightly optimistic.
The small CNN had not converged after 15 epochs, so its comparison with the augmentation
and optimiser variants should not be read as a general conclusion.

## Test II: the agent

The agent turns a plain-language request into a DeepLenseSim run for Model_I or Model_II
(no substructure, CDM or vortex). It:

- validates every tool call with Pydantic (for example vortex needs an axion mass, at most 25 images per run)
- asks the user a question instead of guessing missing values
- shows its plan and waits for human confirmation before simulating
- works with a local Ollama model that supports tools (`qwen2.5:7b`)

The 23 tests use a scripted fake model, so they need neither an LLM nor a GPU.
Run them with `python -m pytest tests -q`.

## Test II environment (Python 3.11, `.venv-sim`)

DeepLenseSim only works with an old set of library versions. What worked here:

    python3.11 -m venv .venv-sim
    pip install "numpy<1.27" "scipy<1.14" "astropy==5.3.4" "numba==0.58.1" colossus lenstronomy==1.9.2 \
                mpmath mcfit emcee h5py pyyaml matplotlib
    git clone https://github.com/mwt5345/DeepLenseSim external/DeepLenseSim
    pip install -e external/DeepLenseSim --no-deps
    git clone https://github.com/dangilman/pyHalo external/pyHalo
    git -C external/pyHalo checkout 64582db
    # pyHalo's June 2022 commit: current pyHalo needs a newer lenstronomy. Its editable install failed
    # here, so it is put on the path with a .pth file in .venv-sim/Lib/site-packages instead.
    pip install "pydantic-ai-slim[openai]" ipykernel nbformat nbclient pytest

LLM: a local Ollama model **with tool support** (`ollama show <model>` must list `tools`).
`llama3` does not; `qwen2.5:7b` does (`ollama pull qwen2.5:7b`).

## Running Test I

Open notebook 01 in Google Colab, choose a T4 GPU and run all cells (about 35 minutes).
The notebook downloads the dataset itself.
