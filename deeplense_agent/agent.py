"""The Pydantic AI agent that turns a natural-language request into DeepLenseSim runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from . import simulator
from .schemas import (
    AXION_MASS_RANGE,
    MAX_IMAGES,
    AgentReply,
    SimulationParams,
    SimulationResult,
)
from .ui import UserInterface

MAX_QUESTIONS = 4  # a chatty model should not interrogate the user forever

INSTRUCTIONS = f"""\
You help a physicist generate simulated strong gravitational lensing images with DeepLenseSim.

Rules:
1. Call get_options first if you are unsure what is allowed.
2. A simulation needs: model_name, substructure and num_images. If the user did not say one of
   them clearly, do NOT guess. Call ask_user with ONE short question about what is missing.
3. Image size cannot be chosen freely: it comes from the model (Model_I is 150x150 pixels,
   Model_II is Euclid 64x64 pixels). If the user asks for a resolution, ask which model they want
   or explain this.
4. substructure='vortex' needs an axion_mass between {AXION_MASS_RANGE[0]:g} and
   {AXION_MASS_RANGE[1]:g} eV. If it is missing, ask for it (1e-23 is a reasonable suggestion).
5. At most {MAX_IMAGES} images per request.
6. Only call run_simulation when every value is known. It shows the plan to the user, who may
   decline with a change request. If declined, apply the change and call it again.
7. Leave halo_mass, z_halo, z_gal, seed and axion_mass at their defaults (or unset) unless the
   user explicitly gave a value. Never pick model_name yourself: if the user has not named
   Model_I or Model_II, ask them, even when other details are known.
8. When done, reply with a short summary. Do not invent file paths or numbers: use the tool result.
If the request has nothing to do with lensing simulations, say so and do not call run_simulation.
"""


@dataclass
class AgentDeps:
    ui: UserInterface
    output_root: Path
    results: list[SimulationResult] = field(default_factory=list)
    questions_asked: int = 0


def build_agent(model: Model | str) -> Agent[AgentDeps, str]:
    # The final output is plain text. Small local models often write their follow-up question
    # as text instead of calling ask_user, so simulate() below also treats a text reply that
    # asks a question as a request to the human, instead of rejecting it.
    agent: Agent[AgentDeps, str] = Agent(
        model,
        deps_type=AgentDeps,
        output_type=str,
        instructions=INSTRUCTIONS,
        retries=3,
    )

    @agent.tool_plain
    def get_options() -> dict:
        """List the models, substructure types and limits the simulator supports."""
        return {
            "model_name": {
                "Model_I": "150x150 pixels, simple Gaussian PSF",
                "Model_II": "64x64 pixels, Euclid VIS instrument",
            },
            "substructure": {
                "no_sub": "smooth lens, no substructure",
                "cdm": "cold dark matter subhalos",
                "vortex": "axion dark matter vortex (needs axion_mass)",
            },
            "num_images": f"1 to {MAX_IMAGES}",
            "halo_mass": "1e10 to 1e14 solar masses, default 1e12",
            "axion_mass_eV": f"{AXION_MASS_RANGE[0]:g} to {AXION_MASS_RANGE[1]:g}",
            "z_halo": "default 0.5 (must be smaller than z_gal)",
            "z_gal": "default 1.0",
        }

    @agent.tool
    def ask_user(ctx: RunContext[AgentDeps], question: str) -> str:
        """Ask the human ONE short clarifying question and return their answer."""
        if ctx.deps.questions_asked >= MAX_QUESTIONS:
            return (
                "No more questions allowed. Use the most reasonable defaults, "
                "say which ones you chose, and continue."
            )
        ctx.deps.questions_asked += 1
        return ctx.deps.ui.ask(question)

    @agent.tool
    def run_simulation(ctx: RunContext[AgentDeps], params: SimulationParams) -> dict:
        """Run DeepLenseSim for the given parameters, after the user approves the plan."""
        approved, comment = ctx.deps.ui.confirm(_describe(params))
        if not approved:
            return {
                "status": "declined_by_user",
                "user_comment": comment or "(no comment)",
                "next_step": "Apply the user's change and call run_simulation again, or ask_user.",
            }
        try:
            result = simulator.run_simulation(params, ctx.deps.output_root)
        except Exception as exc:  # the model gets a readable error and may retry
            raise ModelRetry(f"The simulator failed: {type(exc).__name__}: {exc}") from exc
        ctx.deps.results.append(result)
        return {
            "status": "done",
            "run_id": result.run_id,
            "num_images": len(result.files),
            "image_shape": list(result.image_shape),
            "pixel_scale_arcsec": result.pixel_scale_arcsec,
            "instrument": result.instrument,
            "grid_png": result.grid_png,
            "elapsed_seconds": result.elapsed_seconds,
        }

    return agent


def _describe(p: SimulationParams) -> str:
    lines = [
        f"  model       : {p.model_name}",
        f"  substructure: {p.substructure}",
        f"  images      : {p.num_images}",
        f"  halo mass   : {p.halo_mass:g} solar masses",
        f"  redshifts   : lens z={p.z_halo}, source z={p.z_gal}",
    ]
    if p.axion_mass is not None:
        lines.append(f"  axion mass  : {p.axion_mass:g} eV")
    if p.seed is not None:
        lines.append(f"  seed        : {p.seed}")
    return "\n".join(lines)


@dataclass
class Outcome:
    reply: AgentReply
    results: list[SimulationResult]
    transcript: list[tuple[str, str]]


MAX_TURNS = 6


async def simulate(
    prompt: str,
    *,
    model: Model | str,
    ui: UserInterface,
    output_root: str | Path,
    max_turns: int = MAX_TURNS,
) -> Outcome:
    """Run the agent on one user request. `results` come from the simulator, not the LLM.

    Conversation loop: if the model answers with a question (text containing '?') before any
    simulation has run, the question goes to the human and their answer is fed back in.
    """
    deps = AgentDeps(ui=ui, output_root=Path(output_root))
    agent = build_agent(model)
    history = None
    next_prompt = prompt
    nudges_left = 1
    for _ in range(max_turns):
        run = await agent.run(next_prompt, deps=deps, message_history=history)
        text = run.output.strip()
        history = run.all_messages()
        if "<tool_call>" in text and not deps.results and nudges_left:
            # Some local models print their tool call as text instead of making it. Ask once more.
            nudges_left -= 1
            next_prompt = (
                "Your last message printed a tool call as text, so nothing ran. "
                "Call the run_simulation tool properly now, using only the details already agreed."
            )
            continue
        if deps.results or "?" not in text:
            return Outcome(AgentReply(message=text), deps.results, list(ui.transcript))
        next_prompt = ui.ask(text)  # the model asked in plain text: pass it on to the human
    reply = AgentReply(message="Stopped: too many clarification rounds without a simulation.")
    return Outcome(reply, deps.results, list(ui.transcript))
