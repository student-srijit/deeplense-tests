"""Agent tests that need no real LLM: a scripted FunctionModel plays the model's part."""

import asyncio

import numpy as np
import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from deeplense_agent import PersonaUI, ScriptedUI, SimulationParams, simulate
from deeplense_agent import simulator

GOOD = {"model_name": "Model_I", "substructure": "cdm", "num_images": 1, "seed": 7}


def scripted_model(*steps):
    """Each step is a (tool_name, args) pair; the string 'final' means: give the final answer."""
    state = {"i": 0, "seen": []}

    def respond(messages, info):
        state["seen"].append(messages)
        tool, args = steps[state["i"]]
        state["i"] += 1
        if tool == "final":
            return ModelResponse(parts=[TextPart(args)])
        return ModelResponse(parts=[ToolCallPart(tool, args)])

    model = FunctionModel(respond)
    model.state = state
    return model


def run(prompt, model, ui, tmp_path):
    return asyncio.run(simulate(prompt, model=model, ui=ui, output_root=tmp_path))


# ---------- schema ----------


@pytest.mark.parametrize(
    "bad",
    [
        {**GOOD, "num_images": 1000},
        {**GOOD, "num_images": 0},
        {**GOOD, "z_halo": 2.0, "z_gal": 1.0},
        {**GOOD, "substructure": "vortex"},  # vortex without axion mass
        {**GOOD, "axion_mass": 1e-23},  # axion mass on a cdm lens
        {**GOOD, "substructure": "vortex", "axion_mass": 5.0},  # outside the allowed range
        {**GOOD, "model_name": "Model_IX"},
    ],
)
def test_bad_params_are_rejected(bad):
    with pytest.raises(ValidationError):
        SimulationParams(**bad)


def test_vortex_with_axion_mass_is_accepted():
    p = SimulationParams(**{**GOOD, "substructure": "vortex", "axion_mass": 1e-23})
    assert p.axion_mass == 1e-23


# ---------- simulator ----------


@pytest.mark.parametrize(
    "model_name, shape", [("Model_I", (150, 150)), ("Model_II", (64, 64))]
)
@pytest.mark.parametrize("substructure", ["no_sub", "cdm", "vortex"])
def test_simulator_shapes(tmp_path, model_name, shape, substructure):
    extra = {"axion_mass": 1e-23} if substructure == "vortex" else {}
    params = SimulationParams(
        model_name=model_name, substructure=substructure, num_images=2, seed=1, **extra
    )
    result = simulator.run_simulation(params, tmp_path)
    assert result.image_shape == shape
    assert len(result.files) == 2
    img = np.load(result.files[0])
    assert img.shape == shape and np.isfinite(img).all() and img.max() > 0


def test_same_seed_gives_same_images(tmp_path):
    p = SimulationParams(**GOOD)
    a = np.load(simulator.run_simulation(p, tmp_path / "a").files[0])
    b = np.load(simulator.run_simulation(p, tmp_path / "b").files[0])
    assert np.array_equal(a, b)


# ---------- agent flow ----------


def test_agent_asks_then_simulates(tmp_path):
    model = scripted_model(
        ("ask_user", {"question": "Which substructure do you want?"}),
        ("run_simulation", {"params": GOOD}),
        ("final", "Generated 1 CDM image."),
    )
    ui = ScriptedUI(answers=["cdm"], approvals=[True])
    out = run("make me a lens image", model, ui, tmp_path)

    assert ("agent", "Which substructure do you want?") in out.transcript
    assert ("user", "cdm") in out.transcript
    assert len(out.results) == 1
    assert out.results[0].params.substructure == "cdm"
    assert out.results[0].image_shape == (150, 150)
    assert out.reply.message == "Generated 1 CDM image."


def test_user_can_decline_and_change_the_plan(tmp_path):
    changed = {**GOOD, "num_images": 2}
    model = scripted_model(
        ("run_simulation", {"params": GOOD}),
        ("run_simulation", {"params": changed}),
        ("final", "Done with 2 images."),
    )
    ui = ScriptedUI(approvals=["make it 2 images", True])
    out = run("one cdm lens, Model_I", model, ui, tmp_path)

    assert len(out.results) == 1, "the declined plan must not have been simulated"
    assert len(out.results[0].files) == 2
    # the model was told why its first attempt was declined
    last_messages = model.state["seen"][1]
    assert "make it 2 images" in str(last_messages)


def test_invalid_parameters_go_back_to_the_model_without_running(tmp_path):
    model = scripted_model(
        ("run_simulation", {"params": {**GOOD, "num_images": 1000}}),
        ("run_simulation", {"params": GOOD}),
        ("final", "ok"),
    )
    ui = ScriptedUI(approvals=[True])  # only ONE approval: the invalid call must not reach the user
    out = run("1000 images please", model, ui, tmp_path)

    # what the model saw right after its invalid call: the validation error for num_images
    feedback = [
        str(part.content)
        for msg in model.state["seen"][1]
        for part in getattr(msg, "parts", [])
        if "less_than_equal" in str(getattr(part, "content", ""))
    ]
    assert feedback and "num_images" in feedback[0]
    assert len(out.results) == 1


def test_persona_answers_the_question_not_the_restated_choice():
    ui = PersonaUI(profile={"substructure": "cdm", "which model": "Model_II", "model": "Model_I"})
    # Restating "Model_I" in a sentence without a '?' must not be mistaken for the question.
    q = "Great, we will use Model_I. Which substructure do you want: no_sub, cdm or vortex?"
    assert ui.ask(q) == "cdm"
    assert ui.ask("Which model should I use?") == "Model_II"
    assert ui.ask("Anything else?") == ui.fallback


def test_plain_text_question_is_passed_to_the_human(tmp_path):
    # Small models often ask in plain text instead of calling ask_user; that must still work.
    model = scripted_model(
        ("final", "Which substructure would you like: no_sub, cdm or vortex?"),
        ("run_simulation", {"params": GOOD}),
        ("final", "Done."),
    )
    ui = ScriptedUI(answers=["cdm"], approvals=[True])
    out = run("make me some lenses", model, ui, tmp_path)

    assert ("agent", "Which substructure would you like: no_sub, cdm or vortex?") in out.transcript
    assert ("user", "cdm") in out.transcript
    assert len(out.results) == 1 and out.reply.message == "Done."


def test_tool_call_printed_as_text_gets_one_retry(tmp_path):
    leaked = '<tool_call>{"name": "run_simulation", "arguments": {"model_name": "Model_I"}}</tool_call>'
    model = scripted_model(
        ("final", leaked),
        ("run_simulation", {"params": GOOD}),
        ("final", "Done."),
    )
    out = run("one cdm image with Model_I", model, ScriptedUI(approvals=[True]), tmp_path)
    assert len(out.results) == 1 and out.reply.message == "Done."


def test_text_without_a_question_ends_the_run(tmp_path):
    model = scripted_model(("final", "That is not about lensing simulations."))
    out = run("what is the capital of France?", model, ScriptedUI(), tmp_path)
    assert out.results == [] and "not about lensing" in out.reply.message


def test_question_budget_is_enforced(tmp_path):
    asks = [("ask_user", {"question": f"q{i}"}) for i in range(6)]
    model = scripted_model(*asks, ("final", "gave up asking"))
    ui = ScriptedUI(answers=["a"] * 10)
    out = run("hello", model, ui, tmp_path)
    assert sum(1 for who, _ in out.transcript if who == "agent") == 4  # MAX_QUESTIONS
