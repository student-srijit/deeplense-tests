"""How the agent talks to the human.

The agent never calls input() itself. It goes through a UserInterface, which gives us
- a real console / notebook version (ConsoleUI),
- a scripted version for tests and reproducible demos (ScriptedUI),
and both keep a transcript so the human-in-the-loop steps are visible afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


class UserInterface(Protocol):
    transcript: list[tuple[str, str]]

    def ask(self, question: str) -> str:
        """Ask the user a free-text follow-up question and return their answer."""

    def confirm(self, summary: str) -> tuple[bool, str]:
        """Show the planned simulation. Return (approved, comment_if_declined)."""


@dataclass
class ConsoleUI:
    """Uses input(), which also works inside a Jupyter notebook."""

    transcript: list[tuple[str, str]] = field(default_factory=list)

    def ask(self, question: str) -> str:
        answer = input(f"\nAgent asks: {question}\nYour answer: ").strip()
        self.transcript += [("agent", question), ("user", answer)]
        return answer

    def confirm(self, summary: str) -> tuple[bool, str]:
        reply = input(f"\nAbout to run:\n{summary}\nRun it? [y / or type what to change]: ").strip()
        approved = reply.lower() in {"y", "yes", ""}
        self.transcript += [("agent", f"Confirm: {summary}"), ("user", reply or "y")]
        return approved, "" if approved else reply


@dataclass
class PersonaUI:
    """A simulated human who answers by keyword, so the order of the agent's questions does not matter.

    profile maps a keyword to the answer, e.g. {"substructure": "cdm", "how many": "3"}.
    Only the sentences that contain a '?' are searched (so a restated "we will use Model_I" does
    not count), and the first profile entry that matches wins, so list specific keywords first.
    No match gives `fallback`. Plans are approved unless `approve` is False.
    """

    profile: dict[str, str]
    fallback: str = "I do not mind, please choose a sensible default."
    approve: bool = True
    transcript: list[tuple[str, str]] = field(default_factory=list)

    def ask(self, question: str) -> str:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", question) if "?" in s]
        asked = " ".join(sentences).lower() or question.lower()
        answer = next((a for key, a in self.profile.items() if key in asked), self.fallback)
        self.transcript += [("agent", question), ("user", answer)]
        return answer

    def confirm(self, summary: str) -> tuple[bool, str]:
        self.transcript += [("agent", f"Confirm: {summary}"), ("user", "yes" if self.approve else "no")]
        return self.approve, "" if self.approve else "not this one"


@dataclass
class ScriptedUI:
    """Pre-written answers. `answers` feeds ask(); `approvals` feeds confirm().

    An approval is True (run it) or a string (decline, with that as the change request).
    Running out of answers raises, so a test cannot silently pass with a chatty agent.
    """

    answers: list[str] = field(default_factory=list)
    approvals: list[bool | str] = field(default_factory=list)
    transcript: list[tuple[str, str]] = field(default_factory=list)

    def ask(self, question: str) -> str:
        if not self.answers:
            raise RuntimeError(f"ScriptedUI has no answer left for: {question!r}")
        answer = self.answers.pop(0)
        self.transcript += [("agent", question), ("user", answer)]
        return answer

    def confirm(self, summary: str) -> tuple[bool, str]:
        if not self.approvals:
            raise RuntimeError("ScriptedUI has no approval left to give.")
        decision = self.approvals.pop(0)
        approved = decision is True
        self.transcript += [("agent", f"Confirm: {summary}"), ("user", "yes" if approved else str(decision))]
        return approved, "" if approved else str(decision)
