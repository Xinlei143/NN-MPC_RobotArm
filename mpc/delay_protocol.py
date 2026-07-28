"""Canonical causal variants for fixed-delay residual MPC experiments."""

from __future__ import annotations

from dataclasses import dataclass


PROTOCOL_NAMES = (
    "full",
    "stale_history",
    "naive_delayed",
    "anchor_only",
    "no_future_alignment",
    "no_reanchor",
    "no_feedback",
)


@dataclass(frozen=True)
class DelayProtocol:
    name: str
    future_state: bool
    future_history: bool
    future_reference: bool
    reanchor_residual: bool
    feedback: bool

    @property
    def replay_absolute(self) -> bool:
        return not self.reanchor_residual


_PROTOCOLS = {
    "full": DelayProtocol("full", True, True, True, True, True),
    # Isolate recurrent-context alignment while retaining the forecast
    # activation state and activation-time reference.
    "stale_history": DelayProtocol("stale_history", True, False, True, True, True),
    "naive_delayed": DelayProtocol("naive_delayed", False, False, False, False, False),
    "anchor_only": DelayProtocol("anchor_only", True, True, True, False, False),
    # State forecast and activation-time reference shift are one future-
    # alignment module in the paper.  This variant removes both together.
    "no_future_alignment": DelayProtocol("no_future_alignment", False, False, False, True, True),
    "no_reanchor": DelayProtocol("no_reanchor", True, True, True, False, True),
    "no_feedback": DelayProtocol("no_feedback", True, True, True, True, False),
}


def resolve_delay_protocol(name: str) -> DelayProtocol:
    try:
        return _PROTOCOLS[str(name)]
    except KeyError as exc:
        raise ValueError(f"delay_protocol must be one of {PROTOCOL_NAMES}, got {name!r}") from exc
