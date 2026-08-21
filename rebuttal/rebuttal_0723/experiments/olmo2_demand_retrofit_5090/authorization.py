"""Fail-closed authorization for the optional R4' GPU entry points.

The preparation bundle is read-only by default.  A future execution must
cross both an explicit CLI flag and an independently supplied environment
value.  Keeping this check in a small, torch-free module makes it testable
without touching a checkpoint or allocating a CUDA context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .protocol import ContractError

AUTHORIZATION_ENV = "R4PRIME_GPU_AUTHORIZED"
AUTHORIZED_VALUE = "1"


class AuthorizationError(ContractError):
    """Raised when one side of the dual execution gate is absent."""


@dataclass(frozen=True)
class AuthorizationDecision:
    action: str
    cli_flag: bool
    environment_value: str | None
    environment_gate: bool
    authorized: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "cli_flag": self.cli_flag,
            "environment_name": AUTHORIZATION_ENV,
            "environment_value_present": self.environment_value is not None,
            "environment_gate": self.environment_gate,
            "authorized": self.authorized,
        }


def authorization_decision(
    *,
    cli_authorize: bool,
    action: str,
    environment: Mapping[str, str] | None = None,
) -> AuthorizationDecision:
    """Return the two-gate decision without raising or performing I/O."""

    if not str(action).strip():
        raise ContractError("authorization action must be non-empty")
    values = environment if environment is not None else {}
    environment_value = values.get(AUTHORIZATION_ENV)
    environment_gate = environment_value == AUTHORIZED_VALUE
    return AuthorizationDecision(
        action=str(action),
        cli_flag=bool(cli_authorize),
        environment_value=environment_value,
        environment_gate=environment_gate,
        authorized=bool(cli_authorize) and environment_gate,
    )


def require_authorization(
    *,
    cli_authorize: bool,
    action: str,
    environment: Mapping[str, str] | None = None,
) -> AuthorizationDecision:
    """Require ``--authorize`` and ``R4PRIME_GPU_AUTHORIZED=1``.

    The function deliberately does not check CUDA or assets.  Those checks
    belong to the selected mature trainer and its bound receipts; this gate
    only prevents accidental entry into an external action.
    """

    decision = authorization_decision(
        cli_authorize=cli_authorize,
        action=action,
        environment=environment,
    )
    if decision.authorized:
        return decision
    missing: list[str] = []
    if not decision.cli_flag:
        missing.append("--authorize")
    if not decision.environment_gate:
        missing.append(f"{AUTHORIZATION_ENV}={AUTHORIZED_VALUE}")
    raise AuthorizationError(
        f"GPU action {decision.action!r} is fail-closed; missing "
        + " and ".join(missing)
    )

