"""Dual fail-closed gate for the future GPU-only audit entry point."""

from __future__ import annotations

from typing import Mapping

from .protocol import ContractError


AUTHORIZATION_ENV = "OLMO_FUNCTION_MORPH_GPU_AUTHORIZED"
AUTHORIZATION_VALUE = "1"


class AuthorizationError(ContractError):
    """Raised before imports/model loading when explicit GPU authority is absent."""


def require_gpu_authorization(
    *, cli_authorize: bool, environment: Mapping[str, str]
) -> None:
    missing: list[str] = []
    if not cli_authorize:
        missing.append("--authorize")
    if environment.get(AUTHORIZATION_ENV) != AUTHORIZATION_VALUE:
        missing.append(f"{AUTHORIZATION_ENV}={AUTHORIZATION_VALUE}")
    if missing:
        raise AuthorizationError(
            "GPU audit is fail-closed before torch/model loading; missing "
            + " and ".join(missing)
        )
