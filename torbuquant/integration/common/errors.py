"""Shared integration exceptions."""

from __future__ import annotations


class IntegrationError(RuntimeError):
    """Base class for runtime integration failures."""


class OptionalDependencyError(IntegrationError):
    """Raised when an optional backend package is required but unavailable."""


class ProductionModeError(IntegrationError):
    """Raised when a requested production path cannot satisfy the contract."""

