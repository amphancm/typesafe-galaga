"""TypeSafe-powered direct controller for NES Galaga."""

from .actions import Action
from .state import GalagaSnapshot, GalagaStateParser

__all__ = ["Action", "GalagaSnapshot", "GalagaStateParser"]
