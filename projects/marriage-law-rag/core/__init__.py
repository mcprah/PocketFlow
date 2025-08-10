#!/usr/bin/env python3
"""
Core module for application state management and lifecycle
"""

from .app_state import startup_handler, shutdown_handler

__all__ = ['startup_handler', 'shutdown_handler']