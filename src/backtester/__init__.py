"""
Backtester package implementing SSOT v1.0 channel breakout prototype.

The package exposes a single convenience entrypoint `run_backtest_from_config`
that parses configuration, executes strategies across the requested symbols and
parameter grid, and triggers reporting/exports.
"""

from .cli import run_backtest_from_config

__all__ = ["run_backtest_from_config"]
