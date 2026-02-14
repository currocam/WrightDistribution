"""Shared test configuration."""

import jax

jax.config.update("jax_enable_x64", True)
