"""Resolving the installed package version — the blade runner's incept date.

Both the `vk --version` flag and the model artifact's incept stamp need to
know which build produced them, so the lookup lives here rather than inside
the CLI, where the models layer cannot reach it.
"""
from importlib.metadata import PackageNotFoundError, version as read_version


def resolve_version() -> str:
    """Return the installed package version, or a placeholder when unknown."""
    try:
        return read_version("voight-kampff")
    except PackageNotFoundError:  # running from a source tree without an install
        return "0.0.0+unknown"
