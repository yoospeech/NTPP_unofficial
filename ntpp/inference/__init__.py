"""NTPP inference utilities."""

__all__ = ["NTPPStreamingGenerator"]

def __getattr__(name):
    if name == "NTPPStreamingGenerator":
        from .streaming import NTPPStreamingGenerator

        return NTPPStreamingGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
