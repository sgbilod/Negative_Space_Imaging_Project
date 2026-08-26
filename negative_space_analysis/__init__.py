"""
Negative Space Analysis Package
Provides core analysis algorithms and data structures for negative space imaging
"""

__version__ = "0.1.0"

# Import core analysis modules
try:
    from negative_space_analysis import analysis
    from negative_space_analysis import algorithms
except ImportError:
    pass

__all__ = [
    "analysis",
    "algorithms",
]
