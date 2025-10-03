# Documentation for _toml_compat.py

```python
try:
    import tomllib
except ImportError:
    try:
        from pip._vendor import tomli as tomllib  # type: ignore[no-redef, unused-ignore]
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment, unused-ignore]

__all__ = ("tomllib",)

```