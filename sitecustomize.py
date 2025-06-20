# Ensure CI stubs load before anything else
try:
    from herg import _ci_stubs  # noqa: F401
except Exception:
    pass
