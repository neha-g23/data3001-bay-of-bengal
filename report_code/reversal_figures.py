"""Use the main reversal_figures.py from the repository root, so there is only one copy.

`import reversal_figures` from any script in this folder loads ../reversal_figures.py.
"""
import importlib.util
import os
import sys

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reversal_figures.py")
_spec = importlib.util.spec_from_file_location(__name__, _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
