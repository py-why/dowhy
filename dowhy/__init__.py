import logging

from dowhy.causal_identifier import EstimandType, identify_effect, identify_effect_auto, identify_effect_id
from dowhy.causal_model import CausalModel

logging.getLogger(__name__).addHandler(logging.NullHandler())

#
# 0.0.0 is standard placeholder for poetry-dynamic-versioning
# any changes to this should not be checked in
#
__version__ = "0.0.0"


def enable_notebook_rendering():
    """Enable rich rendering of SymPy expressions in Jupyter notebooks.

    Call this function at the top of a Jupyter notebook to activate
    SymPy's pretty-printer (``sympy.init_printing``), which renders
    symbolic math — such as identified estimands — as nicely formatted
    LaTeX in notebook output cells.

    This function is intentionally **not** called automatically when
    DoWhy is imported, because ``sympy.init_printing`` replaces
    ``sys.displayhook`` globally and can interfere with other libraries
    (e.g. PyTorch) that display non-SymPy objects in the same session.

    Example::

        import dowhy
        dowhy.enable_notebook_rendering()

    """
    from sympy import init_printing

    init_printing()


__all__ = [
    "EstimandType",
    "identify_effect_auto",
    "identify_effect_id",
    "identify_effect",
    "CausalModel",
    "enable_notebook_rendering",
]
