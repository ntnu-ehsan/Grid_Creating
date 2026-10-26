"""Power system data preparation.

Builds a consistent power-system dataset for a configured case study at a
configurable spatial resolution and writes it in IAMC format, together with
an HTML report and an interactive map.
"""

__version__ = "1.0.0"

from .config import Config, load_config  # noqa: F401

__all__ = ["Config", "load_config", "__version__"]
