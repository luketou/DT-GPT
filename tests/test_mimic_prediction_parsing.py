import importlib.util
import pathlib
import sys
import types


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TEST_FILE = (
    _REPO_ROOT
    / "tests"
    / "pipeline"
    / "test_mimic_prediction_parsing.py"
)
sys.path.insert(0, str(_REPO_ROOT))
sys.modules.setdefault("__init__", types.ModuleType("__init__"))
_SPEC = importlib.util.spec_from_file_location("mimic_prediction_parsing_tests", _TEST_FILE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

globals().update(
    {
        name: value
        for name, value in vars(_MODULE).items()
        if name.startswith("ConvertFromStringsToDfTests")
    }
)
