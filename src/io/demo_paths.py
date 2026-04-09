from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO_OUTPUT_ROOT = ROOT / "demo_output"


def _mkdir_or_raise(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"cannot write to demo output directory: {path}. "
            f"Please make sure {DEMO_OUTPUT_ROOT} is writable inside the container."
        ) from exc


def ensure_demo_dir(*parts):
    path = DEMO_OUTPUT_ROOT.joinpath(*parts)
    _mkdir_or_raise(path)
    return path


def demo_output_path(*parts):
    path = DEMO_OUTPUT_ROOT.joinpath(*parts)
    _mkdir_or_raise(path.parent)
    return path
