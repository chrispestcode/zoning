import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def _find_project_root() -> Path:
    """Walk up from this file to the directory containing pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not locate project root (no pyproject.toml found)")


# Every relative path in this project is resolved against PROJECT_ROOT,
# never the current working directory.
PROJECT_ROOT = _find_project_root()


def resolve_path(path: str | os.PathLike) -> Path:
    """Expand env vars and resolve a relative path against PROJECT_ROOT."""
    p = Path(os.path.expandvars(str(path)))
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# CONFIG_PATH may be set in .env to override; relative values are rooted at PROJECT_ROOT.
CONFIG_PATH = resolve_path(os.getenv("CONFIG_PATH", "src/config.yaml"))


def get_logger():
    return logger


def load_config():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    try:
        with open(CONFIG_PATH, "r") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise Exception(f"Error parsing YAML file: {e}")


def get_data_dir() -> Path:
    config = load_config()
    return get_config_path(config, 'paths', 'data')


def get_qdrant_host():
    config = load_config()
    return os.path.expandvars(config['urls']['qdrant-host'])


def get_config_value(config, *keys):
    path_dict = config
    for key in keys:
        path_dict = path_dict[key]
    return os.path.expandvars(path_dict)


def get_config_path(config, *keys) -> Path:
    """Like get_config_value, but returns an absolute Path rooted at PROJECT_ROOT."""
    return resolve_path(get_config_value(config, *keys))
