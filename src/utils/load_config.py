import os
from functools import cache
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

class Config:
    class ConfigNotFoundError(Exception):
        pass

    def __init__(self):
        if not CONFIG_PATH.is_file():
            raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
        try:
            with open(CONFIG_PATH, "r") as file:
                self.config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise self.ConfigNotFoundError(f"Error parsing YAML file: {e}")

    def get_config_path(self, *keys) -> Path:
        """Like get_config_value, but returns an absolute Path rooted at PROJECT_ROOT."""
        return resolve_path(self.get_config_value(*keys))

    def get_config_value(self, *keys):
        path_dict = self.config
        for key in keys:
            path_dict = path_dict[key]
        return os.path.expandvars(path_dict)

@cache
def get_config():
    return Config()

def get_data_dir() -> Path:
    config = get_config()
    return config.get_config_path('paths', 'data')

def get_qdrant_host():
    config = get_config()
    return os.path.expandvars(config.get_config_path('urls', 'qdrant-host'))






