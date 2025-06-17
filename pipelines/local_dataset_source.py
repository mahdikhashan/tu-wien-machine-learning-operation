import os
import shutil

from typing import Any, Dict

from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource


class CustomLocalDatasetSource(FileSystemDatasetSource):
    """
    A concrete implementation of FileSystemDatasetSource for local file paths.
    """
    def __init__(self, path: str = ""):
        self._path = os.path.abspath(path) if len(path) else path
        if self._path:
            if not os.path.exists(self._path):
                raise FileNotFoundError(f"Local dataset path not found: {self._path}")

    @property
    def uri(self) -> str:
        """Returns the absolute URI (path) of the local dataset."""
        if self._path:
            return self._path
        
        return "no uri"

    @staticmethod
    def _get_source_type() -> str:
        """Returns a string identifying this as a custom local filesystem source."""
        return "custom_local_fs"

    def load(self, dst_path: str = None) -> str:
        """
        Loads the dataset to a local filesystem path.
        If dst_path is provided, copies the file/directory there.
        If dst_path is None, returns the original local path.
        """
        if dst_path:
            os.makedirs(dst_path, exist_ok=True)
            destination_path = os.path.join(dst_path, os.path.basename(self._path))
            if os.path.isfile(self._path):
                shutil.copy(self._path, destination_path)
            elif os.path.isdir(self._path):
                shutil.copytree(self._path, destination_path, dirs_exist_ok=True)
            else:
                raise ValueError(f"Path '{self._path}' is neither a file nor a directory.")
            return destination_path
        else:
            return self._path

    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        """
        Determines if this source can resolve the raw source.
        Here, it checks if the raw_source is a string and exists as a local file/directory.
        """
        return isinstance(raw_source, str) and os.path.exists(raw_source)

    @classmethod
    def _resolve(cls, raw_source: Any) -> "CustomLocalDatasetSource":
        """
        Creates an instance of CustomLocalDatasetSource from a raw source string.
        This is typically called by MLflow's internal machinery.
        """
        if not cls._can_resolve(raw_source):
            raise ValueError(f"Raw source '{raw_source}' cannot be resolved by CustomLocalDatasetSource.")
        return cls(raw_source)

    def to_dict(self) -> Dict[Any, Any]:
        """
        Converts the source to a JSON-serializable dictionary.
        This is used for storing the dataset source information in MLflow's metadata.
        """
        return {"path": self._path}

    @classmethod
    def from_dict(cls, source_dict: Dict[Any, Any]) -> "CustomLocalDatasetSource":
        """
        Reconstructs a CustomLocalDatasetSource instance from a dictionary.
        This is used when loading dataset source information from MLflow's metadata.
        """
        if "path" not in source_dict:
            raise ValueError("Missing 'path' in source dictionary for CustomLocalDatasetSource.")
        return cls(source_dict["path"])
