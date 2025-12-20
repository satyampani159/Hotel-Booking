from dataclasses import dataclass
import os


@dataclass
class DataIngestionConfig:
    data_dir: str
    data_file: str

    @property
    def data_path(self) -> str:
        return os.path.join(self.data_dir, self.data_file)


@dataclass
class TrainingConfig:
    test_size: float = 0.25
    random_state: int = 42
    model_dir: str = "artifacts/models"
    model_name: str = "logistic_model.joblib"
