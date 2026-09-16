from abc import ABC, abstractmethod
from typing import Any


class BaseModelAdapter(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the required backend library is installed and importable."""
        pass

    @abstractmethod
    def build_estimator(
        self,
        task_type: str = "binary_classification",
        hyperparameters: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> Any:
        """Build and return an estimator instance conforming to scikit-learn API."""
        pass
