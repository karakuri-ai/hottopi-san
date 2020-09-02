from abc import ABCMeta, abstractmethod
from typing import Sequence


class Classifier(metaclass=ABCMeta):
    """Classifierの抽象クラス"""

    @abstractmethod
    def fit(self, num_epochs=30, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict_np(self, text_list: Sequence[str]):
        raise NotImplementedError

    @abstractmethod
    def predict_with_attension(self, text_list: Sequence[str]):
        raise NotImplementedError

    @abstractmethod
    def save(self, save_path: str):
        raise NotImplementedError

    @abstractmethod
    def load(self, save_path: str):
        raise NotImplementedError
