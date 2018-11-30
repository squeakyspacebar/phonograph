from abc import ABC, abstractmethod


class ModelFactory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def get_generators(self):
        pass

    @abstractmethod
    def process_inputs(self, features):
        pass

    @abstractmethod
    def process_labels(self, labels):
        pass
