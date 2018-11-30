from abc import ABC, abstractmethod


class ModelFactory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def get_generators(self):
        pass