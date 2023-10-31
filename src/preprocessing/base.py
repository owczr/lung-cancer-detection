from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    def __init__(self, path: str):
        self.path = path

    @abstractmethod 
    def process(self):
        pass
    
    @abstractmethod
    def process_and_save(self):
        pass
    