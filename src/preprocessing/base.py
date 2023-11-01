from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """
    Abstract base class for data processing.

    This class provides an interface for processing and saving data.
    
    Attributes:
        path (str): The path to the data to be processed.

    Methods:
        process: Abstract method to process the data.
        save: Abstract method to save the processed data.
        process_and_save: Abstract method to process and save the data in one call.
    """
    def __init__(self, path: str):
        self.path = path

    @abstractmethod 
    def process(self):
        pass

    @abstractmethod
    def save(self, path: str):
        pass
    
    @abstractmethod
    def process_and_save(self, path: str):
        pass
    