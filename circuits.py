from abc import ABC


class Circuit(ABC)
    
    def __init__(self):
        pass
    
    @abstractmethod
    def encode(x, params):
        pass