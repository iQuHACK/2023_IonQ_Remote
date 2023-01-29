from abc import ABC

class Simulator(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def simulate(circuit_fn):
        pass