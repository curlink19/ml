import torch

class TableDataGenerator:
    def __init__(self, generator):
        self.generator = generator

    def get(self, size):
        return torch.from_numpy(self.generator(size)).to(torch.float) 
