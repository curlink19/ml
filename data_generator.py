import torch


class TableDataGenerator:
    def __init__(self, generator):
        self.generator = generator

    def get(self, size=1):
        return [
            torch.from_numpy(self.generator()).to(torch.float)
            for i in range(size)
        ]
