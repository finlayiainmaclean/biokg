from abc import ABC
import torch as th
from tqdm.auto import tqdm
import copy
import typing


class EmbeddingModel(th.nn.Module, ABC):
    def __init__(self, data):
        super().__init__()  # Add this line to initialize nn.Module
        self.device = 'cuda' if th.cuda.is_available() else 'cpu' 
        data = copy.deepcopy(data)
        self.node_type = data.node_type
        self.name = data.name
        for key in data.keys():
            if not isinstance(data[key], typing.Hashable):
                del data[key]
        self.data = data.to(self.device)

    def train_one_epoch(self):
        ...

    def train(self):
        for epoch in tqdm(range(1,self.num_epochs+1)):
            loss = self.train_one_epoch()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}', end="\r")

