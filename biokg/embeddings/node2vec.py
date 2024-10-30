
import torch as th
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data

from biokg.embeddings.base import EmbeddingModel


class Node2VecEmbedding(EmbeddingModel):
    def __init__(self,
                 data: Data,
                 embedding_dim: int = 512,
                 walk_length: int = 20,
                 context_size: int = 10,
                 walks_per_node: int = 10,
                 num_negative_samples: int = 1,
                 p: float = 1.0, q: float = 1.0,
                 batch_size: int = 128,
                 sparse: bool = True,
                 lr: float = 0.01,
                 num_epochs: int = 20):
        super().__init__(data)
        
        self.num_epochs = num_epochs
        self.model = Node2Vec(
            self.data.edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            sparse=sparse,
        ).to(self.device)
        self.batch_size = batch_size
        self.optimizer = th.optim.SparseAdam(list(self.model.parameters()), lr=lr)  # Fix optimizer parameters

    def forward(self):
        return self.model().detach().cpu()

    def train_one_epoch(self):
        self.model.train()
        loader = self.model.loader(batch_size=self.batch_size, shuffle=True, num_workers=1)
        
        total_loss = 0
        for pos_rw, neg_rw in loader:
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        loss = total_loss / len(loader)
        return loss