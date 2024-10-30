
import torch as th
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GraphSAGE
import torch.nn.functional as F
from biokg.embeddings.base import EmbeddingModel


          
class GraphSAGEEmbedding(EmbeddingModel):
    def __init__(self,
                 data:Data,
                 batch_size: int = 128,
                 embedding_dim: int = 256,
                 num_layers: int = 3,
                neg_sampling_ratio:float = 2.0,
                lr: float = 0.001,
                num_epochs: int = 20):
        super().__init__(data)
        self.train_loader = LinkNeighborLoader(
            self.data,
            batch_size=batch_size,
            shuffle=True,
            neg_sampling_ratio=neg_sampling_ratio,
            num_neighbors=[10, 10],
        )
        self.model = GraphSAGE(
            self.data.num_node_features,
            hidden_channels=embedding_dim,
            num_layers=num_layers,
        ).to(self.device)   
        self.optimizer = th.optim.Adam(list(self.parameters()), lr=lr)
        self.num_epochs = num_epochs

    def forward(self):
        # Assume the whole graph fits in memory
        return self.model(self.data.x, self.data.edge_index).cpu()

    def train_one_epoch(self):
        self.model.train()
        
        total_loss = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            h = self.model(batch.x, batch.edge_index)
            h_src = h[batch.edge_label_index[0]]
            h_dst = h[batch.edge_label_index[1]]
            # pred = (h_src * h_dst).sum(dim=-1)
            # Use the same distance metric as node2vec https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/node2vec.py#L14
            pred = (h_src * h_dst).sum(dim=-1) 
            loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * pred.size(0)
    
        return total_loss / self.data.num_nodes
