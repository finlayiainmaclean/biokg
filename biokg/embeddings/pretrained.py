from torch_geometric.data import Data

from biokg.embeddings.base import EmbeddingModel


class PretrainedEmbedding(EmbeddingModel):
    def __init__(self,
                 data: Data):
        super().__init__(data)

    def forward(self):
        return self.data.x_pretrained
        