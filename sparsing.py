from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class SparseRng(BaseTransform):
    def __init__(self, percentage: float=None, num_nodes: int=None):
        super(SparseRng, self).__init__()
        self.percentage = percentage
        self.num_nodes = num_nodes

    def __call__(self, data: Data) -> Data:
        edge_index = data.edge_index
        if self.percentage is not None:
            percentage_num_nodes = int(edge_index.size(1) * self.percentage)
        if self.percentage is not None and self.num_nodes is not None:
            data.edge_index = edge_index[:, :-min(percentage_num_nodes, self.num_nodes)]
        elif self.percentage is not None:
            data.edge_index = edge_index[:, :-percentage_num_nodes]
        elif self.num_nodes is not None:
            data.edge_index = edge_index[:, :-self.num_nodes]

        return data
    

class Sparse(BaseTransform):
    ...