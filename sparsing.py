import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx
from networkit.nxadapter import nx2nk
from descriptors import (
    JaccardCalc,
    CommonNeighborsCalc,
    PreferentialAttachmentCalc,
    AdamicAdarCalc,
    AdjustedRandCalc,
    AlgebraicDistanceCalc,
    KatzCalc,
)


class BaseSparsing(BaseTransform):
    def __init__(self, power: float=None):
        super(BaseSparsing, self).__init__()
        self.power = power
        
    def __call__(self, data: Data) -> Data:
        raise NotImplementedError


class Random(BaseSparsing):
    def __init__(self, power: float=None):
        super(Random, self).__init__(power=power)

    def __call__(self, data: Data) -> Data:
        if self.power is not None:
            edge_index = data.edge_index
            percentage_num_nodes = int(edge_index.size(1) * self.power)
            data.edge_index = edge_index[:, :-percentage_num_nodes]
        return data


class ArithmeticNorm(BaseSparsing):
    def __init__(self, power: int=None):
        super(ArithmeticNorm, self).__init__(power=power)
        
    def __arithmetic(self, x, y):
        return x + y
    
    def __call__(self, data: Data) -> Data:
        if self.power is not None:
            vertex_degrees_size = torch.max(data.edge_index) + 1
            vertex_degrees = torch.zeros(vertex_degrees_size, dtype=torch.float)
            vertex_degrees = torch.bincount(data.edge_index[0]) + torch.bincount(data.edge_index[1])
            
            edge_weights = self.__arithmetic(vertex_degrees[data.edge_index[0]], vertex_degrees[data.edge_index[1]])
            min_edge_weight = torch.min(edge_weights)
            max_edge_weight = torch.max(edge_weights)
            edge_weights = (edge_weights - min_edge_weight) / (max_edge_weight - min_edge_weight + torch.finfo(torch.float32).eps)
            
            edge_index = data.edge_index[:, edge_weights >= self.power]
            data.edge_index = edge_index
        
        return data


class GeometricNorm(BaseSparsing):
    def __init__(self, power: int=None):
        super(GeometricNorm, self).__init__(power=power)
        
    def __geometric(self, x, y):
        return torch.sqrt(x * y)   
    
    def __call__(self, data: Data) -> Data:
        if self.power is not None:
            vertex_degrees_size = torch.max(data.edge_index) + 1
            vertex_degrees = torch.zeros(vertex_degrees_size, dtype=torch.float)
            vertex_degrees = torch.bincount(data.edge_index[0]) + torch.bincount(data.edge_index[1])
            
            edge_weights = self.__geometric(vertex_degrees[data.edge_index[0]], vertex_degrees[data.edge_index[1]])
            min_edge_weight = torch.min(edge_weights)
            max_edge_weight = torch.max(edge_weights)
            edge_weights = (edge_weights - min_edge_weight) / (max_edge_weight - min_edge_weight + torch.finfo(torch.float32).eps)
            
            edge_index = data.edge_index[:, edge_weights >= self.power]
            data.edge_index = edge_index
        
        return data


class HarmonicNorm(BaseSparsing):
    def __init__(self, power: int=None):
        super(HarmonicNorm, self).__init__(power=power)
        
    def __harmonic(self, x, y):
        return x * y / (x + y)
    
    def __call__(self, data: Data) -> Data:
        if self.power is not None:
            vertex_degrees_size = torch.max(data.edge_index) + 1
            vertex_degrees = torch.zeros(vertex_degrees_size, dtype=torch.float)
            vertex_degrees = torch.bincount(data.edge_index[0]) + torch.bincount(data.edge_index[1])
            
            edge_weights = self.__harmonic(vertex_degrees[data.edge_index[0]], vertex_degrees[data.edge_index[1]])
            min_edge_weight = torch.min(edge_weights)
            max_edge_weight = torch.max(edge_weights)
            edge_weights = (edge_weights - min_edge_weight) / (max_edge_weight - min_edge_weight + torch.finfo(torch.float32).eps)
            
            edge_index = data.edge_index[:, edge_weights >= self.power]
            data.edge_index = edge_index
        
        return data


class IndexMain(BaseSparsing):
    def __init__(self, power: int=None, calc=None):
        super(IndexMain, self).__init__(power=power)
        self._calc = calc
    
    def _main_calc(self, G):
        return self._calc.run(G)
    
    def __call__(self, data: Data) -> Data:
        if self.power is not None:
            edge_index = data.edge_index
            mask = edge_index[0] < edge_index[1]
            edge_index = edge_index[:, mask]
            G = to_networkx(data, to_undirected=True)
            G = nx2nk(G)
            edge_weights = self._main_calc(G)
            edge_index = edge_index[:, edge_weights >= self.power]
            edge_index = torch.cat([edge_index, torch.flip(edge_index, dims=[0])], dim=1)
            data.edge_index = edge_index
        return data


class Jaccard(IndexMain):
    def __init__(self, power: int=None, calc=JaccardCalc()):
        super(Jaccard, self).__init__(power=power, calc=calc)
    

class CommonNeighbor(IndexMain):
    def __init__(self, power: int=None, calc=CommonNeighborsCalc()):
        super(CommonNeighbor, self).__init__(power=power, calc=calc)

    
class PreferentialAttachment(IndexMain):
    def __init__(self, power: int=None, calc=PreferentialAttachmentCalc()):
        super(PreferentialAttachment, self).__init__(power=power, calc=calc)


class AdamicAdar(IndexMain):
    def __init__(self, power: int=None, calc=AdamicAdarCalc()):
        super(AdamicAdar, self).__init__(power=power, calc=calc)


class AdjustedRand(IndexMain):
    def __init__(self, power: int=None, calc=AdjustedRandCalc()):
        super(AdjustedRand, self).__init__(power=power, calc=calc)


class AlgebraicDistance(IndexMain):
    def __init__(self, power: int=None, calc=AlgebraicDistanceCalc()):
        super(AlgebraicDistance, self).__init__(power=power, calc=calc)


class Katz(IndexMain):
    def __init__(self, power: int=None, calc=KatzCalc()):
        super(Katz, self).__init__(power=power, calc=calc)
