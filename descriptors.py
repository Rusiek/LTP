import numpy as np
from networkit.centrality import Betweenness
from networkit.distance import APSP
from networkit.graph import Graph
from networkit.linkprediction import (
    JaccardIndex,
    CommonNeighborsIndex,
    PreferentialAttachmentIndex,
    AdamicAdarIndex,
    AdjustedRandIndex,
    AlgebraicDistanceIndex,
    KatzIndex,
    
)
from networkit.sparsification import LocalDegreeScore


def calculate_shortest_paths(graph: Graph) -> np.array:
    # Networkit is faster than NetworkX for large graphs
    apsp = APSP(graph)
    apsp.run()
    path_lengths = apsp.getDistances(asarray=True)

    path_lengths = path_lengths.ravel()

    # filter out 0 length "paths" from node to itself
    path_lengths = path_lengths[np.nonzero(path_lengths)]

    # Networkit assigns extremely high values (~1e308) to mark infinite
    # distances for disconnected components, so we simply drop them
    path_lengths = path_lengths[path_lengths < 1e100]

    return path_lengths


def calculate_edge_betweenness(graph: Graph) -> np.ndarray:
    betweeness = Betweenness(graph, computeEdgeCentrality=True)
    betweeness.run()
    scores = betweeness.edgeScores()
    scores = np.array(scores, dtype=np.float32)
    return scores


class Calculate:
    def __init__(self, method, minmax=False, norm=False):
        self._method = method
        self._minmax = minmax
        self._norm = norm
    
    def run(self, graph: Graph) -> np.ndarray:
        index = self._method(graph)
        scores = [index.run(u, v) for u, v in graph.iterEdges()]
        scores = np.array(scores, dtype=np.float32)
        scores = scores[np.isfinite(scores)]
        if self._minmax:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + np.finfo(np.float32).eps)
        if self._norm:
            scores = (scores - np.mean(scores)) / (np.std(scores) + np.finfo(np.float32).eps)
        return scores

  
class JaccardCalc(Calculate):    
    def __init__(self, method=JaccardIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)


class CommonNeighborsCalc(Calculate):
    def __init__(self, method=CommonNeighborsIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)


class PreferentialAttachmentCalc(Calculate):
    def __init__(self, method=PreferentialAttachmentIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)


class AdamicAdarCalc(Calculate):
    def __init__(self, method=AdamicAdarIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)


class AdjustedRandCalc(Calculate):
    def __init__(self, method=AdjustedRandIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)


class AlgebraicDistanceCalc(Calculate):
    def __init__(self, method=AlgebraicDistanceIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)


class KatzCalc(Calculate):
    def __init__(self, method=KatzIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)


def calculate_local_degree_score(graph: Graph) -> np.ndarray:
    local_degree_score = LocalDegreeScore(graph)
    local_degree_score.run()
    scores = local_degree_score.scores()
    return np.array(scores, dtype=np.float32)
