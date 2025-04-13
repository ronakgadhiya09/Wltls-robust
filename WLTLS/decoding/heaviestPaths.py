"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
--------------------------------------------------------------------------------
See the paper for more details on the different losses.
"""

from WLTLS.decoding.decodingLosses import expLoss
from graphs import Graph, DAG_k_heaviest_path_lengths, restorePathsFromParents
import networkx as nx
import numpy as np
from aux_lib import pairwise, print_debug
from scipy.sparse import csr_matrix
import heapq
import math
import copy

#########################################################################
# Deccodes ECOC using a directed a-cyclic graph
#########################################################################
class HeaviestPaths():
    _DAG = None
    _edgeIndices = None
    _topologicalSort = None
    loss = None

    def __init__(self, DAG : Graph, loss = expLoss):
        self._DAG = DAG
        self._edges = list(self._DAG.edges())
        self.loss = loss

        # Assure the graph is dag, we don't check to save time
        assert (isinstance(self._DAG.G, nx.DiGraph) and nx.is_directed_acyclic_graph(self._DAG.G)), \
            "Graph must be directed a-cyclic"

        # Pre-compute the topological sort so we don't need to re-calculate it at every step
        self._topologicalSort = list(nx.topological_sort(self._DAG.G))

        self._edgeIndices = { e: idx for idx, e in enumerate(self.edges()) }

        self._edgesNum = DAG.edgesNum
        self._verticesNum = DAG.verticesNum

        self.sandboxW = np.zeros((self._verticesNum, self._verticesNum))

        self.initLossMatrix()

        print_debug("Created a heaviest path decoder with {} edges.".format(self._edgesNum))

    # Create an auxiliary binary matrix for *fast* calculation of losses (preparing weights)
    # Useful to employ the reduction to *any* loss function (See Section 5 in the paper)
    def initLossMatrix(self):
        self.lossMat = np.zeros((self._edgesNum,self._edgesNum))

        for i,slice in enumerate(self._DAG.edgesFromSlice):
            for e in slice:
                if not self._DAG.isEdgeShortcut(e):
                    for e2 in slice:
                        if e != e2:
                            self.lossMat[self._edgeIndices[e], self._edgeIndices[e2]] = 1
                else:
                    # If the edge is a shortcut,
                    # it should include (negative) terms from all edges which are farther from the source.
                    # See Appendix "Loss-based decoding generalization" for more details.
                    for slice2 in self._DAG.edgesFromSlice[i:]:
                        for e2 in slice2:
                            # We shouldn't include a (negative) term for the edge itself
                            if e == e2:
                                continue

                            self.lossMat[self._edgeIndices[e], self._edgeIndices[e2]] = 1

        self.lossMat = csr_matrix(self.lossMat)

    # Efficiently calculate the loss-based weights (details in the paper) using matrix multiplication
    def prepareWeights(self, responses):
        posLoss = self.loss(responses, 1)
        negLoss = self.loss(responses, -1)

        cumulative = self.lossMat.dot(negLoss)

        res = -(posLoss + cumulative)

        # Fill the results in the matrix structure
        for e in self.edges():
            self.sandboxW[e] = res[self._edgeIndices[e]]

        return self.sandboxW

    # Finds the best k best codes of the graph given the responses
    def findKBestCodes(self, responses, k:int=1):
        W = self.prepareWeights(responses)

        parents = DAG_k_heaviest_path_lengths(self._DAG.G, W, self._DAG.source,
                                              k=k, topologicalSort=self._topologicalSort)

        paths = restorePathsFromParents(parents, k, self._DAG.source, self._DAG.sink, W)

        return [self._pathToCode(path) for path in paths]

    def findKBestCodesWithImportance(self, responses, feature_weights, k=1):
        """Find k best codes and track feature importance along paths with sensitivity analysis
        
        Args:
            responses: Edge responses from the model
            feature_weights: Feature contributions matrix (features x edges)
            k: Number of best codes to find
            
        Returns:
            Tuple of (codes, path_attributions)
        """
        W = self.prepareWeights(responses)
        
        # Find k best paths
        parents = DAG_k_heaviest_path_lengths(self._DAG.G, W, self._DAG.source, 
                                            k=k, topologicalSort=self._topologicalSort)
        paths = restorePathsFromParents(parents, k, self._DAG.source, self._DAG.sink, W)
        
        codes = []
        path_attributions = []
        
        for path in paths:
            code = self._pathToCode(path)
            codes.append(code)
            
            # Initialize attribution tracking
            path_attribution = EdgeAttribution(0.0)
            path_score = 0.0
            
            # Track contributions for each edge in this path
            edge_count = 0
            for edge in path:
                edge_idx = self._edgeIndices[edge]
                edge_weight = float(W[edge])
                path_score += edge_weight
                edge_count += 1
                
                # Get the edge response (from model prediction)
                edge_response = responses[edge_idx]
                
                # Get feature contributions for this edge
                if feature_weights.ndim == 2:
                    # If feature_weights is a matrix (features x edges)
                    edge_features = feature_weights[:, edge_idx]
                    
                    # Use absolute values to get meaningful contributions
                    abs_contributions = np.abs(edge_features)
                    
                    # Get features with non-zero contribution
                    significant_indices = np.where(abs_contributions > 1e-6)[0]
                    
                    # If no significant features found, try with a lower threshold
                    if len(significant_indices) == 0:
                        significant_indices = np.where(abs_contributions > 0)[0]
                    
                    # If we found any significant features
                    if len(significant_indices) > 0:
                        # Sort by contribution (most important first)
                        sorted_indices = significant_indices[np.argsort(-abs_contributions[significant_indices])]
                        
                        # Take top features (up to 10 per edge)
                        for feat_idx in sorted_indices[:10]:
                            feat_contribution = edge_features[feat_idx]
                            # Scale contribution by edge importance in the path
                            normalized_contribution = (feat_contribution / edge_count) * (edge_weight / (path_score or 1.0))
                            path_attribution.add_feature_contribution(int(feat_idx), float(normalized_contribution))
            
            # Set total path weight and ensure it has non-empty feature weights
            path_attribution.total_weight = path_score
            
            # If we somehow ended up with empty feature weights, add something
            if not path_attribution.feature_weights:
                # Add a dummy feature as a last resort (shouldn't happen with fixes above)
                path_attribution.add_feature_contribution(0, 1.0)
                print("Warning: No feature attributions found for path, adding dummy feature")
            
            path_attributions.append(path_attribution)
                
        return codes, path_attributions

    def _compute_sensitivity(self, edge_features, edge_response, edge_weight):
        """Perform sensitivity analysis by perturbing features and observing changes
        
        Args:
            edge_features: Feature weights for an edge
            edge_response: Response value for the edge
            edge_weight: Computed edge weight
            
        Returns:
            Dictionary mapping feature indices to sensitivity scores
        """
        sensitivity_scores = {}
        perturbation = 0.01  # Small perturbation value
        
        # Find features with significant contribution
        significant_indices = np.where(np.abs(edge_features) > 1e-10)[0]
        
        for feat_idx in significant_indices:
            feat_weight = edge_features[feat_idx]
            original_contribution = feat_weight * edge_response * edge_weight

            # Perturb the feature weight
            perturbed_feat_weight = feat_weight * (1 + perturbation)
            perturbed_contribution = perturbed_feat_weight * edge_response * edge_weight

            # Compute sensitivity as the change in contribution
            sensitivity = (perturbed_contribution - original_contribution) / perturbation
            sensitivity_scores[int(feat_idx)] = float(sensitivity)

        return sensitivity_scores

    def edges(self):
        return self._edges

    # Returns all possible paths of the saved graph
    def allCodes(self):
        for path in nx.all_simple_paths(self._DAG.G, source=self._DAG.source, target=self._DAG.sink):
            yield self._pathToCode(pairwise(path))

    def _pathToCode(self, path):
        # Creates a -1,1 matrix of the paths over the edges
        code = [-1] * self._edgesNum
        for e in path:
            code[self._edgeIndices[e]] = 1

        return code

class EdgeAttribution:
    """Tracks detailed feature contributions for trellis graph edges"""
    
    def __init__(self, weight, features=None):
        self.total_weight = weight
        self.feature_weights = {} if features is None else features 
        self.interaction_weights = {}
        self.edge_contributions = {}
        
    def add_feature_contribution(self, feature_name, contribution, edge_idx=None):
        """Add or update a feature's contribution
        
        Args:
            feature_name: Name/index of the feature
            contribution: Contribution value
            edge_idx: Optional edge index for tracking edge-specific contributions
        """
        self.feature_weights[feature_name] = self.feature_weights.get(feature_name, 0) + contribution
        
        if edge_idx is not None:
            if edge_idx not in self.edge_contributions:
                self.edge_contributions[edge_idx] = {}
            self.edge_contributions[edge_idx][feature_name] = contribution
        
    def add_interaction_effect(self, feature_pair, contribution, edge_idx=None):
        """Track contribution from feature interactions
        
        Args:
            feature_pair: Tuple of interacting feature names/indices
            contribution: Interaction contribution value 
            edge_idx: Optional edge index for tracking edge-specific interactions
        """
        self.interaction_weights[feature_pair] = self.interaction_weights.get(feature_pair, 0) + contribution
        
        if edge_idx is not None:
            if edge_idx not in self.edge_contributions:
                self.edge_contributions[edge_idx] = {}
            self.edge_contributions[edge_idx][feature_pair] = contribution
            
    def get_feature_importance(self, normalize=True, edge_idx=None):
        """Get normalized feature importance scores
        
        Args:
            normalize: Whether to normalize scores to sum to 1
            edge_idx: Optional edge index to get edge-specific scores
            
        Returns:
            Dict mapping features to their importance scores
        """
        if edge_idx is not None and edge_idx in self.edge_contributions:
            scores = self.edge_contributions[edge_idx]
        else:
            scores = self.feature_weights.copy()
            
        if normalize:
            total = sum(abs(v) for v in scores.values())
            if total > 1e-10:
                return {k: v/total for k,v in scores.items()}
        return scores

    def get_interaction_importance(self, normalize=True, edge_idx=None):
        """Get feature interaction importance scores
        
        Args:
            normalize: Whether to normalize scores to sum to 1
            edge_idx: Optional edge index to get edge-specific scores
            
        Returns:
            Dict mapping feature pairs to interaction scores
        """
        if edge_idx is not None and edge_idx in self.edge_contributions:
            scores = {k:v for k,v in self.edge_contributions[edge_idx].items() 
                     if isinstance(k, tuple)}
        else:
            scores = self.interaction_weights
            
        if normalize and scores:
            total = sum(abs(v) for v in scores.values())
            if total > 1e-10:
                return {k: v/total for k,v in scores.items()}
        return scores
        
    def merge(self, other):
        """Merge another EdgeAttribution object into this one"""
        self.total_weight += other.total_weight
        
        # Merge feature weights
        for feat, weight in other.feature_weights.items():
            self.feature_weights[feat] = self.feature_weights.get(feat, 0) + weight
            
        # Merge interaction weights    
        for pair, weight in other.interaction_weights.items():
            self.interaction_weights[pair] = self.interaction_weights.get(pair, 0) + weight
            
        # Merge edge-specific contributions
        for edge_idx, contributions in other.edge_contributions.items():
            if edge_idx not in self.edge_contributions:
                self.edge_contributions[edge_idx] = {}
            for feat, weight in contributions.items():
                self.edge_contributions[edge_idx][feat] = (
                    self.edge_contributions[edge_idx].get(feat, 0) + weight
                )
        return self

    def get_top_features(self, n=5, edge_idx=None):
        """Get the top N most important features and their scores
        
        Args:
            n: Number of top features to return
            edge_idx: Optional edge index to get edge-specific scores
            
        Returns:
            List of tuples (feature_name, score) sorted by absolute score
        """
        # First try to get normalized scores
        scores = self.get_feature_importance(normalize=True, edge_idx=edge_idx)
        
        # If no scores available, use raw feature weights
        if not scores and self.feature_weights:
            scores = self.feature_weights
        
        # If still no scores, return empty list
        if not scores:
            print("Warning: No feature scores available")
            return []
        
        # Sort features by absolute score value
        sorted_features = sorted(scores.items(), 
                               key=lambda x: abs(x[1]), 
                               reverse=True)
        
        # Make sure we convert to proper numeric types
        result = [(int(idx) if isinstance(idx, (int, float, np.integer)) else idx, 
                   float(score)) 
                  for idx, score in sorted_features[:n]]
        
        return result