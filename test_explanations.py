#!/usr/bin/env python

"""
Test script for explanations with dummy data
"""

import numpy as np
from WLTLS.decoding import HeaviestPaths, expLoss
from graphs import TrellisGraph
from WLTLS.codeManager import GreedyCodeManager
from WLTLS.mainModels.finalModel import FinalModel

# Create a small test model
LABELS = 5
FEATURES = 10
DIMS = FEATURES
slice_width = 3

# Create the graph structure
trellisGraph = TrellisGraph(LABELS, slice_width)
heaviestPaths = HeaviestPaths(trellisGraph, loss=expLoss)
codeManager = GreedyCodeManager(LABELS, heaviestPaths.allCodes())

# Create a mock model class to pass to FinalModel
class MockModel:
    def __init__(self, dims, edges):
        self._learners = [MockLearner(dims) for _ in range(edges)]

class MockLearner:
    def __init__(self, dims):
        self.mean = np.random.randn(dims) * 0.1

# Create a dummy model
num_edges = heaviestPaths._edgesNum
mock_model = MockModel(DIMS, num_edges)
finalModel = FinalModel(DIMS, mock_model, codeManager, heaviestPaths)

# Create a test sample
x = np.random.rand(DIMS)

print("\nTesting explanation with dummy data:")
print(f"Model: {DIMS} features, {LABELS} labels, slice width {slice_width}")
print(f"Graph has {num_edges} edges")

# Generate explanation
explanation = finalModel.explain_prediction(x, k=3)

# Display explanation
print("\nExplanation results:")
for pred in explanation:
    print(f"Rank {pred['rank']}: Label {pred['predicted_label']} (Score: {pred['path_score']:.4f})")
    print(f"Confidence: {pred['confidence']:.4f}")
    print("Top features:")
    if pred['important_features']:
        for feat_idx, importance in pred['important_features']:
            print(f"  Feature {feat_idx}: {importance:.4f}")
    else:
        print("  No significant features identified")
    print()

print("Test completed") 