#!/usr/bin/env python

"""
Test script for perturbation-based explanations with dummy data
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
print("Creating trellis graph...")
trellisGraph = TrellisGraph(LABELS, slice_width)
heaviestPaths = HeaviestPaths(trellisGraph, loss=expLoss)
codeManager = GreedyCodeManager(LABELS, heaviestPaths.allCodes())

# Create a mock model class to pass to FinalModel
class MockModel:
    def __init__(self, dims, edges):
        # Create learners with predictable weights for interpretability testing
        self._learners = []
        for i in range(edges):
            learner = MockLearner(dims)
            # Make weights more interpretable - feature i strongly influences edge i % dims
            for j in range(dims):
                if j == (i % dims):
                    learner.mean[j] = 1.0  # Strong positive weight
                elif j == ((i + 1) % dims):
                    learner.mean[j] = -0.5  # Moderate negative weight
                else:
                    learner.mean[j] = 0.1 * np.random.randn()  # Small random weights
            self._learners.append(learner)

class MockLearner:
    def __init__(self, dims):
        self.mean = np.zeros(dims)

# Create a dummy model
print("Creating mock model...")
num_edges = heaviestPaths._edgesNum
mock_model = MockModel(DIMS, num_edges)
finalModel = FinalModel(DIMS, mock_model, codeManager, heaviestPaths)

# Create a test sample with clear patterns
x = np.zeros(DIMS)
x[0] = 1.0  # Strong feature
x[1] = 0.8  # Another strong feature
x[5] = 0.5  # Medium feature
x[9] = 0.2  # Weak feature

print("\nTesting perturbation-based explanation with dummy data:")
print(f"Model: {DIMS} features, {LABELS} labels, slice width {slice_width}")
print(f"Graph has {num_edges} edges")
print(f"Test sample: {x}")

# Generate explanation
print("\nGenerating explanations...")
explanation = finalModel.explain_prediction(x, k=3)

# Display explanation
print("\nExplanation results:")
for pred in explanation:
    print(f"Rank {pred['rank']}: Label {pred['predicted_label']} (Score: {pred['path_score']:.4f})")
    print(f"Confidence: {pred['confidence']:.2%}")
    print(f"Stability: {pred['prediction_stability']:.2%}")
    print("Top features:")
    if pred['important_features']:
        for feat_idx, importance, direction in pred['important_features']:
            print(f"  Feature {feat_idx}: {importance:.4f} {direction}")
    else:
        print("  No significant features identified")
    print()

print("Test completed") 