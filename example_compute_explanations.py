#!/usr/bin/env python

"""
Example script for computing and visualizing perturbation-based explainability
on a pre-trained WLTLS model.
"""

from WLTLS.datasets import read, datasets
from WLTLS.decoding import HeaviestPaths
from WLTLS.mainModels.finalModel import FinalModel
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.sparse import csr_matrix

# Set argument parser
parser = argparse.ArgumentParser(description="Computes perturbation-based explanations for WLTLS predictions")
parser.add_argument("dataset", choices=[d.name for d in datasets.getAll()], help="Dataset name")
parser.add_argument("data_path", help="Path of the directory holding the datasets")
parser.add_argument("model_dir", help="Path of the directory containing the trained model")
parser.add_argument("-k", type=int, default=5, help="Number of top paths to analyze")
parser.add_argument("-n", type=int, default=10, help="Number of test samples to explain")
parser.add_argument("-features", type=int, default=10, help="Number of top features to display per path")
parser.add_argument("-seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
parser.add_argument("--output", type=str, default="explanations", help="Output directory for results")

args = parser.parse_args()

# Set random seed for reproducibility
np.random.seed(args.seed)

# Get dataset parameters
DATASET = datasets.getParams(args.dataset)

# Create output directory if it doesn't exist
if not os.path.exists(args.output):
    os.makedirs(args.output)

# Load the model
print(f"Loading model from {args.model_dir}...")
model_path = os.path.join(args.model_dir, "model.npz")
model_data = np.load(model_path, allow_pickle=True)

# Load dataset
print(f"Loading dataset {args.dataset}...")
Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest, LABELS, DIMS = read(args.data_path, DATASET)

# Load model components
W = model_data['W']
slice_width = model_data.get('slice_width', 2)

# Recreate graph and decoder
from graphs import TrellisGraph
trellisGraph = TrellisGraph(LABELS, slice_width)
from WLTLS.decoding import expLoss
heaviestPaths = HeaviestPaths(trellisGraph, loss=expLoss)

# Recreate code manager
from WLTLS.codeManager import GreedyCodeManager
codeManager = GreedyCodeManager(LABELS, heaviestPaths.allCodes())

# Create final model
finalModel = FinalModel(DIMS, None, codeManager, heaviestPaths)
finalModel.W = W

# Sample test instances for explanation
num_samples = min(args.n, Xtest.shape[0])
sample_indices = np.random.choice(Xtest.shape[0], num_samples, replace=False)

# Store explanation results
explanations = []

# Process each sample
print("\nGenerating perturbation-based explanations...")
for idx in sample_indices:
    # Convert to dense if needed
    if hasattr(Xtest, 'toarray'):
        x = Xtest[idx].toarray().ravel()
    else:
        x = Xtest[idx]
        
    y_true = Ytest[idx]
    
    # Get explanation
    explanation = finalModel.explain_prediction(x, k=args.k)
    
    # Store explanation with sample metadata
    explanations.append({
        'sample_idx': int(idx),
        'true_label': int(y_true),
        'predicted': int(explanation[0]['predicted_label'] if explanation[0]['predicted_label'] is not None else -1),
        'correct': int(y_true) == int(explanation[0]['predicted_label'] if explanation[0]['predicted_label'] is not None else -1),
        'stability': float(explanation[0]['prediction_stability']),
        'paths': explanation
    })
    
    # Print progress
    if (idx % 10) == 0:
        print(f"Processed {idx}/{num_samples} samples")

# Save explanations to file
print(f"Saving explanations to {args.output}...")

# Save as text file
with open(os.path.join(args.output, "perturbation_explanations.txt"), 'w') as f:
    f.write("WLTLS Perturbation-based Explanations\n")
    f.write("=" * 50 + "\n\n")
    
    for result in explanations:
        f.write(f"Sample {result['sample_idx']} (True label: {result['true_label']})\n")
        f.write(f"Prediction correct: {result['correct']}\n")
        f.write(f"Prediction stability: {result['stability']:.2%}\n\n")
        
        for path in result['paths']:
            f.write(f"  Rank {path['rank']}: Label {path['predicted_label']} (Score: {path['path_score']:.4f})\n")
            f.write(f"  Confidence: {path['confidence']:.2%}\n")
            f.write(f"  Stability: {path['prediction_stability']:.2%}\n")
            f.write("  Top features:\n")
            
            if path['important_features']:
                for feat_idx, importance, direction in path['important_features']:
                    f.write(f"    Feature {feat_idx}: {importance:.4f} {direction}\n")
            else:
                f.write("    No significant features found\n")
                
            f.write("\n")  # Add extra newline between ranks
        f.write("\n")  # Add extra newline between samples

# Save as JSON for further processing
with open(os.path.join(args.output, "perturbation_explanations.json"), 'w') as f:
    # Convert numpy values to Python native types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    json_explanations = []
    for exp in explanations:
        json_exp = {k: convert_numpy(v) if k != 'paths' else v for k, v in exp.items()}
        json_paths = []
        for path in exp['paths']:
            json_path = {k: convert_numpy(v) if k != 'important_features' else v for k, v in path.items()}
            json_path['important_features'] = [(int(idx), float(score), str(direction)) 
                                             for idx, score, direction in path['important_features']]
            json_paths.append(json_path)
        json_exp['paths'] = json_paths
        json_explanations.append(json_exp)
    
    json.dump(json_explanations, f, indent=2)

# Generate visualizations if requested
if args.visualize:
    print("Generating visualizations...")
    
    # Create visualizations directory
    viz_dir = os.path.join(args.output, "visualizations")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Generate one visualization per sample
    for i, exp in enumerate(explanations):
        # Create a feature importance visualization for the top path
        if len(exp['paths']) > 0:
            top_path = exp['paths'][0]
            
            # Get feature importances
            features = top_path['important_features']
            
            if features:
                # Unpack feature data
                feat_indices = []
                importances = []
                directions = []
                
                for feat_idx, importance, direction in features:
                    feat_indices.append(feat_idx)
                    importances.append(importance)
                    directions.append(1 if direction == "+" else -1)
                
                # Create horizontal bar chart with color representing direction
                plt.figure(figsize=(10, 6))
                y_pos = range(len(feat_indices))
                
                # Use colors to represent direction (red for negative, blue for positive)
                colors = ['red' if d < 0 else 'blue' for d in directions]
                plt.barh(y_pos, importances, align='center', color=colors)
                
                plt.yticks(y_pos, [f"Feature {idx}" for idx in feat_indices])
                plt.xlabel('Importance')
                plt.title(f"Sample {exp['sample_idx']} - Label {top_path['predicted_label']} Feature Importance")
                
                # Add a legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='blue', label='Increases score (+)'),
                    Patch(facecolor='red', label='Decreases score (-)')
                ]
                plt.legend(handles=legend_elements)
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"sample_{exp['sample_idx']}_features.png"))
                plt.close()
                
                # Create path confidence visualization
                plt.figure(figsize=(10, 6))
                labels = [f"Label {path['predicted_label']}" for path in exp['paths']]
                confidences = [path['confidence'] for path in exp['paths']]
                plt.bar(labels, confidences)
                plt.ylabel('Confidence')
                plt.title(f"Sample {exp['sample_idx']} - Path Confidences")
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"sample_{exp['sample_idx']}_confidences.png"))
                plt.close()
    
    # Generate overall analysis plots
    
    # 1. Accuracy plot
    correct_count = sum(1 for exp in explanations if exp['correct'])
    accuracy = correct_count / len(explanations)
    
    plt.figure(figsize=(8, 6))
    plt.bar(['Correct', 'Incorrect'], [accuracy, 1-accuracy])
    plt.ylabel('Proportion')
    plt.title(f"Prediction Accuracy on Explained Samples ({accuracy:.1%})")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "accuracy.png"))
    plt.close()
    
    # 2. Stability distribution plot
    stabilities = [exp['stability'] for exp in explanations]
    
    plt.figure(figsize=(10, 6))
    plt.hist(stabilities, bins=10, range=(0, 1), alpha=0.7)
    plt.axvline(np.mean(stabilities), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(stabilities):.2f}')
    plt.xlabel('Stability Score')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Stability Scores')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "stability_distribution.png"))
    plt.close()

print("Explanation analysis complete!") 