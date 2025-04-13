from WLTLS.datasets import read
from WLTLS.decoding import HeaviestPaths, expLoss, squaredLoss, squaredHingeLoss
from WLTLS.mainModels.finalModel import FinalModel
from graphs import TrellisGraph
import argparse
import warnings
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from WLTLS.mainModels import WltlsModel
from WLTLS.learners import AveragedPerceptron, AROW
from aux_lib import Timing
from WLTLS.codeManager import GreedyCodeManager, RandomCodeManager
from WLTLS.datasets import datasets
from WLTLS.experiment import Experiment
from scipy.sparse import csr_matrix
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Const choices
LEARNER_AROW = "AROW"
LEARNER_PERCEPTRON = "perceptron"
LOSS_EXP = "exponential"
LOSS_SQUARED = "squared"
LOSS_SQUARED_HINGE = "squared_hinge"
ASSIGNMENT_GREEDY = "greedy"
ASSIGNMENT_RANDOM = "random"

LINE_WIDTH = 80
def printSeparator():
    print("=" * LINE_WIDTH)

all_datasets = [d.name for d in datasets.getAll()]

# Set argument parser
parser = argparse.ArgumentParser(description="Runs a single W-LTLS experiment. " +
                                             "See https://github.com/ievron/wltls/ for documentation and license details.")
parser.add_argument("dataset",
                    choices=all_datasets,
                    help="Dataset name")
parser.add_argument("data_path", help="Path of the directory holding the datasets downloaded from PD-Sparse")
parser.add_argument("model_dir", help="Path of a directory to save the model in (model.npz)")
parser.add_argument("-slice_width", type=int, help="The slice width of the trellis graph", default=2)
parser.add_argument("-decoding_loss",
                    choices=[LOSS_EXP, LOSS_SQUARED_HINGE, LOSS_SQUARED],
                    nargs="?",
                    const=LOSS_EXP,
                    default=LOSS_EXP,
                    help="The loss for the loss-based decoding scheme")
parser.add_argument("-epochs", type=int, help="Number of epochs", default=-1)
parser.add_argument("-rnd_seed", type=int, help="Random seed")
parser.add_argument("-path_assignment", choices=[ASSIGNMENT_RANDOM, ASSIGNMENT_GREEDY],
                    nargs="?", const=ASSIGNMENT_RANDOM, help="Path assignment policy", default=ASSIGNMENT_RANDOM)
parser.add_argument("-binary_classifier", choices=[LEARNER_AROW, LEARNER_PERCEPTRON],
                    nargs="?", const=LEARNER_AROW,
                    help="The binary classifier for learning the binary subproblems",
                    default=LEARNER_AROW)
parser.add_argument("--plot_graph", dest='show_graph', action='store_true', help="Plot the trellis graph on start")
parser.add_argument("--sparse", dest='try_sparse', action='store_true',
                    help="Experiment sparse models at the end of training")
parser.add_argument("--explain", dest='explain_predictions', action='store_true',
                    help="Generate path-based explanations for predictions")
parser.add_argument("--robustness", dest='check_robustness', action='store_true',
                    help="Perform robustness analysis on predictions")
parser.add_argument("--epsilon", type=float, default=0.1,
                    help="Epsilon value for robustness analysis perturbations")
parser.add_argument("--top_k", type=int, default=5,
                    help="Number of top paths to analyze for explanations")
parser.set_defaults(show_graph=False, explain_predictions=False, check_robustness=False)

args = parser.parse_args()

# If user gave a random seed
if args.rnd_seed is not None:
    import random
    random.seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)

from WLTLS.datasets import datasets
DATASET =   datasets.getParams(args.dataset)
EPOCHS =    args.epochs if args.epochs >= 1 else DATASET.epochs
LOG_PATH = os.path.join(args.model_dir, "model")

warnings.filterwarnings("ignore",".*GUI is implemented.*")

printSeparator()
print("Learning a Wide-LTLS model, proposed in:")
print("Efficient Loss-Based Decoding On Graphs For Extreme Classification. NIPS 2018.")
printSeparator()

# Load dataset
Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest, LABELS, DIMS = read(args.data_path, DATASET)

printSeparator()

assert args.slice_width >= 2 and args.slice_width <= LABELS,\
    "Slice width must be larger than 1 and smaller than the number of classes."


# Decide the loss
if args.decoding_loss == LOSS_EXP:
    loss = expLoss
elif args.decoding_loss == LOSS_SQUARED_HINGE:
    loss = squaredHingeLoss
else:
    loss = squaredLoss

# Create the graph
trellisGraph = TrellisGraph(LABELS, args.slice_width)
heaviestPaths = HeaviestPaths(trellisGraph, loss=loss)

# Plot the graph if needed
if args.show_graph:
    trellisGraph.show(block=True)

# Process arguments
if args.path_assignment == ASSIGNMENT_RANDOM:
    codeManager = RandomCodeManager(LABELS, heaviestPaths.allCodes())
else:
    codeManager = GreedyCodeManager(LABELS, heaviestPaths.allCodes())

if args.binary_classifier == LEARNER_AROW:
    learner = AROW
else:
    learner = AveragedPerceptron

print("Using {} as the binary classifier.".format(args.binary_classifier))
print("Decoding according to the {} loss.".format(args.decoding_loss))

# Create the model
mainModel = WltlsModel(LABELS, DIMS, learner, codeManager, heaviestPaths)

printSeparator()

# Run the experiment
Experiment(mainModel, EPOCHS).run(Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest,
                                  modelLogPath=LOG_PATH,
                                  returnBestValidatedModel=True)


printSeparator()

# Create a final model (for fast inference) and test it
finalModel = FinalModel(DIMS, mainModel, codeManager, heaviestPaths)
del mainModel

printSeparator()

# Add explainability and robustness analysis
if args.explain_predictions or args.check_robustness:
    print("\nPerforming detailed analysis on test predictions...")
    
    # Sample a subset of test instances for detailed analysis
    num_samples = min(100, Xtest.shape[0])  # Using shape[0] instead of len() for sparse matrix
    sample_indices = np.random.choice(Xtest.shape[0], num_samples, replace=False)
    
    results = {
        'explanations': [],
        'robustness': []
    }
    
    for idx in sample_indices:
        # Convert sparse row to dense array for analysis if needed
        if hasattr(Xtest, 'toarray'):
            x = Xtest[idx].toarray().ravel()
        else:
            x = Xtest[idx]
            
        y_true = Ytest[idx]
        
        if args.explain_predictions:
            explanation = finalModel.explain_prediction(x, k=args.top_k)
            results['explanations'].append({
                'sample_idx': idx,
                'true_label': y_true,
                'explanation': explanation
            })
        
        if args.check_robustness:
            robustness = finalModel.get_robustness_score(x, epsilon=args.epsilon)
            results['robustness'].append({
                'sample_idx': idx,
                'true_label': y_true,
                'robustness_score': robustness['stability_score'],
                'predicted_label': robustness['original_prediction']
            })
    
    # Save analysis results
    if args.explain_predictions:
        exp_file = os.path.join(args.model_dir, "explanations.txt")
        with open(exp_file, 'w') as f:
            f.write("Path-based Explanations Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results['explanations']:
                f.write(f"Sample {result['sample_idx']} (True label: {result['true_label']})\n")
                for pred in result['explanation']:
                    f.write(f"  Rank {pred['rank']}: Label {pred['predicted_label']} (Score: {pred['path_score']:.4f})\n")
                    f.write(f"  Confidence: {pred['confidence']:.2%}\n")
                    f.write(f"  Stability: {pred['prediction_stability']:.2%}\n")
                    f.write("  Top features:\n")
                    if pred['important_features']:
                        for feat_idx, importance, direction in pred['important_features']:
                            f.write(f"    Feature {feat_idx}: {importance:.4f} {direction}\n")
                    else:
                        f.write("    No significant features identified for this path\n")
                    f.write("\n")  # Add extra newline between ranks
                f.write("\n")  # Add extra newline between samples
    
        print(f"\nExplanations saved to: {exp_file}")
    
    if args.check_robustness:
        rob_file = os.path.join(args.model_dir, "robustness.txt")
        with open(rob_file, 'w') as f:
            f.write("Robustness Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            
            total_stability = 0
            for result in results['robustness']:
                f.write(f"Sample {result['sample_idx']}:\n")
                f.write(f"  True label: {result['true_label']}\n")
                f.write(f"  Predicted: {result['predicted_label']}\n")
                f.write(f"  Stability score: {result['robustness_score']:.4f}\n\n")
                total_stability += result['robustness_score']
            
            avg_stability = total_stability / len(results['robustness'])
            f.write(f"\nOverall average stability score: {avg_stability:.4f}")
        
        print(f"Robustness analysis saved to: {rob_file}")

printSeparator()

# Check if we want to experiment sparse models
if args.try_sparse:
    print("Experimenting sparse models:")

    from WLTLS.sparseExperiment import SparseExperiment

    ex = SparseExperiment(codeManager, heaviestPaths)
    ex.run(finalModel.W, Xtest, Ytest, Xvalid, Yvalid)

    printSeparator()
