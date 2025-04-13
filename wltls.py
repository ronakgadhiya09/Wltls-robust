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
parser.set_defaults(show_graph=False)

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

# Use fixed path for scores file
scores_file = "/content/sector_scores.txt"
result = finalModel.test(Xtest, Ytest, save_scores=True, scores_file=scores_file)

print("The final model was tested in {} and achieved {:.1f}% accuracy.".format(
    Timing.secondsToString(result["time"]), result["accuracy"]
))



# scores_file = "/content/sector_scores.txt"
# result = finalModel.test(Xtest, Ytest, scores_file=scores_file)
# print("The final model was tested in {} and achieved {:.1f}% accuracy.".format(
#     Timing.secondsToString(result["time"]), result["accuracy"]
# ))


print("Generating LIME perturbations")
Xtest_dense = Xtest.toarray()
sample_id = 0
x0 = Xtest_dense[sample_id]
y0 = Ytest[sample_id]  # confidence vector for the sample

predicted_class = np.argmax(y0)

# Step 2: Perturbations
num_perturbations = 500
noise_std = 0.03
perturbations = np.random.normal(loc=0.0, scale=noise_std, size=(num_perturbations, Xtest_dense.shape[1]))
X_perturbed = x0 + perturbations
X_perturbed_sparse = csr_matrix(X_perturbed)

# generate output
y_temp = Ytest
y_perturbed_sparse = csr_matrix(y_temp[:num_perturbations])

print("Generated perturbations")
print("Now second predict by model")
scores_file = "/content/lime_output.txt"
result = finalModel.test(X_perturbed_sparse, y_perturbed_sparse, scores_file=scores_file)


#One Hot Encoded results for perturbations

lime_out = pd.read_csv("/content/lime_output.txt", sep=r'\s+', engine='python')
lime_out = lime_out.drop(lime_out.columns[0], axis=1)  # Drop ID/index column

lime_np = lime_out.to_numpy()

class_indices = np.argmax(lime_np, axis=1)       # Shape: (n_samples,)
scores = np.max(lime_np, axis=1)                 # Shape: (n_samples,)

encoder = OneHotEncoder(categories=[np.arange(105)], handle_unknown='ignore', sparse_output=False)
encoder.fit(np.arange(105).reshape(-1, 1))

y_from_perturbed = encoder.transform(class_indices.reshape(-1, 1))

print('Computing rbf similarity')
def compute_similarity(original_instance, perturbed_samples, kernel_width=0.85):
    # Computes similarity scores using the RBF kernel (LIME-style)
    distances = np.linalg.norm(perturbed_samples - original_instance, axis=1)
    weights = np.exp(- (distances ** 2) / (2 * kernel_width ** 2))
    return weights

# Compute similarity weights
weights = compute_similarity(x0, X_perturbed)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_perturbed)
print('Fit LIME regression models')
# Step 2: Fit multi-output Ridge regression
from sklearn.linear_model import Ridge
import numpy as np

def fit_local_model(X_perturbed, Y, weights):
    n_classes = Y.shape[1]
    coef_matrix = []

    for class_idx in range(n_classes):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_perturbed, Y[:, class_idx], sample_weight=weights)
        coef_matrix.append(ridge.coef_)

    return np.array(coef_matrix)

output_path = '/content/explainations.csv'
feature_importance = fit_local_model(X_perturbed, y_from_perturbed, weights)
n_classes = feature_importance.shape[0]
feature_names=[f"Feature_{i}" for i in range(X_perturbed.shape[1])]
lime_df = pd.DataFrame(feature_importance, columns=feature_names)

lime_df.to_csv(output_path, index=False)

printSeparator()

# Check if we want to experiment sparse models
if args.try_sparse:
    print("Experimenting sparse models:")

    from WLTLS.sparseExperiment import SparseExperiment

    ex = SparseExperiment(codeManager, heaviestPaths)
    ex.run(finalModel.W, Xtest, Ytest, Xvalid, Yvalid)

    printSeparator()
