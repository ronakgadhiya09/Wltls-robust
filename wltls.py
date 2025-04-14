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

# Explainability and robustness constants
DEFENSE_NONE = "none"
DEFENSE_DETECT = "detect"
DEFENSE_DENOISE = "denoise"
DEFENSE_CERTIFY = "certify"

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

# Enhanced explainability arguments
parser.add_argument("--explain", dest='explain_predictions', action='store_true',
                    help="Generate path-based explanations for predictions")
parser.add_argument("--top_k", type=int, default=5,
                    help="Number of top paths to analyze for explanations")
parser.add_argument("--counterfactual", dest='generate_counterfactuals', action='store_true',
                    help="Generate counterfactual explanations for predictions")
parser.add_argument("--visualize", dest='create_visualizations', action='store_true',
                    help="Create visualizations for explanations (requires matplotlib)")
parser.add_argument("--report", dest='create_html_report', action='store_true',
                    help="Create an HTML report with all explanations and visualizations")

# Enhanced robustness arguments
parser.add_argument("--robustness", dest='check_robustness', action='store_true',
                    help="Perform robustness analysis on predictions")
parser.add_argument("--defense", choices=[DEFENSE_NONE, DEFENSE_DETECT, DEFENSE_DENOISE, DEFENSE_CERTIFY],
                    default=DEFENSE_NONE, help="Adversarial defense method to use")
parser.add_argument("--attack", dest='perform_attacks', action='store_true',
                    help="Perform adversarial attacks to test robustness")
parser.add_argument("--epsilon", type=float, default=0.1,
                    help="Epsilon value for robustness analysis perturbations")
parser.add_argument("--certify", dest='certify_robustness', action='store_true',
                    help="Perform certified robustness analysis (stronger but slower)")

parser.set_defaults(show_graph=False, explain_predictions=False, check_robustness=False,
                   generate_counterfactuals=False, create_visualizations=False, 
                   create_html_report=False, perform_attacks=False, certify_robustness=False)

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
if args.explain_predictions or args.check_robustness:
    print("\nEnhanced with explainability and robustness features.")
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

# Configure robustness settings based on arguments
if args.check_robustness:
    defense_mode = args.defense
    if defense_mode != DEFENSE_NONE:
        print(f"\nActivating robust defense mode: {defense_mode}")
        # Set the defense mode
        finalModel.adaptive_robustness['defense_mode'] = defense_mode
        
        # Additional settings based on defense mode
        if defense_mode == DEFENSE_CERTIFY or args.certify_robustness:
            finalModel.adaptive_robustness['smoothing_factor'] = args.epsilon
            print(f"Using smoothing factor: {args.epsilon}")

# Add explainability and robustness analysis
if args.explain_predictions or args.check_robustness:
    print("\nPerforming detailed analysis on test predictions...")
    
    # Sample a subset of test instances for detailed analysis
    num_samples = min(100, Xtest.shape[0])  # Using shape[0] instead of len() for sparse matrix
    sample_indices = np.random.choice(Xtest.shape[0], num_samples, replace=False)
    
    # Create results directory if needed
    analysis_dir = os.path.join(args.model_dir, "analysis")
    if args.create_visualizations or args.create_html_report:
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
    
    results = {
        'explanations': [],
        'robustness': [],
        'visualizations': []
    }
    
    # Import visualization tools if needed
    if args.create_visualizations:
        try:
            import matplotlib.pyplot as plt
            from enhanced_wltls_demo import (
                plot_feature_importance, 
                plot_counterfactual, 
                plot_robustness_analysis, 
                plot_path_visualization,
                plot_overall_robustness,
                create_html_report
            )
            visualization_available = True
        except ImportError:
            print("Warning: Matplotlib not available. Visualizations will be skipped.")
            visualization_available = False
    else:
        visualization_available = False
    
    # Process samples
    attack_results = []
    all_robustness_metrics = []
    
    for idx in sample_indices:
        # Convert sparse row to dense array for analysis if needed
        if hasattr(Xtest, 'toarray'):
            x = Xtest[idx].toarray().ravel()
        else:
            x = Xtest[idx]
            
        y_true = Ytest[idx]
        
        if args.explain_predictions:
            # Generate explanations
            explanation = finalModel.explain_prediction(x, k=args.top_k)
            results['explanations'].append({
                'sample_idx': idx,
                'true_label': y_true,
                'explanation': explanation
            })
            
            # Create visualizations if requested
            if args.create_visualizations and visualization_available:
                # Get top explanation
                top_explanation = explanation[0]
                
                # Create feature importance visualization
                feature_importance = np.zeros(DIMS)
                for feat_idx, imp, _ in top_explanation['important_features']:
                    feature_importance[feat_idx] = imp
                
                importance_viz = plot_feature_importance(
                    feature_importance,
                    title=f"Feature_Importance_Sample_{idx}",
                    output_dir=analysis_dir
                )
                
                # Create path visualization
                path_viz = plot_path_visualization(
                    idx,
                    top_explanation['path_nodes'],
                    finalModel,
                    analysis_dir
                )
                
                # Create counterfactual visualization if requested and available
                counterfactual_viz = None
                if args.generate_counterfactuals and top_explanation['counterfactual']['possible']:
                    counterfactual_viz = plot_counterfactual(
                        x,
                        top_explanation['counterfactual']['counterfactual_features'],
                        title=f"Counterfactual_Sample_{idx}",
                        output_dir=analysis_dir
                    )
                
                # Store visualization paths
                results['visualizations'].append({
                    'sample_idx': idx,
                    'importance': importance_viz,
                    'path': path_viz,
                    'counterfactual': counterfactual_viz
                })
        
        if args.check_robustness:
            # Perform robustness analysis
            robustness = finalModel.get_robustness_score(x, epsilon=args.epsilon)
            
            # Store robustness metrics
            results['robustness'].append({
                'sample_idx': idx,
                'true_label': y_true,
                'robustness_score': robustness['stability_score'],
                'certified_radius': robustness['certified_radius'],
                'adversarial_detected': robustness['adversarial_detected'],
                'predicted_label': robustness['original_prediction']
            })
            
            all_robustness_metrics.append(robustness)
            
            # Perform attack if requested
            if args.perform_attacks:
                # Get current prediction
                responses = x.dot(finalModel.W).ravel()
                code = finalModel.decoder.findKBestCodes(responses, 1)[0]
                label = finalModel.codeManager.codeToLabel(code)
                
                # Run adversarial attack
                attack_result = finalModel._adversarial_search(x, code, label, epsilon=args.epsilon)
                attack_results.append(attack_result)
                
                # Create visualization
                if args.create_visualizations and visualization_available:
                    robustness_viz = plot_robustness_analysis(
                        idx,
                        robustness,
                        analysis_dir
                    )
                    
                    # Add to visualizations
                    if idx in [v['sample_idx'] for v in results['visualizations']]:
                        for v in results['visualizations']:
                            if v['sample_idx'] == idx:
                                v['robustness'] = robustness_viz
                    else:
                        results['visualizations'].append({
                            'sample_idx': idx,
                            'robustness': robustness_viz
                        })
    
    # Create overall robustness visualization if requested
    if args.check_robustness and args.create_visualizations and visualization_available and attack_results:
        stability_scores = [r['stability_score'] for r in results['robustness']]
        overall_viz = plot_overall_robustness(stability_scores, attack_results, analysis_dir)
        results['overall_robustness_viz'] = overall_viz
    
    # Create HTML report if requested
    if args.create_html_report and visualization_available:
        # Prepare data for report
        report_data = {
            'dataset_name': DATASET.name,
            'num_labels': LABELS,
            'num_features': DIMS,
            'slice_width': args.slice_width,
            'accuracy': finalModel.test(Xtest[:100], Ytest[:100])['accuracy'],
            'samples': [],
            'avg_stability': np.mean([r['robustness_score'] for r in results['robustness']]) if results['robustness'] else 0,
            'avg_certified_radius': np.mean([r['certified_radius'] for r in results['robustness']]) if results['robustness'] else 0,
            'adversarial_detected': sum(1 for r in results['robustness'] if r['adversarial_detected']) if results['robustness'] else 0,
            'total_samples': len(results['robustness']) if results['robustness'] else 0,
            'attack_attempts': len(attack_results),
            'successful_attacks': sum(1 for r in attack_results if r['attack_succeeded']) if attack_results else 0
        }
        
        # Add sample data
        for viz in results['visualizations']:
            sample_idx = viz['sample_idx']
            
            # Find explanation data
            explanation_data = None
            for exp in results['explanations']:
                if exp['sample_idx'] == sample_idx:
                    explanation_data = exp
                    break
            
            # Find robustness data
            robustness_data = None
            for rob in results['robustness']:
                if rob['sample_idx'] == sample_idx:
                    robustness_data = rob
                    break
            
            # Skip if no data
            if not explanation_data and not robustness_data:
                continue
            
            # Create sample data
            sample_data = {
                'sample_idx': sample_idx,
                'true_label': explanation_data['true_label'] if explanation_data else robustness_data['true_label']
            }
            
            # Add explanation data
            if explanation_data and explanation_data['explanation']:
                top_explanation = explanation_data['explanation'][0]
                sample_data['predicted_label'] = top_explanation['predicted_label']
                sample_data['confidence'] = top_explanation['confidence']
                sample_data['stability'] = top_explanation['prediction_stability']
                
                # Add visualization paths if available
                if 'importance' in viz:
                    sample_data['feature_importance_viz'] = os.path.basename(viz['importance'])
                
                if 'path' in viz:
                    sample_data['path_viz'] = os.path.basename(viz['path'])
                
                if 'counterfactual' in viz and viz['counterfactual']:
                    sample_data['counterfactual_viz'] = os.path.basename(viz['counterfactual'])
                    sample_data['counterfactual_label'] = top_explanation['counterfactual']['alternative_label']
            
            # Add robustness data
            if robustness_data:
                if 'robustness' in viz:
                    sample_data['robustness_viz'] = os.path.basename(viz['robustness'])
                
                if 'predicted_label' not in sample_data:
                    sample_data['predicted_label'] = robustness_data['predicted_label']
                
                if 'stability' not in sample_data:
                    sample_data['stability'] = robustness_data['robustness_score']
            
            report_data['samples'].append(sample_data)
        
        # Add overall robustness visualization
        if 'overall_robustness_viz' in results:
            report_data['overall_robustness_viz'] = os.path.basename(results['overall_robustness_viz'])
        
        # Create report
        html_path = create_html_report(report_data, analysis_dir)
        print(f"\nHTML report generated: {html_path}")
    
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
                        
                    # Add counterfactual explanation if available
                    if pred['counterfactual']['possible']:
                        f.write("  Counterfactual explanation:\n")
                        f.write(f"    To change prediction to label {pred['counterfactual']['alternative_label']}, change:\n")
                        for feature in pred['counterfactual']['counterfactual_features']:
                            f.write(f"      Feature {feature['feature_idx']} ({feature['direction']})\n")
                            
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
                f.write(f"  Stability score: {result['robustness_score']:.4f}\n")
                f.write(f"  Certified radius: {result['certified_radius']:.4f}\n")
                f.write(f"  Adversarial detected: {result['adversarial_detected']}\n\n")
                total_stability += result['robustness_score']
            
            avg_stability = total_stability / len(results['robustness']) if results['robustness'] else 0
            f.write(f"\nOverall average stability score: {avg_stability:.4f}")
            
            if attack_results:
                success_count = sum(1 for r in attack_results if r['attack_succeeded'])
                f.write(f"\nAdversarial attacks: {success_count} succeeded out of {len(attack_results)}")
                f.write(f"\nAttack success rate: {success_count*100/len(attack_results):.2f}%")
        
        print(f"Robustness analysis saved to: {rob_file}")

printSeparator()

# Check if we want to experiment sparse models
if args.try_sparse:
    print("Experimenting sparse models:")

    from WLTLS.sparseExperiment import SparseExperiment

    ex = SparseExperiment(codeManager, heaviestPaths)
    ex.run(finalModel.W, Xtest, Ytest, Xvalid, Yvalid)

    printSeparator()
