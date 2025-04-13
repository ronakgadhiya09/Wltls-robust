"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
"""

import numpy as np
from aux_lib import Timing

#############################################################################################
# A model for fast inference.
# Merges separate previously-learned weight vectors into on weights matrix.
# Note that we learn the vectors separately to support parallel learning and/or
# the greedy path-to-label allocation policy.
#############################################################################################
class FinalModel:
    def __init__(self, DIMS, trainedModel, codeManager, decoder):
        print("Preparing a final model (this may take some time)...")

        # Merge trained model into one matrix for faster inference
        # (maybe could be done more efficiently in terms of memory)
        self.W = np.ravel(np.array([l.mean for l in trainedModel._learners]).T).reshape((DIMS, -1))

        # Delete the separate vectors from the memory
        for learner in trainedModel._learners:
            del learner

        self.codeManager = codeManager
        self.decoder = decoder

        print("The final model created successfully.")

    def _get_class_scores(self, responses):
        # Get all possible codes and their scores
        codes = self.decoder.allCodes()
        scores = {}
        
        # For each class, find its code and corresponding score
        for label in range(self.codeManager.LABELS):
            code = self.codeManager.labelToCode(label)
            if code is not None:
                # Compute score for this class based on its code
                score = sum([r if c == 1 else -r for r, c in zip(responses, code)])
                scores[label] = score
        
        return scores

    # Test the final model
    def test(self, Xtest, Ytest, save_scores=True, scores_file=None):
        t = Timing()
        Ypredicted = [0] * Xtest.shape[0]
        all_scores = []

        print("Computing confidence scores for {} test samples...".format(Xtest.shape[0]))
        
        for i, x in enumerate(Xtest):
            # Get responses from predictors
            responses = x.dot(self.W).ravel()
            
            # Get scores and feature importance for all classes
            scores = {}
            features = {}
            
            # Get k-best paths with importance scores
            codes, path_attributions = self.decoder.findKBestCodesWithImportance(responses, x.dot(self.W), k=5)
            
            for code, attribution in zip(codes, path_attributions):
                label = self.codeManager.codeToLabel(code)
                if label is not None:
                    score = sum([r if c == 1 else -r for r, c in zip(responses, code)])
                    scores[label] = score
                    
                    # Get top features for this path
                    top_features = attribution.get_top_features(n=5)
                    features[label] = top_features
            
            all_scores.append(scores)

            # Print scores and features for first few examples
            if i < 10:  # Show first 10 examples
                print(f"\nSample {i} (True label: {Ytest[i]})")
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for rank, (label, score) in enumerate(sorted_scores[:5], 1):
                    print(f"Rank {rank}: Label {label} (Score: {score:.4f})")
                    print("Top features:")
                    if label in features:
                        for feat_idx, importance in features[label]:
                            print(f"  Feature {feat_idx}: {importance:.4f}")

            # Find best code using graph inference
            code = self.decoder.findKBestCodes(responses, 1)[0]
            Ypredicted[i] = self.codeManager.codeToLabel(code)

        if save_scores:
            if scores_file is None:
                scores_file = "confidence_scores.txt"
            
            try:
                with open(scores_file, 'w') as f:
                    # Write header with class labels
                    f.write("Sample_ID\t" + "\t".join(f"Class_{i}" for i in range(self.codeManager.LABELS)) + "\n")
                    for i, scores in enumerate(all_scores):
                        score_values = [str(scores.get(j, float('-inf'))) for j in range(self.codeManager.LABELS)]
                        f.write(f"{i}\t" + "\t".join(score_values) + "\n")
                print("\nConfidence scores successfully saved to: {}".format(scores_file))
            except Exception as e:
                print("\nWarning: Failed to save confidence scores to file: {}".format(str(e)))

        correct = sum([y1 == y2 for y1, y2 in zip(Ypredicted, Ytest)])
        accuracy = correct * 100 / Xtest.shape[0]

        elapsed = t.get_elapsed_secs()
        print("\nTesting completed in {:.2f} seconds".format(elapsed))
        
        return {
            "accuracy": accuracy,
            "time": elapsed,
            "y_predicted": Ypredicted,
            "all_scores": all_scores
        }

    def explain_prediction(self, x, k=5):
        """Explain model predictions using perturbation-based feature importance analysis
        
        This method uses a model-agnostic approach to determine feature importance by
        measuring how perturbing each feature affects the model's predictions.
        """
        # Get the original prediction and responses
        responses = x.dot(self.W).ravel()
        original_code = self.decoder.findKBestCodes(responses, 1)[0]
        original_label = self.codeManager.codeToLabel(original_code)
        original_score = sum([r if c == 1 else -r for r, c in zip(responses, original_code)])
        
        # Get the k-best codes and their scores
        codes = self.decoder.findKBestCodes(responses, k)
        code_scores = []
        for code in codes:
            score = sum([r if c == 1 else -r for r, c in zip(responses, code)])
            code_scores.append(score)
        
        # Calculate total score for confidence calculation
        total_score = sum(abs(s) for s in code_scores)
        
        # Calculate feature importance using perturbation analysis
        feature_importance = {}
        feature_effects = {}
        
        # Use only features with non-zero values to speed up analysis
        active_features = np.where(np.abs(x) > 1e-6)[0]
        if len(active_features) == 0:
            active_features = np.arange(min(20, len(x)))
        
        # For each feature, measure impact when zeroed out
        for feat_idx in active_features:
            # Create a perturbed version of the input with this feature zeroed
            x_perturbed = x.copy()
            original_value = x_perturbed[feat_idx]
            x_perturbed[feat_idx] = 0
            
            # Get new prediction
            perturbed_responses = x_perturbed.dot(self.W).ravel()
            perturbed_code = self.decoder.findKBestCodes(perturbed_responses, 1)[0]
            perturbed_score = sum([r if c == 1 else -r for r, c in zip(perturbed_responses, original_code)])
            
            # Calculate impact on score (as % decrease)
            score_impact = (original_score - perturbed_score) / (abs(original_score) + 1e-10)
            
            # Track prediction changes
            prediction_changed = not np.array_equal(original_code, perturbed_code)
            
            # Store feature importance
            feature_importance[feat_idx] = abs(score_impact)
            
            # Store feature effect (direction of influence)
            feature_effects[feat_idx] = {
                'score_impact': score_impact,
                'prediction_changed': prediction_changed,
                'original_value': original_value
            }
        
        # Normalize feature importance to sum to 1.0
        importance_sum = sum(feature_importance.values()) or 1.0
        normalized_importance = {k: v/importance_sum for k, v in feature_importance.items()}
        
        # Sort features by importance
        sorted_features = sorted(normalized_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Generate detailed explanations for each predicted class
        explanations = []
        for i, (code, score) in enumerate(zip(codes, code_scores)):
            label = self.codeManager.codeToLabel(code)
            confidence = abs(score) / total_score if total_score > 0 else 0.0
            
            # Get top features with importance and effect direction
            top_features = []
            for feat_idx, importance in sorted_features[:10]:  # Take top 10 features
                # Direction of effect (positive means increasing the prediction score)
                effect = feature_effects[feat_idx]['score_impact']
                effect_direction = "+" if effect > 0 else "-"
                
                # Only include features with non-negligible importance
                if abs(importance) > 0.01:
                    top_features.append((
                        int(feat_idx),
                        float(importance),
                        effect_direction
                    ))
            
            # Generate explanation details
            explanations.append({
                'rank': i + 1,
                'predicted_label': label,
                'path_score': score,
                'confidence': confidence,
                'important_features': top_features,
                'prediction_stability': self._calculate_stability(x, code)
            })
        
        return explanations
    
    def _calculate_stability(self, x, code, n_samples=10, noise_level=0.05):
        """Calculate prediction stability under small random perturbations"""
        stability_count = 0
        
        for _ in range(n_samples):
            # Add small Gaussian noise
            noise = np.random.normal(0, noise_level, size=x.shape)
            x_noisy = x + noise * np.abs(x)  # Scale noise by feature magnitude
            
            # Get prediction with noise
            noisy_responses = x_noisy.dot(self.W).ravel()
            noisy_code = self.decoder.findKBestCodes(noisy_responses, 1)[0]
            
            # Count stability
            if np.array_equal(code, noisy_code):
                stability_count += 1
        
        return stability_count / n_samples

    def get_robustness_score(self, x, epsilon=0.1):
        """Calculate prediction robustness by analyzing path stability under perturbation"""
        responses = x.dot(self.W).ravel()
        original_code = self.decoder.findKBestCodes(responses, 1)[0]
        
        # Generate perturbations
        perturbations = np.random.normal(0, epsilon, size=(10, len(x)))
        perturbed_predictions = []
        
        for perturb in perturbations:
            perturbed_x = x + perturb
            perturbed_responses = perturbed_x.dot(self.W).ravel()
            perturbed_code = self.decoder.findKBestCodes(perturbed_responses, 1)[0]
            perturbed_predictions.append(perturbed_code)
        
        # Calculate stability score (% of perturbations that yield same prediction)
        stability = sum(np.array_equal(original_code, p) for p in perturbed_predictions) / len(perturbations)
        
        return {
            'stability_score': stability,
            'original_prediction': self.codeManager.codeToLabel(original_code)
        }
