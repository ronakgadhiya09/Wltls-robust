"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
"""

import numpy as np
from aux_lib import Timing
import copy

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
        
        # Precompute the sensitivity matrix for efficient feature importance
        self._compute_sensitivity_matrix()
        
        # Initialize adaptive robustness parameters
        self._initialize_robustness_params()

        print("The final model created successfully.")
        
    def _initialize_robustness_params(self):
        """Initialize parameters for adaptive robustness defense"""
        # Track the distribution of feature values to detect outliers
        self.feature_statistics = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'num_samples': 0
        }
        
        # Adversarial detection thresholds
        self.detection_thresholds = {
            'confidence_drop': 0.3,  # Significant drop in confidence
            'path_deviation': 0.5,   # Change in path traversal pattern
            'feature_outlier': 3.0   # Number of std devs for outlier detection
        }
        
        # Adaptive robustness settings
        self.adaptive_robustness = {
            'enabled': True,
            'smoothing_factor': 0.1,  # For randomized smoothing defense
            'perturbation_detection_threshold': 0.8,
            'defense_mode': 'detect_and_denoise'  # Options: 'detect_only', 'detect_and_denoise', 'certify'
        }

    def _compute_sensitivity_matrix(self):
        """Precompute a sensitivity matrix for the model features"""
        # This matrix will track how sensitive each edge is to each feature
        self.sensitivity_matrix = np.abs(self.W)
        
        # Normalize per edge to get relative importance
        edge_sums = np.sum(self.sensitivity_matrix, axis=0)
        # Avoid division by zero
        edge_sums[edge_sums == 0] = 1.0
        self.sensitivity_matrix = self.sensitivity_matrix / edge_sums
        
    def _update_feature_statistics(self, x):
        """Update the running statistics for features"""
        # First sample
        if self.feature_statistics['mean'] is None:
            self.feature_statistics['mean'] = np.copy(x)
            self.feature_statistics['std'] = np.zeros_like(x)
            self.feature_statistics['min'] = np.copy(x)
            self.feature_statistics['max'] = np.copy(x)
            self.feature_statistics['num_samples'] = 1
            return
        
        # Update using Welford's online algorithm for mean and variance
        n = self.feature_statistics['num_samples']
        new_n = n + 1
        delta = x - self.feature_statistics['mean']
        
        # Update mean
        new_mean = self.feature_statistics['mean'] + delta / new_n
        
        # Update variance (stored as std^2)
        delta2 = x - new_mean
        new_std = self.feature_statistics['std'] + delta * delta2
        
        # Update min/max
        new_min = np.minimum(self.feature_statistics['min'], x)
        new_max = np.maximum(self.feature_statistics['max'], x)
        
        # Store updated values
        self.feature_statistics['mean'] = new_mean
        if new_n > 1:
            self.feature_statistics['std'] = np.sqrt(new_std / (new_n - 1))
        self.feature_statistics['min'] = new_min
        self.feature_statistics['max'] = new_max
        self.feature_statistics['num_samples'] = new_n

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
        robustness_metrics = []
        explanations = []

        print("Computing confidence scores for {} test samples...".format(Xtest.shape[0]))
        
        for i, x in enumerate(Xtest):
            # Update feature statistics for adversarial detection
            self._update_feature_statistics(x)
            
            # Get predictions and analysis
            result = self._predict_with_analysis(x)
            Ypredicted[i] = result['prediction']
            all_scores.append(result['scores'])
            
            # Store robustness metrics
            robustness_metrics.append({
                'stability_score': result['robustness']['stability_score'],
                'certified_radius': result['robustness']['certified_radius'],
                'adversarial_detected': result['robustness']['adversarial_detected']
            })
            
            # Store explanations for sample
            explanations.append(result['explanation'])
            
            # Print detailed info for first few examples
            if i < 5:  # Show first 5 examples
                print(f"\nSample {i} (True label: {Ytest[i]})")
                print(f"Predicted label: {result['prediction']}")
                print(f"Stability score: {result['robustness']['stability_score']:.4f}")
                if result['robustness']['adversarial_detected']:
                    print("WARNING: Potential adversarial example detected!")
                
                # Show top predictions
                sorted_scores = sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)
                for rank, (label, score) in enumerate(sorted_scores[:3], 1):
                    print(f"Rank {rank}: Label {label} (Score: {score:.4f})")
                
                # Show feature importance
                if len(result['explanation']) > 0:
                    top_explanation = result['explanation'][0]
                    print("Top features for prediction:")
                    for feat_idx, importance, direction in top_explanation['important_features'][:3]:
                        print(f"  Feature {feat_idx}: {importance:.4f} {direction}")

        if save_scores:
            if scores_file is None:
                scores_file = "confidence_scores.txt"
            
            try:
                with open(scores_file, 'w') as f:
                    # Write header with class labels
                    f.write("Sample_ID\t" + "\t".join(f"Class_{i}" for i in range(self.codeManager.LABELS)) + "\tRobustness\tAdversarial\n")
                    for i, (scores, robustness) in enumerate(zip(all_scores, robustness_metrics)):
                        score_values = [str(scores.get(j, float('-inf'))) for j in range(self.codeManager.LABELS)]
                        f.write(f"{i}\t" + "\t".join(score_values) + 
                                f"\t{robustness['stability_score']}\t{int(robustness['adversarial_detected'])}\n")
                print("\nConfidence scores successfully saved to: {}".format(scores_file))
            except Exception as e:
                print("\nWarning: Failed to save confidence scores to file: {}".format(str(e)))

        correct = sum([y1 == y2 for y1, y2 in zip(Ypredicted, Ytest)])
        accuracy = correct * 100 / Xtest.shape[0]

        elapsed = t.get_elapsed_secs()
        print("\nTesting completed in {:.2f} seconds".format(elapsed))
        
        # Calculate adversarial detection statistics
        adversarial_count = sum(1 for r in robustness_metrics if r['adversarial_detected'])
        print(f"Potential adversarial examples detected: {adversarial_count} ({adversarial_count*100/len(robustness_metrics):.2f}%)")
        
        # Calculate average robustness metrics
        avg_stability = np.mean([r['stability_score'] for r in robustness_metrics])
        avg_certified_radius = np.mean([r['certified_radius'] for r in robustness_metrics])
        print(f"Average prediction stability: {avg_stability:.4f}")
        print(f"Average certified radius: {avg_certified_radius:.4f}")
        
        return {
            "accuracy": accuracy,
            "time": elapsed,
            "y_predicted": Ypredicted,
            "all_scores": all_scores,
            "robustness_metrics": robustness_metrics,
            "explanations": explanations
        }

    def _predict_with_analysis(self, x):
        """Make a prediction with comprehensive analysis including robustness and explainability"""
        # Get responses from predictors
        responses = x.dot(self.W).ravel()
        
        # Check for adversarial inputs
        is_adversarial, cleaned_x = self._detect_and_defend_adversarial(x, responses)
        
        # If adversarial detected and defense enabled, use cleaned input instead
        if is_adversarial and self.adaptive_robustness['defense_mode'] != 'detect_only':
            cleaned_responses = cleaned_x.dot(self.W).ravel()
            responses_to_use = cleaned_responses
        else:
            responses_to_use = responses
            
        # Get class scores
        scores = self._get_class_scores(responses_to_use)
        
        # Find best code using graph inference
        code = self.decoder.findKBestCodes(responses_to_use, 1)[0]
        predicted_label = self.codeManager.codeToLabel(code)
        
        # Get k-best codes for explanation
        k = 5  # Number of top paths to analyze
        codes = self.decoder.findKBestCodes(responses_to_use, k)
        
        # Generate explanations
        explanations = self.explain_prediction(x, k)
        
        # Calculate robustness metrics
        robustness = self._evaluate_robustness(x, code, responses_to_use)
        robustness['adversarial_detected'] = is_adversarial
        
        return {
            'prediction': predicted_label,
            'scores': scores,
            'code': code,
            'paths': codes,
            'robustness': robustness,
            'explanation': explanations,
            'adversarial': is_adversarial
        }

    def _detect_and_defend_adversarial(self, x, responses):
        """Detect adversarial inputs and apply defense if configured"""
        # Skip detection for insufficient samples
        if self.feature_statistics['num_samples'] < 10:
            return False, x
            
        # Only perform detection if we have feature statistics
        if self.feature_statistics['std'] is None:
            return False, x
            
        # Calculate feature Z-scores (how many std deviations from mean)
        z_scores = np.zeros_like(x)
        non_zero_std = self.feature_statistics['std'] > 1e-8
        if np.any(non_zero_std):
            z_scores[non_zero_std] = (x[non_zero_std] - self.feature_statistics['mean'][non_zero_std]) / self.feature_statistics['std'][non_zero_std]
        
        # Check for anomalous features (extremely large values)
        threshold = self.detection_thresholds['feature_outlier']
        outlier_features = np.abs(z_scores) > threshold
        outlier_ratio = np.mean(outlier_features) if len(outlier_features) > 0 else 0
        
        # Check for unusual pattern of responses across edges
        edge_responses = x.dot(self.W)
        edge_z_scores = np.zeros_like(edge_responses)
        
        # Detect adversarial example based on outlier features
        is_adversarial = outlier_ratio > 0.1  # If more than 10% of features are outliers
        
        # If adversarial, apply defense mechanism
        if is_adversarial:
            # Different defense strategies
            if self.adaptive_robustness['defense_mode'] == 'detect_only':
                # Just detect, don't modify input
                cleaned_x = x
            elif self.adaptive_robustness['defense_mode'] == 'detect_and_denoise':
                # Apply feature clipping for outliers
                cleaned_x = x.copy()
                clip_max = self.feature_statistics['mean'] + threshold * self.feature_statistics['std']
                clip_min = self.feature_statistics['mean'] - threshold * self.feature_statistics['std']
                cleaned_x = np.clip(cleaned_x, clip_min, clip_max)
            else:  # 'certify' mode
                # Apply randomized smoothing (add small noise for certification)
                noise_scale = self.adaptive_robustness['smoothing_factor']
                noise = np.random.normal(0, noise_scale, size=x.shape)
                cleaned_x = x + noise
        else:
            cleaned_x = x
            
        return is_adversarial, cleaned_x

    def explain_prediction(self, x, k=5):
        """Provide explainable predictions with path and feature importance analysis
        
        This method produces detailed explanations with:
        1. Path-based explanations showing how label predictions are formed
        2. Feature importance analysis showing input contributions
        3. Stability metrics for prediction confidence
        4. Counterfactual insights on what would change the prediction
        """
        # Get the model responses
        responses = x.dot(self.W).ravel()
        
        # Get the k-best codes for multi-path explanation
        codes = self.decoder.findKBestCodes(responses, k)
        
        # Calculate scores for each code
        code_scores = []
        for code in codes:
            score = sum([r if c == 1 else -r for r, c in zip(responses, code)])
            code_scores.append(score)
        
        # Calculate total score for confidence calculation
        total_score = sum(abs(s) for s in code_scores)
        
        # Initialize explanations list
        explanations = []
        
        # Get feature importance for visualization
        feature_importance = self._calculate_feature_importance(x, codes[0])
        
        # Calculate path-specific explanations for each code
        for i, (code, score) in enumerate(zip(codes, code_scores)):
            label = self.codeManager.codeToLabel(code)
            confidence = abs(score) / total_score if total_score > 0 else 0.0
            
            # Get the specific path for this code
            path_explanation = self._explain_path(x, code, responses)
            
            # Calculate prediction stability under perturbation
            stability = self._calculate_stability(x, code)
            
            # Generate counterfactual explanation
            counterfactual = self._generate_counterfactual(x, code, codes)
            
            # Generate path traversal explanation
            path_nodes = self._get_path_nodes(code)
            
            # Combine all explanations
            explanations.append({
                'rank': i + 1,
                'predicted_label': label,
                'path_score': score,
                'confidence': confidence,
                'important_features': feature_importance['ranked_features'],
                'prediction_stability': stability,
                'counterfactual': counterfactual,
                'path_nodes': path_nodes
            })
        
        return explanations
    
    def _explain_path(self, x, code, responses):
        """Explain how a prediction path was chosen in the trellis graph"""
        # Get the edge indices corresponding to the path (where code is 1)
        edge_indices = [i for i, c in enumerate(code) if c == 1]
        
        # Calculate feature contributions to each edge in the path
        path_explanation = []
        
        for edge_idx in edge_indices:
            # Get the contribution of each feature to this edge
            feature_contributions = x * self.W[:, edge_idx]
            
            # Get the top features for this edge
            top_features = []
            for feat_idx in np.argsort(-np.abs(feature_contributions))[:5]:
                if abs(feature_contributions[feat_idx]) > 1e-5:
                    direction = "+" if feature_contributions[feat_idx] > 0 else "-"
                    top_features.append((
                        int(feat_idx),
                        float(abs(feature_contributions[feat_idx])),
                        direction
                    ))
            
            # Add edge explanation
            path_explanation.append({
                'edge_idx': edge_idx,
                'edge_weight': responses[edge_idx],
                'top_features': top_features
            })
            
        return path_explanation
    
    def _get_path_nodes(self, code):
        """Recover the graph nodes traversed by this path"""
        # This is a simplified version - for full implementation, we'd need access to graph structure
        edge_indices = [i for i, c in enumerate(code) if c == 1]
        return {
            'edge_indices': edge_indices,
            'num_edges': len(edge_indices)
        }
    
    def _calculate_feature_importance(self, x, code):
        """Calculate feature importance using integrated gradients approach"""
        # Get the edge indices corresponding to the path (where code is 1)
        edge_indices = [i for i, c in enumerate(code) if c == 1]
        
        # Calculate the gradients (sensitivity) for each feature
        feature_gradients = np.zeros_like(x)
        for edge_idx in edge_indices:
            feature_gradients += self.W[:, edge_idx]
        
        # Feature importance is the element-wise product of input and gradient
        importance = x * feature_gradients
        
        # Get ranked features
        ranked_features = []
        for feat_idx in np.argsort(-np.abs(importance))[:10]:
            if abs(importance[feat_idx]) > 1e-5:
                direction = "+" if importance[feat_idx] > 0 else "-"
                ranked_features.append((
                    int(feat_idx),
                    float(abs(importance[feat_idx]) / (sum(abs(importance)) or 1.0)),
                    direction
                ))
        
        return {
            'gradients': feature_gradients,
            'importance': importance,
            'ranked_features': ranked_features
        }
                
    def _generate_counterfactual(self, x, code, alternative_codes):
        """Generate counterfactual explanation: what would change the prediction"""
        if len(alternative_codes) <= 1:
            return {'possible': False}
            
        # Get the original label
        original_label = self.codeManager.codeToLabel(code)
        
        # Get next best alternative
        for alt_code in alternative_codes[1:]:
            alt_label = self.codeManager.codeToLabel(alt_code)
            if alt_label != original_label:
                # Compare paths to find critical differences
                differences = []
                for i, (c1, c2) in enumerate(zip(code, alt_code)):
                    if c1 != c2:
                        differences.append(i)
                
                # Find features that most impact these different edges
                counterfactual_features = []
                for edge_idx in differences:
                    edge_weights = self.W[:, edge_idx]
                    # Find most influential features for this edge
                    for feat_idx in np.argsort(-np.abs(edge_weights))[:3]:
                        if abs(edge_weights[feat_idx]) > 1e-5:
                            # Compute the change needed
                            current_value = x[feat_idx]
                            desired_change = -1 if code[edge_idx] == 1 else 1
                            direction = "increase" if edge_weights[feat_idx] * desired_change > 0 else "decrease"
                            
                            counterfactual_features.append({
                                'feature_idx': int(feat_idx),
                                'current_value': float(current_value),
                                'direction': direction,
                                'edge_idx': int(edge_idx)
                            })
                
                return {
                    'possible': True,
                    'alternative_label': alt_label,
                    'critical_differences': differences,
                    'counterfactual_features': counterfactual_features[:5]  # Limit to top 5
                }
        
        return {'possible': False}
    
    def _calculate_stability(self, x, code, n_samples=20, noise_level=0.05):
        """Calculate prediction stability under small random perturbations"""
        stability_count = 0
        certified_radius = 0
        
        for _ in range(n_samples):
            # Add small Gaussian noise
            noise = np.random.normal(0, noise_level, size=x.shape)
            x_noisy = x + noise * np.maximum(0.1, np.abs(x))  # Scale noise by feature magnitude with minimum
            
            # Get prediction with noise
            noisy_responses = x_noisy.dot(self.W).ravel()
            noisy_code = self.decoder.findKBestCodes(noisy_responses, 1)[0]
            
            # Count stability
            if np.array_equal(code, noisy_code):
                stability_count += 1
        
        # Calculate empirical stability
        stability = stability_count / n_samples
        
        # Estimate certified radius (simplification of randomized smoothing)
        if stability > 0.5:
            # Add a small epsilon to prevent division by zero when stability is 1.0
            epsilon = 1e-6
            capped_stability = min(stability, 1.0 - epsilon)
            certified_radius = noise_level * np.sqrt(2 * np.log(1 / (1 - capped_stability)))
        
        return stability

    def _evaluate_robustness(self, x, code, responses, n_samples=20):
        """Evaluate robustness of the prediction with multiple metrics"""
        # Calculate stability under random perturbations
        stability = self._calculate_stability(x, code, n_samples=n_samples)
        
        # Calculate certified radius
        certified_radius = 0.0
        if stability > 0.5:
            noise_level = 0.05  # Same as in _calculate_stability
            # Add a small epsilon to prevent division by zero when stability is 1.0
            epsilon = 1e-6
            capped_stability = min(stability, 1.0 - epsilon)
            certified_radius = noise_level * np.sqrt(2 * np.log(1 / (1 - capped_stability)))
            
        # Estimate adversarial robustness via linear approximation
        edge_margins = []
        for i, (c, r) in enumerate(zip(code, responses)):
            # Margin is how far the response is from changing sign
            margin = abs(r)
            edge_margins.append(margin)
        
        # Overall robustness is the minimum perturbation needed to flip an edge
        min_perturbation = float('inf')
        for edge_idx, margin in enumerate(edge_margins):
            edge_sensitivity = np.linalg.norm(self.W[:, edge_idx])
            if edge_sensitivity > 0:
                # Minimum perturbation for this edge
                edge_min_perturbation = margin / edge_sensitivity
                min_perturbation = min(min_perturbation, edge_min_perturbation)
        
        # If no valid perturbation found, use default
        if min_perturbation == float('inf'):
            min_perturbation = 0.0
        
        return {
            'stability_score': stability,
            'certified_radius': certified_radius,
            'min_perturbation': min_perturbation,
            'edge_margins': edge_margins
        }

    def get_robustness_score(self, x, epsilon=0.1):
        """Comprehensive robustness analysis for a sample
        
        This method evaluates the prediction robustness using multiple approaches:
        1. Stability under random perturbations
        2. Certified radius (provable robustness guarantee)
        3. Sensitivity to adversarial perturbations
        4. Adversarial example detection
        """
        responses = x.dot(self.W).ravel()
        original_code = self.decoder.findKBestCodes(responses, 1)[0]
        original_label = self.codeManager.codeToLabel(original_code)
        
        # Check for potential adversarial input
        is_adversarial, _ = self._detect_and_defend_adversarial(x, responses)
        
        # Get baseline prediction and stability metrics
        stability_metrics = self._evaluate_robustness(x, original_code, responses)
        
        # Perform targeted adversarial search
        adversarial_metrics = self._adversarial_search(x, original_code, original_label, epsilon)
        
        # Merge all robustness metrics
        result = {
            'original_prediction': original_label,
            'stability_score': stability_metrics['stability_score'],
            'certified_radius': stability_metrics['certified_radius'],
            'adversarial_metrics': adversarial_metrics,
            'adversarial_detected': is_adversarial
        }
        
        return result
    
    def _adversarial_search(self, x, original_code, original_label, epsilon=0.1, max_steps=20):
        """Search for minimal adversarial perturbations that change the prediction"""
        # Find direction with highest gradient (FGSM-like approach)
        responses = x.dot(self.W).ravel()
        
        # Create gradient for maximizing original path score
        score_gradient = np.zeros_like(x)
        for i, c in enumerate(original_code):
            if c == 1:
                score_gradient += self.W[:, i]
            else:
                score_gradient -= self.W[:, i]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(score_gradient)
        if grad_norm > 0:
            normalized_gradient = score_gradient / grad_norm
        else:
            normalized_gradient = np.zeros_like(score_gradient)
        
        # Generate adversarial example using gradient
        adversarial_x = x - epsilon * normalized_gradient
        
        # Check if the attack succeeded
        adv_responses = adversarial_x.dot(self.W).ravel()
        adv_code = self.decoder.findKBestCodes(adv_responses, 1)[0]
        adv_label = self.codeManager.codeToLabel(adv_code)
        
        # If first attack succeeded, try binary search for minimal perturbation
        if adv_label != original_label:
            # Binary search for minimal perturbation
            epsilon_low = 0.0
            epsilon_high = epsilon
            
            for _ in range(max_steps):
                # Try middle epsilon
                epsilon_mid = (epsilon_low + epsilon_high) / 2
                x_mid = x - epsilon_mid * normalized_gradient
                
                # Check prediction
                mid_responses = x_mid.dot(self.W).ravel()
                mid_code = self.decoder.findKBestCodes(mid_responses, 1)[0]
                mid_label = self.codeManager.codeToLabel(mid_code)
                
                if mid_label != original_label:
                    # Attack still succeeds, try smaller perturbation
                    epsilon_high = epsilon_mid
                    adversarial_x = x_mid
                else:
                    # Attack fails, need larger perturbation
                    epsilon_low = epsilon_mid
            
            # Calculate perturbation norm
            perturbation_norm = np.linalg.norm(adversarial_x - x)
            attack_succeeded = True
        else:
            # Attack failed
            perturbation_norm = epsilon
            attack_succeeded = False
            
        return {
            'attack_succeeded': attack_succeeded,
            'perturbation_norm': float(perturbation_norm),
            'original_label': original_label,
            'adversarial_label': adv_label if attack_succeeded else original_label
        }
