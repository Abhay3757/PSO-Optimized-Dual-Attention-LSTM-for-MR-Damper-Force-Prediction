"""
Multi-Objective Particle Swarm Optimization (MOPSO) for Hyperparameter Optimization

This module implements MOPSO to optimize hyperparameters of the Dual-Attention LSTM model
with multiple objectives:
1. Minimize RMSE (accuracy) - High Priority
2. Minimize inference latency (efficiency)
3. Minimize overfitting (validation-train error gap)

The optimization considers both performance and computational efficiency.
"""

import numpy as np
import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.termination import get_termination
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from dual_attention_lstm import DualAttentionLSTM, ModelTrainer
from data_preprocessing import DataPreprocessor


@dataclass
class HyperparameterBounds:
    """Define bounds for hyperparameters to optimize"""
    num_layers_min: int = 1
    num_layers_max: int = 4
    hidden_dim_min: int = 32
    hidden_dim_max: int = 256
    learning_rate_min: float = 1e-5
    learning_rate_max: float = 1e-2
    dropout_min: float = 0.0
    dropout_max: float = 0.5
    lookback_min: int = 10
    lookback_max: int = 50


@dataclass
class MOPSOConfig:
    """Configuration for MOPSO optimization"""
    population_size: int = 20
    max_generations: int = 10
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    tournament_size: int = 2
    weights: Tuple[float, float, float] = (0.90, 0.05, 0.05)  # RMSE, latency, overfitting
    early_stopping_patience: int = 5
    train_epochs: int = 50
    min_epochs: int = 10


class HyperparameterOptimizationProblem(Problem):
    """
    Multi-objective optimization problem for hyperparameter tuning
    """
    
    def __init__(self, 
                 data_dict: Dict,
                 bounds: HyperparameterBounds,
                 config: MOPSOConfig,
                 device: torch.device):
        """
        Initialize the optimization problem
        
        Args:
            data_dict: Preprocessed data dictionary
            bounds: Hyperparameter bounds
            config: MOPSO configuration
            device: Device for training
        """
        self.data_dict = data_dict
        self.bounds_config = bounds
        self.config = config
        self.device = device
        self.input_dim = data_dict['metadata']['num_features']
        
        # Define variable bounds
        xl = np.array([
            bounds.num_layers_min,
            bounds.hidden_dim_min,
            bounds.learning_rate_min,
            bounds.dropout_min,
            bounds.lookback_min
        ])
        
        xu = np.array([
            bounds.num_layers_max,
            bounds.hidden_dim_max,
            bounds.learning_rate_max,
            bounds.dropout_max,
            bounds.lookback_max
        ])
        
        super().__init__(
            n_var=5,  # num_layers, hidden_dim, learning_rate, dropout, lookback
            n_obj=3,  # RMSE, latency, overfitting
            xl=xl,
            xu=xu
        )
        
        self.evaluation_history = []
    
    def bounds(self):
        """
        Return bounds for optimization variables as expected by pymoo
        
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        return self.xl, self.xu
    
    def decode_variables(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Decode optimization variables to hyperparameters
        
        Args:
            x: Array of optimization variables
            
        Returns:
            Dictionary of hyperparameters
        """
        return {
            'num_layers': int(x[0]),
            'hidden_dim': int(x[1]),
            'learning_rate': float(x[2]),
            'dropout': float(x[3]),
            'lookback_window': int(x[4])
        }
    
    def create_model_and_data(self, hyperparams: Dict[str, Any]) -> Tuple[DualAttentionLSTM, Dict]:
        """
        Create model and prepare data with given hyperparameters
        
        Args:
            hyperparams: Hyperparameter dictionary
            
        Returns:
            Tuple of (model, data_loaders)
        """
        # Create model
        model_config = {
            'input_dim': self.input_dim,
            'hidden_dim': hyperparams['hidden_dim'],
            'num_layers': hyperparams['num_layers'],
            'dropout': hyperparams['dropout'],
            'feature_attention_dim': min(64, hyperparams['hidden_dim'] // 2),
            'temporal_attention_dim': min(64, hyperparams['hidden_dim'] // 2),
            'output_dim': 1
        }
        
        model = DualAttentionLSTM(**model_config).to(self.device)
        
        # Prepare data with new lookback window if needed
        if hyperparams['lookback_window'] != self.data_dict['metadata']['lookback_window']:
            # Re-create data with new lookback window
            preprocessor = DataPreprocessor(lookback_window=hyperparams['lookback_window'])
            
            # Use the normalized data but recreate sequences
            train_norm = self.data_dict['normalized_data']['train']
            val_norm = self.data_dict['normalized_data']['val']
            
            train_seq, train_targets = preprocessor.create_time_series_windows(train_norm)
            val_seq, val_targets = preprocessor.create_time_series_windows(val_norm)
            
            # Create new dataloaders
            from torch.utils.data import DataLoader
            from data_preprocessing import MRDamperDataset
            
            train_dataset = MRDamperDataset(train_seq, train_targets)
            val_dataset = MRDamperDataset(val_seq, val_targets)
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            
            data_loaders = {'train': train_loader, 'val': val_loader}
        else:
            # Use existing data loaders
            data_loaders = {
                'train': self.data_dict['dataloaders']['train'],
                'val': self.data_dict['dataloaders']['val']
            }
        
        return model, data_loaders
    
    def train_and_evaluate_model(self, hyperparams: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Train and evaluate model with given hyperparameters
        
        Args:
            hyperparams: Hyperparameter dictionary
            
        Returns:
            Tuple of (rmse, inference_latency, overfitting_gap)
        """
        try:
            # Create model and data
            model, data_loaders = self.create_model_and_data(hyperparams)
            trainer = ModelTrainer(model, self.device)
            
            # Setup training
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
            
            # Training loop with early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(self.config.train_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                for batch_idx, (data, target) in enumerate(data_loaders['train']):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(data)['output'].squeeze()
                    loss = criterion(output, target)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(data_loaders['train'])
                train_losses.append(avg_train_loss)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, target in data_loaders['val']:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)['output'].squeeze()
                        loss = criterion(output, target)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(data_loaders['val'])
                val_losses.append(avg_val_loss)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience and epoch >= self.config.min_epochs:
                    break
            
            # Calculate metrics
            final_train_loss = min(train_losses[-5:])  # Average of last 5 epochs
            final_val_loss = min(val_losses[-5:])
            
            rmse = np.sqrt(final_val_loss)
            
            # Measure inference latency
            sample_input = next(iter(data_loaders['val']))[0][:1]  # Single sample
            inference_latency = trainer.estimate_inference_time(sample_input, num_runs=50)
            
            # Calculate overfitting gap
            overfitting_gap = abs(final_val_loss - final_train_loss)
            
            return rmse, inference_latency, overfitting_gap
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            # Return penalty values for failed evaluations
            return 10.0, 1000.0, 10.0
    
    def _evaluate(self, x: np.ndarray, out: Dict, *args, **kwargs):
        """
        Evaluate the multi-objective function
        
        Args:
            x: Design variables (hyperparameters)
            out: Output dictionary to store objectives
        """
        n_samples = x.shape[0]
        objectives = np.zeros((n_samples, self.n_obj))
        
        for i in range(n_samples):
            hyperparams = self.decode_variables(x[i])
            
            print(f"Evaluating individual {i+1}/{n_samples}: {hyperparams}")
            
            rmse, latency, overfitting = self.train_and_evaluate_model(hyperparams)
            
            # Store objectives (to be minimized)
            objectives[i, 0] = rmse  # Accuracy (RMSE)
            objectives[i, 1] = latency / 100.0  # Efficiency (normalized latency)
            objectives[i, 2] = overfitting  # Generalization (overfitting gap)
            
            # Store evaluation history
            evaluation_result = {
                'hyperparams': hyperparams,
                'rmse': rmse,
                'latency': latency,
                'overfitting': overfitting,
                'weighted_score': (
                    self.config.weights[0] * rmse + 
                    self.config.weights[1] * (latency / 100.0) + 
                    self.config.weights[2] * overfitting
                )
            }
            self.evaluation_history.append(evaluation_result)
            
            print(f"Results - RMSE: {rmse:.4f}, Latency: {latency:.2f}ms, Overfitting: {overfitting:.4f}")
        
        out["F"] = objectives


class MOPSOOptimizer:
    """
    Multi-Objective Particle Swarm Optimization for hyperparameter tuning
    """
    
    def __init__(self, 
                 data_dict: Dict,
                 bounds: Optional[HyperparameterBounds] = None,
                 config: Optional[MOPSOConfig] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize MOPSO optimizer
        
        Args:
            data_dict: Preprocessed data dictionary
            bounds: Hyperparameter bounds
            config: MOPSO configuration
            device: Device for training
        """
        self.data_dict = data_dict
        self.bounds_config = bounds or HyperparameterBounds()
        self.config = config or MOPSOConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.optimization_results = None
        self.best_hyperparams = None
        
    def optimize(self) -> Dict[str, Any]:
        """
        Run MOPSO optimization
        
        Returns:
            Dictionary with optimization results
        """
        print("=" * 60)
        print("Starting Multi-Objective Particle Swarm Optimization (MOPSO)")
        print("=" * 60)
        print(f"Population size: {self.config.population_size}")
        print(f"Max generations: {self.config.max_generations}")
        print(f"Device: {self.device}")
        print(f"Objectives: RMSE (weight: {self.config.weights[0]}), "
              f"Latency (weight: {self.config.weights[1]}), "
              f"Overfitting (weight: {self.config.weights[2]})")
        print("=" * 60)
        
        # Create optimization problem
        problem = HyperparameterOptimizationProblem(
            self.data_dict, self.bounds_config, self.config, self.device
        )
        
        # Configure NSGA-II algorithm (MOPSO equivalent in pymoo)
        algorithm = NSGA2(
            pop_size=self.config.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=self.config.crossover_prob, eta=15),
            mutation=PM(prob=self.config.mutation_prob, eta=20),
            eliminate_duplicates=True
        )
        
        # Setup termination criterion
        termination = get_termination("n_gen", self.config.max_generations)
        
        # Run optimization
        start_time = time.time()
        
        res = minimize(
            problem,
            algorithm,
            termination,
            verbose=True,
            save_history=True
        )
        
        optimization_time = time.time() - start_time
        
        print("=" * 60)
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print("=" * 60)
        
        # Process results
        self.optimization_results = res
        self._analyze_results(problem)
        
        return self._prepare_results_summary(optimization_time)
    
    def _analyze_results(self, problem: HyperparameterOptimizationProblem):
        """Analyze optimization results and find best hyperparameters"""
        
        # Get Pareto front
        pareto_front_F = self.optimization_results.F
        pareto_front_X = self.optimization_results.X
        
        # Calculate weighted scores for Pareto front solutions
        weighted_scores = []
        for i, objectives in enumerate(pareto_front_F):
            weighted_score = (
                self.config.weights[0] * objectives[0] +  # RMSE
                self.config.weights[1] * objectives[1] +  # Latency
                self.config.weights[2] * objectives[2]    # Overfitting
            )
            weighted_scores.append((weighted_score, i))
        
        # Find best solution based on weighted score
        best_idx = min(weighted_scores, key=lambda x: x[0])[1]
        best_variables = pareto_front_X[best_idx]
        best_objectives = pareto_front_F[best_idx]
        
        self.best_hyperparams = problem.decode_variables(best_variables)
        self.best_objectives = best_objectives
        
        print(f"\nBest hyperparameters found:")
        for key, value in self.best_hyperparams.items():
            print(f"  {key}: {value}")
        
        print(f"\nBest objectives:")
        print(f"  RMSE: {best_objectives[0]:.4f}")
        print(f"  Latency: {best_objectives[1]*100:.2f} ms")
        print(f"  Overfitting: {best_objectives[2]:.4f}")
        print(f"  Weighted Score: {min(weighted_scores)[0]:.4f}")
    
    def _prepare_results_summary(self, optimization_time: float) -> Dict[str, Any]:
        """Prepare comprehensive results summary"""
        
        return {
            'best_hyperparams': self.best_hyperparams,
            'best_objectives': {
                'rmse': float(self.best_objectives[0]),
                'latency_ms': float(self.best_objectives[1] * 100),
                'overfitting_gap': float(self.best_objectives[2])
            },
            'pareto_front': {
                'objectives': self.optimization_results.F.tolist(),
                'variables': self.optimization_results.X.tolist()
            },
            'optimization_time': optimization_time,
            'config': {
                'population_size': self.config.population_size,
                'max_generations': self.config.max_generations,
                'weights': self.config.weights
            },
            'convergence_history': [
                {'gen': i, 'best_f': hist.opt[0].F.min(axis=0).tolist()}
                for i, hist in enumerate(self.optimization_results.history)
            ]
        }
    
    def get_best_hyperparams(self) -> Dict[str, Any]:
        """Get the best hyperparameters found"""
        if self.best_hyperparams is None:
            raise ValueError("Optimization has not been run yet")
        return self.best_hyperparams.copy()


def main():
    """Test the MOPSO optimizer with a small example"""
    print("Testing MOPSO Optimizer with reduced settings...")
    
    # Load sample data
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor(lookback_window=20)
    data_dict = preprocessor.preprocess_full_pipeline(
        csv_path="Dataset/ranran_50_converted.csv",
        batch_size=32
    )
    
    # Configure for quick test
    bounds = HyperparameterBounds(
        num_layers_min=1, num_layers_max=2,
        hidden_dim_min=32, hidden_dim_max=64,
        lookback_min=15, lookback_max=25
    )
    
    config = MOPSOConfig(
        population_size=4,  # Very small for testing
        max_generations=2,  # Very few generations
        train_epochs=10     # Few training epochs
    )
    
    # Run optimization
    optimizer = MOPSOOptimizer(data_dict, bounds, config)
    results = optimizer.optimize()
    
    print("\nOptimization Results:")
    print(f"Best hyperparameters: {results['best_hyperparams']}")
    print(f"Best objectives: {results['best_objectives']}")


if __name__ == "__main__":
    main()