"""
Main Pipeline for PSO-Optimized Dual-Attention LSTM MR Damper Force Prediction

This is the main orchestration script that integrates all components:
1. Data preprocessing
2. MOPSO hyperparameter optimization  
3. Model training with optimal hyperparameters
4. Comprehensive evaluation
5. Visualization and reporting

The pipeline is designed to be modular, configurable, and reproducible.
"""

import torch
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import argparse
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all our modules
from data_preprocessing import DataPreprocessor
from dual_attention_lstm import create_model, DualAttentionLSTM
from mopso_optimizer import MOPSOOptimizer, HyperparameterBounds, MOPSOConfig
from training import ModelTrainerAdvanced, TrainingConfig, ModelEvaluator
from visualization import MRDamperVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mr_damper_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MRDamperPipeline:
    """
    Main pipeline class for MR Damper force prediction
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path(config.get('results_dir', 'results'))
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.results_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Initialize components
        self.preprocessor = None
        self.optimizer = None
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.visualizer = MRDamperVisualizer(save_dir=str(self.results_dir / 'visualizations'))
        
        # Results storage
        self.pipeline_results = {}
        
        logger.info(f"Pipeline initialized with device: {self.device}")
        logger.info(f"Results will be saved to: {self.results_dir}")
    
    def run_data_preprocessing(self) -> Dict[str, Any]:
        """
        Run data preprocessing step
        
        Returns:
            Preprocessed data dictionary
        """
        logger.info("=" * 60)
        logger.info("Step 1: Data Preprocessing")
        logger.info("=" * 60)
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(
            normalize_method=self.config['preprocessing']['normalize_method'],
            lookback_window=self.config['preprocessing']['lookback_window']
        )
        
        # Run preprocessing
        data_dict = self.preprocessor.preprocess_full_pipeline(
            csv_path=self.config['data']['csv_path'],
            train_ratio=self.config['data']['train_ratio'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio'],
            batch_size=self.config['training']['batch_size']
        )
        
        self.pipeline_results['data_preprocessing'] = {
            'num_features': data_dict['metadata']['num_features'],
            'feature_names': data_dict['metadata']['feature_names'],
            'train_samples': len(data_dict['datasets']['train']),
            'val_samples': len(data_dict['datasets']['val']),
            'test_samples': len(data_dict['datasets']['test']),
            'lookback_window': data_dict['metadata']['lookback_window']
        }
        
        logger.info("Data preprocessing completed successfully")
        return data_dict
    
    def run_hyperparameter_optimization(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run MOPSO hyperparameter optimization
        
        Args:
            data_dict: Preprocessed data dictionary
            
        Returns:
            Optimization results
        """
        logger.info("=" * 60)
        logger.info("Step 2: MOPSO Hyperparameter Optimization")
        logger.info("=" * 60)
        
        if not self.config['optimization']['enable_mopso']:
            logger.info("MOPSO optimization disabled, using default hyperparameters")
            default_hyperparams = self.config['model']['default_hyperparams']
            return {'best_hyperparams': default_hyperparams, 'optimization_time': 0}
        
        # Setup optimization bounds and config
        bounds = HyperparameterBounds(
            **self.config['optimization']['bounds']
        )
        
        mopso_config = MOPSOConfig(
            **self.config['optimization']['mopso_config']
        )
        
        # Initialize and run optimizer
        self.optimizer = MOPSOOptimizer(data_dict, bounds, mopso_config, self.device)
        optimization_results = self.optimizer.optimize()
        
        self.pipeline_results['optimization'] = optimization_results
        
        logger.info("MOPSO optimization completed successfully")
        return optimization_results
    
    def run_model_training(self, 
                          data_dict: Dict[str, Any], 
                          hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run model training with optimized hyperparameters
        
        Args:
            data_dict: Preprocessed data dictionary
            hyperparams: Optimized hyperparameters
            
        Returns:
            Training results
        """
        logger.info("=" * 60)
        logger.info("Step 3: Model Training")
        logger.info("=" * 60)
        
        # Create model with optimized hyperparameters
        model_config = {
            'input_dim': data_dict['metadata']['num_features'],
            'hidden_dim': hyperparams['hidden_dim'],
            'num_layers': hyperparams['num_layers'],
            'dropout': hyperparams['dropout'],
            'feature_attention_dim': min(64, hyperparams['hidden_dim'] // 2),
            'temporal_attention_dim': min(64, hyperparams['hidden_dim'] // 2),
            'output_dim': 1
        }
        
        self.model = create_model(model_config)
        
        # Setup training configuration
        training_config = TrainingConfig(
            **self.config['training']['training_config'],
            learning_rate=hyperparams['learning_rate']
        )
        
        # Initialize trainer
        self.trainer = ModelTrainerAdvanced(
            model=self.model,
            device=self.device,
            config=training_config,
            save_dir=str(self.results_dir / 'checkpoints')
        )
        
        # Handle different lookback windows
        if hyperparams['lookback_window'] != data_dict['metadata']['lookback_window']:
            logger.info(f"Re-creating data with lookback window: {hyperparams['lookback_window']}")
            
            # Create new preprocessor with optimized lookback window
            temp_preprocessor = DataPreprocessor(
                normalize_method=self.config['preprocessing']['normalize_method'],
                lookback_window=hyperparams['lookback_window']
            )
            temp_preprocessor.feature_scaler = self.preprocessor.feature_scaler
            temp_preprocessor.target_scaler = self.preprocessor.target_scaler
            
            # Re-create sequences with new lookback window
            train_seq, train_targets = temp_preprocessor.create_time_series_windows(
                data_dict['normalized_data']['train']
            )
            val_seq, val_targets = temp_preprocessor.create_time_series_windows(
                data_dict['normalized_data']['val']
            )
            
            # Create new dataloaders
            from torch.utils.data import DataLoader
            from data_preprocessing import MRDamperDataset
            
            train_dataset = MRDamperDataset(train_seq, train_targets)
            val_dataset = MRDamperDataset(val_seq, val_targets)
            
            train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)
        else:
            train_loader = data_dict['dataloaders']['train']
            val_loader = data_dict['dataloaders']['val']
        
        # Train model
        training_results = self.trainer.train_model(train_loader, val_loader)
        
        self.pipeline_results['training'] = {
            'model_config': model_config,
            'training_config': training_config.__dict__,
            'hyperparameters': hyperparams,
            'training_time': training_results['training_time'],
            'best_val_loss': training_results['best_val_loss'],
            'final_train_loss': training_results['final_train_loss']
        }
        
        logger.info("Model training completed successfully")
        return training_results
    
    def run_model_evaluation(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive model evaluation
        
        Args:
            data_dict: Preprocessed data dictionary
            
        Returns:
            Evaluation results
        """
        logger.info("=" * 60)
        logger.info("Step 4: Model Evaluation")
        logger.info("=" * 60)
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(self.model, self.device, self.preprocessor)
        
        # Evaluate on test set
        test_metrics, test_results = self.evaluator.evaluate_model(
            data_dict['dataloaders']['test'], 
            phase="test"
        )
        
        # Evaluate on validation set for comparison
        val_metrics, val_results = self.evaluator.evaluate_model(
            data_dict['dataloaders']['val'], 
            phase="validation"
        )
        
        # Analyze attention mechanisms
        attention_analysis = self.evaluator.analyze_attention_importance(
            test_results['feature_attention_weights'],
            test_results['temporal_attention_weights'],
            data_dict['metadata']['feature_names']
        )
        
        evaluation_results = {
            'test_metrics': test_metrics.__dict__,
            'val_metrics': val_metrics.__dict__,
            'test_results': test_results,
            'val_results': val_results,
            'attention_analysis': attention_analysis
        }
        
        self.pipeline_results['evaluation'] = evaluation_results
        
        logger.info("Model evaluation completed successfully")
        return evaluation_results
    
    def run_visualization(self, 
                         data_dict: Dict[str, Any], 
                         evaluation_results: Dict[str, Any],
                         training_results: Dict[str, Any]) -> None:
        """
        Generate comprehensive visualizations
        
        Args:
            data_dict: Preprocessed data dictionary
            evaluation_results: Evaluation results
            training_results: Training results
        """
        logger.info("=" * 60)
        logger.info("Step 5: Visualization Generation")
        logger.info("=" * 60)
        
        test_results = evaluation_results['test_results']
        
        # Get original data for hysteresis and velocity plots
        test_df = data_dict['raw_data']['test']
        # Align the test data with predictions (account for lookback window)
        lookback = data_dict['metadata']['lookback_window']
        aligned_test_df = test_df.iloc[lookback:lookback + len(test_results['predictions_original'])].reset_index(drop=True)
        
        # 1. Predictions vs Actual
        self.visualizer.plot_predictions_vs_actual(
            test_results['predictions_original'],
            test_results['targets_original'],
            save_name="test_predictions_vs_actual.png"
        )
        
        # 2. Hysteresis loops
        if len(aligned_test_df) > 0:
            self.visualizer.plot_hysteresis_loops(
                aligned_test_df['D'].values,
                test_results['targets_original'],
                test_results['predictions_original'],
                save_name="hysteresis_loops.png"
            )
            
            # 3. Force-velocity loops
            self.visualizer.plot_force_velocity_loops(
                aligned_test_df['V'].values,
                test_results['targets_original'],
                test_results['predictions_original'],
                save_name="force_velocity_loops.png"
            )
        
        # 4. Attention heatmaps
        self.visualizer.plot_attention_heatmaps(
            test_results['feature_attention_weights'],
            test_results['temporal_attention_weights'],
            data_dict['metadata']['feature_names'],
            num_samples=5,
            save_name="attention_heatmaps.png"
        )
        
        # 5. Feature importance
        self.visualizer.plot_feature_importance(
            evaluation_results['attention_analysis']['feature_importance'],
            save_name="feature_importance.png"
        )
        
        # 6. Training history
        if 'history' in training_results:
            self.visualizer.plot_training_history(
                training_results['history'],
                save_name="training_history.png"
            )
        
        # 7. Error analysis
        self.visualizer.plot_error_analysis(
            test_results['predictions_original'],
            test_results['targets_original'],
            save_name="error_analysis.png"
        )
        
        # 8. Performance metrics
        self.visualizer.plot_performance_metrics(
            evaluation_results['test_metrics'],
            save_name="performance_metrics.png"
        )
        
        logger.info("Visualization generation completed successfully")
    
    def save_final_results(self) -> None:
        """Save final pipeline results to JSON file"""
        
        # Add pipeline metadata
        self.pipeline_results['pipeline_metadata'] = {
            'run_timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'config': self.config,
            'total_pipeline_time': getattr(self, 'total_pipeline_time', 0)
        }
        
        # Save results
        results_file = self.results_dir / 'pipeline_results.json'
        
        # Convert numpy arrays and non-serializable objects
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return obj
        
        # Recursively convert results
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {key: deep_convert(value) for key, value in obj.items() 
                       if not key.endswith('_weights') and not key.endswith('_normalized')}  # Skip large arrays
            elif isinstance(obj, list):
                return [deep_convert(item) for item in obj]
            else:
                return convert_for_json(obj)
        
        serializable_results = deep_convert(self.pipeline_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Final results saved to: {results_file}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline from start to finish
        
        Returns:
            Complete pipeline results
        """
        start_time = time.time()
        
        logger.info("Starting MR Damper Force Prediction Pipeline")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        try:
            # Step 1: Data Preprocessing
            data_dict = self.run_data_preprocessing()
            
            # Step 2: MOPSO Optimization
            optimization_results = self.run_hyperparameter_optimization(data_dict)
            best_hyperparams = optimization_results['best_hyperparams']
            
            # Step 3: Model Training
            training_results = self.run_model_training(data_dict, best_hyperparams)
            
            # Step 4: Model Evaluation
            evaluation_results = self.run_model_evaluation(data_dict)
            
            # Step 5: Visualization
            self.run_visualization(data_dict, evaluation_results, training_results)
            
            # Calculate total time
            self.total_pipeline_time = time.time() - start_time
            
            # Save final results
            self.save_final_results()
            
            logger.info("=" * 60)
            logger.info("Pipeline completed successfully!")
            logger.info(f"Total pipeline time: {self.total_pipeline_time:.2f} seconds")
            logger.info(f"Best RMSE: {evaluation_results['test_metrics']['rmse']:.4f}")
            logger.info(f"Best R²: {evaluation_results['test_metrics']['r2']:.4f}")
            logger.info(f"Results saved to: {self.results_dir}")
            logger.info("=" * 60)
            
            return self.pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    return {
        "data": {
            "csv_path": "Dataset/ranran_50_converted.csv",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15
        },
        "preprocessing": {
            "normalize_method": "standard",
            "lookback_window": 20
        },
        "optimization": {
            "enable_mopso": True,
            "bounds": {
                "num_layers_min": 1,
                "num_layers_max": 3,
                "hidden_dim_min": 64,
                "hidden_dim_max": 256,
                "learning_rate_min": 1e-4,
                "learning_rate_max": 1e-2,
                "dropout_min": 0.0,
                "dropout_max": 0.4,
                "lookback_min": 15,
                "lookback_max": 30
            },
            "mopso_config": {
                "population_size": 8,
                "max_generations": 5,
                "train_epochs": 30,
                "weights": [0.7, 0.2, 0.1]
            }
        },
        "model": {
            "default_hyperparams": {
                "num_layers": 2,
                "hidden_dim": 128,
                "learning_rate": 1e-3,
                "dropout": 0.2,
                "lookback_window": 20
            }
        },
        "training": {
            "batch_size": 64,
            "training_config": {
                "epochs": 100,
                "early_stopping_patience": 15,
                "min_epochs": 20,
                "log_interval": 10,
                "save_best_model": True,
                "save_checkpoints": True
            }
        },
        "results_dir": "results"
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MR Damper Force Prediction Pipeline')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to configuration JSON file')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with minimal settings')
    
    args = parser.parse_args()
    
    if args.create_config:
        config = create_default_config()
        with open('pipeline_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("Default configuration saved to 'pipeline_config.json'")
        return
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
        
        if args.quick_test:
            # Modify config for quick testing
            config['optimization']['enable_mopso'] = False
            config['training']['training_config']['epochs'] = 20
            config['training']['training_config']['early_stopping_patience'] = 5
            logger.info("Running in quick test mode")
    
    # Initialize and run pipeline
    pipeline = MRDamperPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    return results


if __name__ == "__main__":
    main()