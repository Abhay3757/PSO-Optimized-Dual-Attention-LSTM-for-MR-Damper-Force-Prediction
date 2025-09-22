"""
Training and Evaluation Module for MR Damper Force Prediction

This module provides comprehensive training, validation, and testing functionality
for the Dual-Attention LSTM model, including metrics computation, model checkpointing,
and performance analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from dual_attention_lstm import DualAttentionLSTM, ModelTrainer
from data_preprocessing import DataPreprocessor


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    early_stopping_patience: int = 15
    min_epochs: int = 20
    gradient_clip_norm: float = 1.0
    save_best_model: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    log_interval: int = 10
    validation_interval: int = 1


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    rmse: float
    mae: float
    r2: float
    mape: float
    training_time: float
    inference_time: float
    model_size_mb: float
    num_parameters: int


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class ModelTrainerAdvanced:
    """
    Advanced training utilities for the Dual-Attention LSTM model
    """
    
    def __init__(self, 
                 model: DualAttentionLSTM,
                 device: torch.device,
                 config: TrainingConfig,
                 save_dir: str = "checkpoints"):
        """
        Initialize trainer
        
        Args:
            model: The model to train
            device: Device to use for training
            config: Training configuration
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Setup loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            restore_best_weights=True
        )
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)['output'].squeeze()
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, torch.Tensor]]:
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average validation loss, attention weights sample)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        attention_weights = None
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output_dict = self.model(data)
                output = output_dict['output'].squeeze()
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store attention weights from first batch for analysis
                if batch_idx == 0:
                    attention_weights = {
                        'feature_attention': output_dict['feature_attention_weights'][:1].cpu(),
                        'temporal_attention': output_dict['temporal_attention_weights'][:1].cpu()
                    }
        
        return total_loss / num_batches, attention_weights
    
    def train_model(self, 
                   train_loader: DataLoader, 
                   val_loader: DataLoader) -> Dict[str, Any]:
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training results dictionary
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training configuration: {self.config}")
        print("=" * 60)
        
        training_start_time = time.time()
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase (if interval is met)
            if epoch % self.config.validation_interval == 0:
                val_loss, attention_weights = self.validate_epoch(val_loader)
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Check for best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.config.save_best_model:
                        self.save_checkpoint(epoch, val_loss, is_best=True)
                
                # Early stopping check
                if epoch >= self.config.min_epochs:
                    if self.early_stopping(val_loss, self.model):
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
            else:
                val_loss = None
            
            # Record history
            epoch_time = time.time() - epoch_start_time
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss if val_loss is not None else np.nan)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_time'].append(epoch_time)
            
            # Logging
            if epoch % self.config.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                val_loss_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss_str} | "
                      f"LR: {lr:.2e} | Time: {epoch_time:.2f}s")
            
            # Save checkpoint
            if (self.config.save_checkpoints and 
                epoch % self.config.checkpoint_interval == 0):
                self.save_checkpoint(epoch, val_loss or train_loss)
        
        training_time = time.time() - training_start_time
        
        print("=" * 60)
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return {
            'training_time': training_time,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_loss,
            'history': self.history,
            'attention_weights': attention_weights
        }
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'history': self.history,
            'config': self.config
        }
        
        if is_best:
            checkpoint_path = self.save_dir / 'best_model.pth'
        else:
            checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        return checkpoint


class ModelEvaluator:
    """
    Comprehensive model evaluation utilities
    """
    
    def __init__(self, model: DualAttentionLSTM, device: torch.device, preprocessor: DataPreprocessor):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            device: Device for evaluation
            preprocessor: Data preprocessor for inverse transforms
        """
        self.model = model.to(device)
        self.device = device
        self.preprocessor = preprocessor
        
    def evaluate_model(self, test_loader: DataLoader, phase: str = "test") -> Tuple[EvaluationMetrics, Dict]:
        """
        Comprehensive model evaluation
        
        Args:
            test_loader: Test data loader
            phase: Phase name for logging
            
        Returns:
            Tuple of (metrics, detailed_results)
        """
        print(f"Evaluating model on {phase} set...")
        
        self.model.eval()
        predictions = []
        targets = []
        feature_attentions = []
        temporal_attentions = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f"Evaluating {phase}"):
                data, target = data.to(self.device), target.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                output_dict = self.model(data)
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)
                
                # Collect predictions and targets
                output = output_dict['output'].squeeze()
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                
                # Collect attention weights
                feature_attentions.append(output_dict['feature_attention_weights'].cpu().numpy())
                temporal_attentions.append(output_dict['temporal_attention_weights'].cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        feature_attentions = np.concatenate(feature_attentions, axis=0)
        temporal_attentions = np.concatenate(temporal_attentions, axis=0)
        
        # Inverse transform to original scale
        predictions_orig = self.preprocessor.inverse_transform_target(predictions)
        targets_orig = self.preprocessor.inverse_transform_target(targets)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(targets_orig, predictions_orig))
        mae = mean_absolute_error(targets_orig, predictions_orig)
        r2 = r2_score(targets_orig, predictions_orig)
        
        # MAPE calculation (avoiding division by zero)
        mape = np.mean(np.abs((targets_orig - predictions_orig) / 
                             np.maximum(np.abs(targets_orig), 1e-8))) * 100
        
        # Model characteristics
        trainer = ModelTrainer(self.model, self.device)
        num_parameters = trainer.count_parameters()
        model_size_mb = trainer.get_model_size_mb()
        avg_inference_time = np.mean(inference_times)
        
        # Create metrics object
        metrics = EvaluationMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            mape=mape,
            training_time=0.0,  # To be filled by caller
            inference_time=avg_inference_time,
            model_size_mb=model_size_mb,
            num_parameters=num_parameters
        )
        
        # Detailed results
        detailed_results = {
            'predictions_normalized': predictions,
            'targets_normalized': targets,
            'predictions_original': predictions_orig,
            'targets_original': targets_orig,
            'feature_attention_weights': feature_attentions,
            'temporal_attention_weights': temporal_attentions,
            'inference_times': inference_times
        }
        
        print(f"{phase.capitalize()} Evaluation Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Avg Inference Time: {avg_inference_time:.2f} ms")
        print(f"  Model Size: {model_size_mb:.2f} MB")
        print(f"  Parameters: {num_parameters:,}")
        
        return metrics, detailed_results
    
    def analyze_attention_importance(self, 
                                   feature_attentions: np.ndarray,
                                   temporal_attentions: np.ndarray,
                                   feature_names: List[str]) -> Dict[str, Any]:
        """
        Analyze attention mechanism importance
        
        Args:
            feature_attentions: Feature attention weights
            temporal_attentions: Temporal attention weights
            feature_names: Names of input features
            
        Returns:
            Dictionary with attention analysis
        """
        # Feature importance analysis
        avg_feature_attention = np.mean(feature_attentions, axis=(0, 1))
        feature_importance = dict(zip(feature_names, avg_feature_attention))
        
        # Temporal importance analysis
        avg_temporal_attention = np.mean(temporal_attentions, axis=0)
        
        # Find most important timesteps
        important_timesteps = np.argsort(avg_temporal_attention)[-5:][::-1]
        
        return {
            'feature_importance': feature_importance,
            'avg_temporal_attention': avg_temporal_attention,
            'most_important_timesteps': important_timesteps.tolist(),
            'feature_attention_std': np.std(feature_attentions, axis=(0, 1)).tolist(),
            'temporal_attention_std': np.std(temporal_attentions, axis=0).tolist()
        }


def save_results(results: Dict[str, Any], save_path: str):
    """Save evaluation results to JSON file"""
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(save_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to: {save_path}")


def main():
    """Test the training and evaluation pipeline"""
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor(lookback_window=20)
    data_dict = preprocessor.preprocess_full_pipeline(
        csv_path="Dataset/ranran_50_converted.csv",
        batch_size=32
    )
    
    # Create model
    config = {
        'input_dim': data_dict['metadata']['num_features'],
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.2
    }
    
    from dual_attention_lstm import create_model
    model = create_model(config)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_config = TrainingConfig(
        epochs=20,  # Quick test
        early_stopping_patience=5,
        log_interval=5
    )
    
    trainer = ModelTrainerAdvanced(model, device, training_config)
    
    # Train model
    training_results = trainer.train_model(
        data_dict['dataloaders']['train'],
        data_dict['dataloaders']['val']
    )
    
    # Evaluate model
    evaluator = ModelEvaluator(model, device, preprocessor)
    test_metrics, test_results = evaluator.evaluate_model(
        data_dict['dataloaders']['test'], 
        phase="test"
    )
    
    # Analyze attention
    attention_analysis = evaluator.analyze_attention_importance(
        test_results['feature_attention_weights'],
        test_results['temporal_attention_weights'],
        data_dict['metadata']['feature_names']
    )
    
    print("\nAttention Analysis:")
    print("Feature Importance:")
    for feature, importance in attention_analysis['feature_importance'].items():
        print(f"  {feature}: {importance:.4f}")
    
    print(f"Most important timesteps: {attention_analysis['most_important_timesteps']}")


if __name__ == "__main__":
    main()