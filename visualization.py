"""
Visualization Module for MR Damper Force Prediction

This module provides comprehensive visualization tools for:
1. Predicted vs ground truth force comparison
2. Force-displacement hysteresis loops
3. Force-velocity loops
4. Attention mechanism heatmaps (feature and temporal)
5. Training progress and convergence
6. Model performance metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MRDamperVisualizer:
    """
    Comprehensive visualization class for MR Damper analysis
    """
    
    def __init__(self, save_dir: str = "visualizations", dpi: int = 300, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save plots
            dpi: Resolution for saved plots
            figsize: Default figure size
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'tertiary': '#F18F01',
            'quaternary': '#C73E1D',
            'success': '#2ECC71',
            'warning': '#F39C12',
            'error': '#E74C3C',
            'info': '#3498DB',
            'dark': '#2C3E50',
            'light': '#BDC3C7'
        }
    
    def plot_predictions_vs_actual(self, 
                                 predictions: np.ndarray,
                                 actual: np.ndarray,
                                 title: str = "Predicted vs Actual Force",
                                 save_name: str = "predictions_vs_actual.png") -> None:
        """
        Plot predicted vs actual force values
        
        Args:
            predictions: Predicted force values
            actual: Actual force values
            title: Plot title
            save_name: Filename to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(actual, predictions, alpha=0.6, color=self.colors['primary'], s=20)
        
        # Perfect prediction line
        min_val = min(actual.min(), predictions.min())
        max_val = max(actual.max(), predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Force (N)', fontsize=12)
        ax1.set_ylabel('Predicted Force (N)', fontsize=12)
        ax1.set_title('Predicted vs Actual Force', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(actual, predictions)
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                fontsize=12)
        
        # Time series comparison
        time_indices = np.arange(len(actual))
        ax2.plot(time_indices, actual, label='Actual', color=self.colors['secondary'], linewidth=1.5)
        ax2.plot(time_indices, predictions, label='Predicted', color=self.colors['primary'], 
                linewidth=1.5, alpha=0.8)
        
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Force (N)', fontsize=12)
        ax2.set_title('Time Series Comparison', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_hysteresis_loops(self,
                            displacement: np.ndarray,
                            force_actual: np.ndarray,
                            force_predicted: np.ndarray,
                            title: str = "Force-Displacement Hysteresis Loops",
                            save_name: str = "hysteresis_loops.png") -> None:
        """
        Plot force-displacement hysteresis loops
        
        Args:
            displacement: Displacement values
            force_actual: Actual force values
            force_predicted: Predicted force values
            title: Plot title
            save_name: Filename to save the plot
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Actual hysteresis loop
        ax1.plot(displacement, force_actual, color=self.colors['secondary'], 
                linewidth=2, label='Actual', alpha=0.8)
        ax1.set_xlabel('Displacement (m)', fontsize=12)
        ax1.set_ylabel('Force (N)', fontsize=12)
        ax1.set_title('Actual Hysteresis', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Predicted hysteresis loop
        ax2.plot(displacement, force_predicted, color=self.colors['primary'], 
                linewidth=2, label='Predicted', alpha=0.8)
        ax2.set_xlabel('Displacement (m)', fontsize=12)
        ax2.set_ylabel('Force (N)', fontsize=12)
        ax2.set_title('Predicted Hysteresis', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Comparison
        ax3.plot(displacement, force_actual, color=self.colors['secondary'], 
                linewidth=2, label='Actual', alpha=0.7)
        ax3.plot(displacement, force_predicted, color=self.colors['primary'], 
                linewidth=2, label='Predicted', alpha=0.7, linestyle='--')
        ax3.set_xlabel('Displacement (m)', fontsize=12)
        ax3.set_ylabel('Force (N)', fontsize=12)
        ax3.set_title('Comparison', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_force_velocity_loops(self,
                                velocity: np.ndarray,
                                force_actual: np.ndarray,
                                force_predicted: np.ndarray,
                                title: str = "Force-Velocity Loops",
                                save_name: str = "force_velocity_loops.png") -> None:
        """
        Plot force-velocity loops
        
        Args:
            velocity: Velocity values
            force_actual: Actual force values
            force_predicted: Predicted force values
            title: Plot title
            save_name: Filename to save the plot
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Actual force-velocity loop
        ax1.plot(velocity, force_actual, color=self.colors['tertiary'], 
                linewidth=2, label='Actual', alpha=0.8)
        ax1.set_xlabel('Velocity (m/s)', fontsize=12)
        ax1.set_ylabel('Force (N)', fontsize=12)
        ax1.set_title('Actual Force-Velocity', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Predicted force-velocity loop
        ax2.plot(velocity, force_predicted, color=self.colors['quaternary'], 
                linewidth=2, label='Predicted', alpha=0.8)
        ax2.set_xlabel('Velocity (m/s)', fontsize=12)
        ax2.set_ylabel('Force (N)', fontsize=12)
        ax2.set_title('Predicted Force-Velocity', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Comparison
        ax3.plot(velocity, force_actual, color=self.colors['tertiary'], 
                linewidth=2, label='Actual', alpha=0.7)
        ax3.plot(velocity, force_predicted, color=self.colors['quaternary'], 
                linewidth=2, label='Predicted', alpha=0.7, linestyle='--')
        ax3.set_xlabel('Velocity (m/s)', fontsize=12)
        ax3.set_ylabel('Force (N)', fontsize=12)
        ax3.set_title('Comparison', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_attention_heatmaps(self,
                              feature_attention: np.ndarray,
                              temporal_attention: np.ndarray,
                              feature_names: List[str],
                              num_samples: int = 5,
                              save_name: str = "attention_heatmaps.png") -> None:
        """
        Plot attention mechanism heatmaps
        
        Args:
            feature_attention: Feature attention weights (samples, timesteps, features)
            temporal_attention: Temporal attention weights (samples, timesteps)
            feature_names: Names of input features
            num_samples: Number of samples to visualize
            save_name: Filename to save the plot
        """
        # Select random samples for visualization
        selected_samples = np.random.choice(len(feature_attention), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
        if num_samples == 1:
            axes = axes.reshape(2, 1)
        
        # Feature attention heatmaps
        for i, sample_idx in enumerate(selected_samples):
            fa = feature_attention[sample_idx].T  # (features, timesteps)
            
            im1 = axes[0, i].imshow(fa, cmap='YlOrRd', aspect='auto')
            axes[0, i].set_title(f'Feature Attention - Sample {sample_idx}', fontsize=12)
            axes[0, i].set_xlabel('Time Steps')
            axes[0, i].set_ylabel('Features')
            axes[0, i].set_yticks(range(len(feature_names)))
            axes[0, i].set_yticklabels(feature_names)
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            cbar1.set_label('Attention Weight')
        
        # Temporal attention heatmaps
        for i, sample_idx in enumerate(selected_samples):
            ta = temporal_attention[sample_idx].reshape(1, -1)  # (1, timesteps)
            
            im2 = axes[1, i].imshow(ta, cmap='Blues', aspect='auto')
            axes[1, i].set_title(f'Temporal Attention - Sample {sample_idx}', fontsize=12)
            axes[1, i].set_xlabel('Time Steps')
            axes[1, i].set_ylabel('Attention')
            axes[1, i].set_yticks([0])
            axes[1, i].set_yticklabels(['Temporal'])
            
            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            cbar2.set_label('Attention Weight')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self,
                              feature_importance: Dict[str, float],
                              title: str = "Average Feature Importance",
                              save_name: str = "feature_importance.png") -> None:
        """
        Plot feature importance from attention weights
        
        Args:
            feature_importance: Dictionary of feature names and their importance scores
            title: Plot title
            save_name: Filename to save the plot
        """
        features = list(feature_importance.keys())
        importance_scores = list(feature_importance.values())
        
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        bars = plt.bar(features, importance_scores, color=[self.colors['primary'], 
                                                         self.colors['secondary'], 
                                                         self.colors['tertiary'], 
                                                         self.colors['quaternary']])
        
        plt.xlabel('Input Features', fontsize=12)
        plt.ylabel('Average Attention Weight', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars, importance_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(importance_scores),
                    f'{score:.3f}', ha='center', va='bottom', fontsize=11)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self,
                            history: Dict[str, List],
                            save_name: str = "training_history.png") -> None:
        """
        Plot training and validation history
        
        Args:
            history: Training history dictionary
            save_name: Filename to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Training and validation loss
        ax1.plot(epochs, history['train_loss'], label='Training Loss', 
                color=self.colors['primary'], linewidth=2)
        if 'val_loss' in history and not all(np.isnan(history['val_loss'])):
            valid_val_loss = [x for x in history['val_loss'] if not np.isnan(x)]
            valid_epochs = [i+1 for i, x in enumerate(history['val_loss']) if not np.isnan(x)]
            ax1.plot(valid_epochs, valid_val_loss, label='Validation Loss', 
                    color=self.colors['secondary'], linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2.plot(epochs, history['learning_rate'], color=self.colors['tertiary'], linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Epoch time
        ax3.plot(epochs, history['epoch_time'], color=self.colors['quaternary'], linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Training Time per Epoch')
        ax3.grid(True, alpha=0.3)
        
        # Loss convergence (log scale)
        ax4.semilogy(epochs, history['train_loss'], label='Training Loss', 
                    color=self.colors['primary'], linewidth=2)
        if 'val_loss' in history and not all(np.isnan(history['val_loss'])):
            ax4.semilogy(valid_epochs, valid_val_loss, label='Validation Loss', 
                        color=self.colors['secondary'], linewidth=2)
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss (log scale)')
        ax4.set_title('Loss Convergence')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_error_analysis(self,
                          predictions: np.ndarray,
                          actual: np.ndarray,
                          save_name: str = "error_analysis.png") -> None:
        """
        Plot comprehensive error analysis
        
        Args:
            predictions: Predicted values
            actual: Actual values
            save_name: Filename to save the plot
        """
        errors = predictions - actual
        relative_errors = errors / (actual + 1e-8) * 100
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Error distribution
        ax1.hist(errors, bins=50, alpha=0.7, color=self.colors['primary'], density=True)
        ax1.axvline(np.mean(errors), color=self.colors['error'], linestyle='--', 
                   label=f'Mean: {np.mean(errors):.2f}')
        ax1.axvline(np.median(errors), color=self.colors['warning'], linestyle='--', 
                   label=f'Median: {np.median(errors):.2f}')
        ax1.set_xlabel('Prediction Error (N)')
        ax1.set_ylabel('Density')
        ax1.set_title('Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Relative error distribution
        ax2.hist(relative_errors, bins=50, alpha=0.7, color=self.colors['secondary'], density=True)
        ax2.axvline(np.mean(relative_errors), color=self.colors['error'], linestyle='--', 
                   label=f'Mean: {np.mean(relative_errors):.2f}%')
        ax2.axvline(np.median(relative_errors), color=self.colors['warning'], linestyle='--', 
                   label=f'Median: {np.median(relative_errors):.2f}%')
        ax2.set_xlabel('Relative Error (%)')
        ax2.set_ylabel('Density')
        ax2.set_title('Relative Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Error vs actual values
        ax3.scatter(actual, errors, alpha=0.6, color=self.colors['tertiary'], s=20)
        ax3.axhline(0, color='red', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Actual Force (N)')
        ax3.set_ylabel('Prediction Error (N)')
        ax3.set_title('Error vs Actual Values')
        ax3.grid(True, alpha=0.3)
        
        # QQ plot for normality check
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Error Normality)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_performance_metrics(self,
                               metrics: Dict[str, float],
                               save_name: str = "performance_metrics.png") -> None:
        """
        Plot performance metrics as a dashboard
        
        Args:
            metrics: Dictionary of performance metrics
            save_name: Filename to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Metric names and values
        metric_names = ['RMSE', 'MAE', 'R²', 'MAPE (%)']
        metric_values = [metrics.get('rmse', 0), metrics.get('mae', 0), 
                        metrics.get('r2', 0), metrics.get('mape', 0)]
        
        # Bar chart of main metrics
        bars = ax1.bar(metric_names, metric_values, 
                      color=[self.colors['primary'], self.colors['secondary'], 
                            self.colors['success'], self.colors['warning']])
        ax1.set_title('Model Performance Metrics', fontsize=14)
        ax1.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(metric_values),
                    f'{value:.4f}', ha='center', va='bottom', fontsize=11)
        
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Model efficiency metrics
        efficiency_metrics = ['Inference Time (ms)', 'Model Size (MB)', 'Parameters (K)']
        efficiency_values = [metrics.get('inference_time', 0), 
                           metrics.get('model_size_mb', 0),
                           metrics.get('num_parameters', 0) / 1000]
        
        ax2.bar(efficiency_metrics, efficiency_values, 
               color=[self.colors['info'], self.colors['quaternary'], self.colors['dark']])
        ax2.set_title('Model Efficiency Metrics', fontsize=14)
        ax2.set_ylabel('Value')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, value in enumerate(efficiency_values):
            ax2.text(i, value + 0.01*max(efficiency_values), f'{value:.2f}', 
                    ha='center', va='bottom', fontsize=11)
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Gauge chart for R²
        r2_value = metrics.get('r2', 0)
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax3.plot(theta, r, 'k-', linewidth=3)
        
        # Color code the gauge based on R² value
        if r2_value >= 0.9:
            gauge_color = self.colors['success']
        elif r2_value >= 0.8:
            gauge_color = self.colors['warning']
        else:
            gauge_color = self.colors['error']
        
        gauge_theta = np.linspace(0, r2_value * np.pi, 50)
        ax3.plot(gauge_theta, np.ones_like(gauge_theta), color=gauge_color, linewidth=8)
        
        ax3.set_xlim(0, np.pi)
        ax3.set_ylim(0, 1.2)
        ax3.set_title(f'R² Score: {r2_value:.4f}', fontsize=14)
        ax3.axis('off')
        
        # Summary text
        summary_text = f"""
        Model Performance Summary:
        
        Accuracy Metrics:
        • RMSE: {metrics.get('rmse', 0):.4f} N
        • MAE: {metrics.get('mae', 0):.4f} N
        • R²: {metrics.get('r2', 0):.4f}
        • MAPE: {metrics.get('mape', 0):.2f}%
        
        Efficiency Metrics:
        • Inference: {metrics.get('inference_time', 0):.2f} ms
        • Model Size: {metrics.get('model_size_mb', 0):.2f} MB
        • Parameters: {metrics.get('num_parameters', 0):,}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self,
                                   results: Dict[str, Any],
                                   save_name: str = "interactive_dashboard.html") -> None:
        """
        Create an interactive dashboard using Plotly
        
        Args:
            results: Complete results dictionary
            save_name: Filename to save the HTML dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Predictions vs Actual', 'Training History', 
                          'Feature Importance', 'Error Distribution',
                          'Hysteresis Loop', 'Attention Heatmap'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add plots to dashboard
        predictions = results.get('predictions_original', [])
        actual = results.get('targets_original', [])
        
        if len(predictions) > 0 and len(actual) > 0:
            # Predictions vs Actual
            fig.add_trace(
                go.Scatter(x=actual, y=predictions, mode='markers',
                          name='Predictions', marker=dict(color='blue', size=4)),
                row=1, col=1
            )
            
            # Perfect prediction line
            min_val, max_val = min(actual.min(), predictions.min()), max(actual.max(), predictions.max())
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                          mode='lines', name='Perfect Prediction', 
                          line=dict(color='red', dash='dash')),
                row=1, col=1
            )
        
        # Training history (if available)
        if 'history' in results:
            history = results['history']
            epochs = list(range(1, len(history['train_loss']) + 1))
            
            fig.add_trace(
                go.Scatter(x=epochs, y=history['train_loss'], 
                          mode='lines', name='Training Loss'),
                row=1, col=2
            )
            
            if 'val_loss' in history:
                valid_val_loss = [x for x in history['val_loss'] if not np.isnan(x)]
                valid_epochs = [i+1 for i, x in enumerate(history['val_loss']) if not np.isnan(x)]
                fig.add_trace(
                    go.Scatter(x=valid_epochs, y=valid_val_loss, 
                              mode='lines', name='Validation Loss'),
                    row=1, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="MR Damper Force Prediction Dashboard",
            showlegend=True
        )
        
        # Save interactive dashboard
        pyo.plot(fig, filename=str(self.save_dir / save_name), auto_open=False)
        print(f"Interactive dashboard saved to: {self.save_dir / save_name}")


def main():
    """Test the visualization module"""
    
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Sample data
    time = np.linspace(0, 10, n_samples)
    displacement = 0.1 * np.sin(2 * np.pi * time) + 0.02 * np.random.randn(n_samples)
    velocity = np.gradient(displacement)
    force_actual = 100 * displacement + 50 * velocity + 10 * np.random.randn(n_samples)
    force_predicted = force_actual + 5 * np.random.randn(n_samples)
    
    # Sample attention weights
    feature_attention = np.random.rand(20, 20, 4)  # 20 samples, 20 timesteps, 4 features
    temporal_attention = np.random.rand(20, 20)    # 20 samples, 20 timesteps
    
    # Feature importance
    feature_importance = {'A': 0.35, 'D': 0.25, 'Y': 0.15, 'V': 0.25}
    
    # Training history
    history = {
        'train_loss': [1.0 - 0.05*i + 0.02*np.random.randn() for i in range(50)],
        'val_loss': [1.1 - 0.04*i + 0.03*np.random.randn() for i in range(50)],
        'learning_rate': [0.001 * (0.95**i) for i in range(50)],
        'epoch_time': [2.0 + 0.1*np.random.randn() for _ in range(50)]
    }
    
    # Performance metrics
    metrics = {
        'rmse': 15.2,
        'mae': 12.1,
        'r2': 0.92,
        'mape': 8.5,
        'inference_time': 2.3,
        'model_size_mb': 1.2,
        'num_parameters': 245000
    }
    
    # Test visualizer
    visualizer = MRDamperVisualizer()
    
    print("Testing visualization functions...")
    
    visualizer.plot_predictions_vs_actual(force_predicted, force_actual)
    visualizer.plot_hysteresis_loops(displacement, force_actual, force_predicted)
    visualizer.plot_force_velocity_loops(velocity, force_actual, force_predicted)
    visualizer.plot_attention_heatmaps(feature_attention, temporal_attention, 
                                     ['A', 'D', 'Y', 'V'], num_samples=3)
    visualizer.plot_feature_importance(feature_importance)
    visualizer.plot_training_history(history)
    visualizer.plot_error_analysis(force_predicted, force_actual)
    visualizer.plot_performance_metrics(metrics)
    
    print("All visualizations completed successfully!")


if __name__ == "__main__":
    main()