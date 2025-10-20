#!/usr/bin/env python3
"""
Script to read TensorBoard event files from general_tensorboard directory
and plot average_reward data.
"""
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Make project root importable when run directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    # Prefer tensorboard event accumulator if available
    from tensorboard.backend.event_processing import event_accumulator
    _HAS_EVENT_ACC = True
except ImportError:
    event_accumulator = None
    _HAS_EVENT_ACC = False


def find_event_files(tb_dir: str) -> List[str]:
    """Find all TensorBoard event files in the given directory."""
    if not os.path.isdir(tb_dir):
        return []
    files = []
    for fn in os.listdir(tb_dir):
        if fn.startswith("events.out"):
            files.append(os.path.join(tb_dir, fn))
    return sorted(files)


def load_average_reward_from_tensorboard(tb_dir: str, tag: str = 'average_reward') -> List[Tuple[int, float]]:
    """
    Load average_reward data from all TensorBoard event files in the directory.
    
    Args:
        tb_dir: Path to the TensorBoard directory containing event files
        tag: The tag name to search for (default: 'average_reward')
    
    Returns:
        List of (step, value) tuples sorted by step
    """
    result = []

    if _HAS_EVENT_ACC:
        # Use TensorBoard event accumulator (preferred method)
        event_files = find_event_files(tb_dir)
        
        for event_file in event_files:
            try:
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                
                # Check if the tag exists in scalars
                if tag in ea.Tags().get('scalars', []):
                    items = ea.Scalars(tag)
                    for item in items:
                        result.append((item.step, item.value))
                # Also check in tensors (in case it's stored as tensor)
                elif tag in ea.Tags().get('tensors', []):
                    items = ea.Tensors(tag)
                    for item in items:
                        step = item.step
                        val = None
                        if item.tensor_proto and item.tensor_proto.tensor_content:
                            try:
                                val = float(np.frombuffer(item.tensor_proto.tensor_content, dtype=np.float32)[0])
                            except Exception:
                                pass
                        if val is not None:
                            result.append((step, val))
            except Exception as e:
                print(f"Warning: Could not read {event_file}: {e}")
                continue
    else:
        # Fallback: use tensorflow.summary iterator
        try:
            import tensorflow as tf  # type: ignore
            from tensorflow.python.summary.summary_iterator import summary_iterator
        except ImportError:
            raise ImportError("Neither tensorboard nor tensorflow is available. Please install one of them.")

        event_files = find_event_files(tb_dir)
        
        for event_file in event_files:
            try:
                for e in summary_iterator(event_file):
                    if not hasattr(e, 'summary') or e.summary is None:
                        continue
                    step = int(getattr(e, 'step', 0))
                    for v in e.summary.value:
                        if v.tag == tag:
                            val = None
                            if v.HasField('simple_value'):
                                val = float(v.simple_value)
                            else:
                                try:
                                    val = float(np.frombuffer(v.tensor.tensor_content, dtype=np.float32)[0])
                                except Exception:
                                    pass
                            if val is not None:
                                result.append((step, val))
            except Exception as e:
                print(f"Warning: Could not read {event_file}: {e}")
                continue

    # Sort by step
    result.sort(key=lambda x: x[0])
    return result


def plot_average_reward(tb_dir: str = 'flightData/general_tensorboard', 
                        tag: str = 'reward/average_reward',
                        save_path: str = None,
                        figsize: tuple = (12, 6)):
    """
    Read all TensorBoard event files from general_tensorboard directory
    and create a matplotlib plot of average_reward.
    
    Args:
        tb_dir: Path to the TensorBoard directory (default: 'flightData/general_tensorboard')
        tag: The tag name to search for (default: 'average_reward')
        save_path: Optional path to save the plot image
        figsize: Figure size as (width, height) in inches
    
    Returns:
        matplotlib figure and axes objects
    """
    # Handle relative path from project root
    if not os.path.isabs(tb_dir):
        # Try relative to current file's parent directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tb_dir = os.path.join(base_dir, tb_dir)
    
    if not os.path.isdir(tb_dir):
        raise FileNotFoundError(f"TensorBoard directory not found: {tb_dir}")
    
    print(f"Reading TensorBoard events from: {tb_dir}")
    
    # Load the data
    data = load_average_reward_from_tensorboard(tb_dir, tag)
    
    if not data:
        raise ValueError(f"No data found for tag '{tag}' in {tb_dir}")
    
    # Separate steps and values
    steps, values = zip(*data)
    steps = np.array(steps)
    values = np.array(values)
    
    print(f"Found {len(data)} data points")
    print(f"Step range: {steps.min()} to {steps.max()}")
    print(f"Value range: {values.min():.4f} to {values.max():.4f}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the data
    ax.plot(steps, values, linewidth=1.5, alpha=0.7, label=tag)
    
    # Add smoothed curve if there are enough points
    if len(values) > 10:
        from scipy.ndimage import uniform_filter1d
        try:
            window_size = min(50, len(values) // 10)
            smoothed = uniform_filter1d(values, size=window_size, mode='nearest')
            ax.plot(steps, smoothed, linewidth=2.5, color='red', alpha=0.8, label=f'{tag} (smoothed)')
        except ImportError:
            # If scipy is not available, use simple moving average
            window_size = min(50, len(values) // 10)
            smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='same')
            ax.plot(steps, smoothed, linewidth=2.5, color='red', alpha=0.8, label=f'{tag} (smoothed)')
    
    # Formatting
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Average Reward Over Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Add statistics text box
    stats_text = f'Min: {values.min():.4f}\nMax: {values.max():.4f}\nMean: {values.mean():.4f}\nStd: {values.std():.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig, ax


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Plot average reward from TensorBoard event files'
    )
    parser.add_argument(
        '--dir', 
        type=str, 
        default='flightData/general_tensorboard',
        help='Path to TensorBoard directory (default: flightData/general_tensorboard)'
    )
    parser.add_argument(
        '--tag',
        type=str,
        default='reward/average_reward',
        help='Tag name to plot (default: reward/average_reward)'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save the plot image (optional)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot (useful when only saving)'
    )
    
    args = parser.parse_args()
    
    try:
        fig, ax = plot_average_reward(
            tb_dir=args.dir,
            tag=args.tag,
            save_path=args.save
        )
        
        if not args.no_show:
            plt.show()
        else:
            plt.close(fig)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
