import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from collections import defaultdict, deque
import torch

plt.ion()  # Turn on interactive mode

# Global figure instance to prevent creating multiple windows
# This ensures we reuse the same window across all plotting calls
_global_fig = None

class QValueTracker:
    """
    Enhanced tracking class for Q-value analysis during training.
    
    Tracks:
    1. Q-values by state representation
    2. Cumulative mean Q-values by action (straight, right, left)
    3. Q-value distributions and variance over time
    4. State-action value evolution
    
    Based on monitoring practices from:
    - Mnih et al., "Human-level control through deep reinforcement learning" (Nature 2015)
    - Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (AAAI 2016)
    """
    def __init__(self, window_size=1000):
        # Track Q-values for each action (0=straight, 1=right, 2=left)
        self.action_q_values = {0: deque(maxlen=window_size), 
                               1: deque(maxlen=window_size), 
                               2: deque(maxlen=window_size)}
        
        # Cumulative statistics for each action
        self.cumulative_q_means = {0: [], 1: [], 2: []}
        self.cumulative_q_counts = {0: 0, 1: 0, 2: 0}
        self.cumulative_q_sums = {0: 0.0, 1: 0.0, 2: 0.0}
        
        # State-based Q-value tracking (using state hash for memory efficiency)
        self.state_q_history = defaultdict(list)
        
        # Max Q-values per episode for stability monitoring
        self.max_q_per_episode = []
        self.mean_q_per_episode = []
        
        # Action selection distribution tracking
        self.action_distribution = {0: 0, 1: 0, 2: 0}
        
    def update_q_values(self, state, q_values, action_taken):
        """
        Update Q-value tracking with current state and predicted Q-values.
        
        Args:
            state: Current state vector (numpy array)
            q_values: Q-values for all actions from the model (torch tensor or numpy array)
            action_taken: Action index that was selected (0, 1, or 2)
        """
        # Convert to numpy if torch tensor
        if torch.is_tensor(q_values):
            q_vals = q_values.detach().cpu().numpy().flatten()
        else:
            q_vals = np.array(q_values).flatten()
            
        # Track Q-values for each action
        for action_idx in range(3):
            q_val = float(q_vals[action_idx])
            self.action_q_values[action_idx].append(q_val)
            
            # Update cumulative statistics
            self.cumulative_q_counts[action_idx] += 1
            self.cumulative_q_sums[action_idx] += q_val
            current_mean = self.cumulative_q_sums[action_idx] / self.cumulative_q_counts[action_idx]
            self.cumulative_q_means[action_idx].append(current_mean)
        
        # Track state-specific Q-values (use state hash to avoid memory explosion)
        state_hash = hash(tuple(state.flatten()[:5]))  # Use first 5 features for state representation
        self.state_q_history[state_hash].append(q_vals.copy())
        
        # Track action selection for exploration analysis
        self.action_distribution[action_taken] += 1
        
        # Episode-level statistics
        self.max_q_per_episode.append(np.max(q_vals))
        self.mean_q_per_episode.append(np.mean(q_vals))
    
    def get_recent_stats(self, window=100):
        """Get recent Q-value statistics for monitoring."""
        stats = {}
        for action in range(3):
            recent_q = list(self.action_q_values[action])[-window:]
            if recent_q:
                stats[f'action_{action}_mean'] = np.mean(recent_q)
                stats[f'action_{action}_std'] = np.std(recent_q)
        return stats

# Global tracker instance
q_tracker = QValueTracker()

def plot(scores, mean_scores):
    """
    Original plotting function maintained for backward compatibility.
    Enhanced with Q-value visualization.
    """
    global _global_fig
    
    # Create or reuse the same figure to prevent multiple windows
    if _global_fig is None:
        _global_fig = plt.figure(figsize=(15, 10))
    else:
        plt.figure(_global_fig.number)  # Switch to existing figure
        plt.clf()  # Clear the figure content
    
    display.clear_output(wait=True)
    display.display(_global_fig)
    
    # 1. Original score plot (top-left)
    plt.subplot(2, 3, 1)
    plt.title('Game Scores')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Scores', alpha=0.7)
    plt.plot(mean_scores, label='Mean Scores', linewidth=2)
    plt.ylim(ymin=0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text information
    if scores:
        plt.text(0.02, 0.98, f'Games: {len(scores)}', transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10, color='red')
        plt.text(0.02, 0.90, f'Mean Score: {mean_scores[-1]:.2f}', transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10, color='blue')
    
    # 2. Q-value evolution by action (top-middle)
    plt.subplot(2, 3, 2)
    plt.title('Cumulative Mean Q-Values by Action')
    plt.xlabel('Training Steps')
    plt.ylabel('Mean Q-Value')
    
    action_names = ['Straight', 'Right', 'Left']
    colors = ['blue', 'red', 'green']
    
    for action in range(3):
        if q_tracker.cumulative_q_means[action]:
            plt.plot(q_tracker.cumulative_q_means[action], 
                    label=f'{action_names[action]}', 
                    color=colors[action], linewidth=2)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Recent Q-value distribution (top-right)
    plt.subplot(2, 3, 3)
    plt.title('Recent Q-Value Distribution')
    plt.xlabel('Q-Value')
    plt.ylabel('Frequency')
    
    # Plot histogram of recent Q-values for each action
    recent_window = 200
    for action in range(3):
        recent_q = list(q_tracker.action_q_values[action])[-recent_window:]
        if len(recent_q) > 10:  # Only plot if we have enough data
            plt.hist(recent_q, alpha=0.6, bins=20, label=f'{action_names[action]}', 
                    color=colors[action])
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Max Q-values per episode (bottom-left)
    plt.subplot(2, 3, 4)
    plt.title('Max Q-Value Evolution')
    plt.xlabel('Training Steps')
    plt.ylabel('Max Q-Value')
    
    if q_tracker.max_q_per_episode:
        # Smooth the max Q-values with moving average
        window_size = min(50, len(q_tracker.max_q_per_episode) // 4)
        if window_size > 1:
            smoothed_max_q = np.convolve(q_tracker.max_q_per_episode, 
                                       np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(q_tracker.max_q_per_episode)), 
                    smoothed_max_q, label='Smoothed Max Q', color='purple', linewidth=2)
        
        plt.plot(q_tracker.max_q_per_episode, alpha=0.5, label='Raw Max Q', color='gray')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    
    # 5. Action selection distribution (bottom-middle)
    plt.subplot(2, 3, 5)
    plt.title('Action Selection Distribution')
    
    total_actions = sum(q_tracker.action_distribution.values())
    if total_actions > 0:
        action_percentages = [q_tracker.action_distribution[i] / total_actions * 100 
                            for i in range(3)]
        
        bars = plt.bar(action_names, action_percentages, color=colors, alpha=0.7)
        plt.ylabel('Percentage (%)')
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, action_percentages):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    
    # 6. Q-value variance over time (bottom-right)
    plt.subplot(2, 3, 6)
    plt.title('Q-Value Variance (Training Stability)')
    plt.xlabel('Training Steps')
    plt.ylabel('Q-Value Std Dev')
    
    # Calculate rolling standard deviation of Q-values
    window_size = 100
    if len(q_tracker.mean_q_per_episode) >= window_size:
        q_variance = []
        for i in range(window_size, len(q_tracker.mean_q_per_episode)):
            recent_q = q_tracker.mean_q_per_episode[i-window_size:i]
            q_variance.append(np.std(recent_q))
        
        plt.plot(range(window_size, len(q_tracker.mean_q_per_episode)), 
                q_variance, color='orange', linewidth=2, label='Q-Value Std Dev')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.pause(.1)

def plot_enhanced(scores, mean_scores, agent=None, current_state=None):
    """
    Enhanced plotting function that includes Q-value analysis.
    
    Args:
        scores: List of game scores
        mean_scores: List of mean scores
        agent: Agent object with model for Q-value prediction
        current_state: Current game state for real-time Q-value display
    """
    # Update Q-value tracking if agent and state provided
    if agent is not None and current_state is not None:
        with torch.no_grad():
            state_tensor = torch.tensor(current_state, dtype=torch.float).unsqueeze(0)
            q_values = agent.model(state_tensor)
            # Get the action that would be taken
            action = torch.argmax(q_values).item()
            q_tracker.update_q_values(current_state, q_values, action)
    
    # Use the enhanced plotting
    plot(scores, mean_scores)