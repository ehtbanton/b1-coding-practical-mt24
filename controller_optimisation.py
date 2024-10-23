import numpy as np
from itertools import product
from typing import Tuple, List
import matplotlib.pyplot as plt
from uuv_mission.dynamic import *
from uuv_mission.terrain import *
from uuv_mission.controller import *

class ControllerOptimizer:
    def __init__(
        self,
        mission: Mission,
        kp_range: Tuple[float, float, int] = (0.0, 1.0, 1000),
        kd_range: Tuple[float, float, int] = (0.0, 1.0, 1000)
    ):
        """
        Initialize the optimizer with parameter ranges to search.
        
        Args:
            mission: Mission object containing reference trajectory
            kp_range: Tuple of (min, max, num_points) for Kp search
            kd_range: Tuple of (min, max, num_points) for Kd search
        """
        self.mission = mission
        self.kp_values = np.linspace(*kp_range)
        self.kd_values = np.linspace(*kd_range)
        self.results = []
        
    def compute_tracking_error(self, trajectory: np.ndarray) -> float:
        """Compute mean squared error between reference and actual trajectory."""
        reference = self.mission.reference
        actual = trajectory[:, 1]  # y-positions from trajectory
        return np.mean((reference - actual) ** 2)
    
    def optimize(self, noise_variance: float = 0.5, num_trials: int = 3) -> Tuple[float, float, float]:
        """
        Find optimal Kp and Kd values by grid search.
        
        Args:
            noise_variance: Variance of random disturbances
            num_trials: Number of trials to average over for each parameter combination
            
        Returns:
            Tuple of (optimal_kp, optimal_kd, best_error)
        """
        best_error = float('inf')
        optimal_kp = optimal_kd = None
        
        # Try each combination of Kp and Kd
        for kp, kd in product(self.kp_values, self.kd_values):
            total_error = 0
            
            # Run multiple trials to account for random disturbances
            for _ in range(num_trials):
                # Create fresh instances for each trial
                submarine = Submarine()
                controller = Controller(Kp=kp, Kd=kd)
                closed_loop = ClosedLoop(submarine, controller)
                
                # Simulate with current parameters
                trajectory = closed_loop.simulate_with_random_disturbances(
                    self.mission, 
                    variance=noise_variance
                )
                
                total_error += self.compute_tracking_error(trajectory.position)
            
            avg_error = total_error / num_trials
            self.results.append((kp, kd, avg_error))
            
            if avg_error < best_error:
                best_error = avg_error
                optimal_kp = kp
                optimal_kd = kd
                
        return optimal_kp, optimal_kd, best_error
    
    def plot_error_surface(self):
        """Create a 3D surface plot of the error landscape."""
        if not self.results:
            raise ValueError("Must run optimize() before plotting")
            
        kp_vals, kd_vals, errors = zip(*self.results)
        
        # Create meshgrid for surface plot
        KP, KD = np.meshgrid(self.kp_values, self.kd_values)
        Z = np.zeros_like(KP)
        
        # Reshape errors into 2D array
        for i, kp in enumerate(self.kp_values):
            for j, kd in enumerate(self.kd_values):
                idx = self.results.index((kp, kd, 
                    next(err for k, d, err in self.results if k == kp and d == kd)))
                Z[j, i] = self.results[idx][2]
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(KP, KD, Z, cmap='viridis')
        
        ax.set_xlabel('Kp')
        ax.set_ylabel('Kd')
        ax.set_zlabel('MSE')
        plt.colorbar(surface)
        plt.title('Error Surface for PD Controller Parameters')
        
        return fig

def find_optimal_gains(
    mission: Mission,
    kp_range: Tuple[float, float, int] = (0.0, 1.0, 1000),
    kd_range: Tuple[float, float, int] = (0.0, 1.0, 1000),
    noise_variance: float = 0.5,
    num_trials: int = 3
) -> Tuple[float, float, float]:
    """
    Convenience function to find optimal gains.
    
    Returns:
        Tuple of (optimal_kp, optimal_kd, best_error)
    """
    optimizer = ControllerOptimizer(mission, kp_range, kd_range)
    return optimizer.optimize(noise_variance, num_trials)






###########################################################
################  PERFORMING OPTIMISATION  ################
###########################################################

mission = Mission.from_csv('data/mission.csv')
# Create parameter optimizer
optimizer = ControllerOptimizer(
    mission,
    kp_range=(0.05, 0.3, 10),  # Will try 10 values of Kp between 0.05 and 0.3
    kd_range=(0.3, 0.8, 10)    # Will try 10 values of Kd between 0.3 and 0.8
)

# Find optimal parameters
optimal_kp, optimal_kd, best_error = optimizer.optimize(
    noise_variance=0.5,
    num_trials=3  # Number of trials to average over for each parameter combination
)

print(f"Optimal gains: Kp={optimal_kp:.3f}, Kd={optimal_kd:.3f}")
print(f"Best MSE: {best_error:.3f}")

# Visualize error surface
optimizer.plot_error_surface()
plt.show()

# Test optimal controller
submarine = Submarine()
controller = Controller(Kp=optimal_kp, Kd=optimal_kd)
closed_loop = ClosedLoop(submarine, controller)
trajectory = closed_loop.simulate_with_random_disturbances(mission)
trajectory.plot_completed_mission(mission)