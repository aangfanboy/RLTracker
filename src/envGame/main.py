import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import threading
import time

from queue import Queue

import os
import sys
import math

# Add the project root to Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils.math import calculateAttitudeMatrixFromQuaternion, calculateQuaternionFromAttitudeMatrix, quaternionMultiply, floatMatrix
from core.missile import Missile
from core.target import Target

class Enviroment:
    def resetMap(self):
        self.map = self.create_3D_map(self.map.shape[0], self.map.shape[1], numMountains=self.numMountains, maxHeight=self.maxHeight)

        self.update_queue.put(1)  # Request visualization update

    def __init__(self, xDim: int, yDim: int, numMountains: int = 10, maxHeight: int = 300, missile: Missile = None, target: Target = None, visualize: bool = True):
        self.missile = missile
        self.target = target
        self.visualize = visualize
        self.numMountains = numMountains
        self.maxHeight = maxHeight
        self.FPS = 20
        self.fig = None
        self.running = False
        self.working = True
        self.lock = threading.Lock()
        self.update_queue: Queue[int] = Queue()
        self.quaternion_queue: Queue[floatMatrix] = Queue()
        self.quaternion: floatMatrix = self.missile.quaternions

        self.map: floatMatrix = self.create_3D_map(xDim, yDim, numMountains=numMountains, maxHeight=maxHeight)
        self.point_coordinate: floatMatrix = self.missile.position
        self.target_coordinate: floatMatrix = self.target.position
        self.quiver_artists = []   

    def check_in_bounds(self, coordinate: floatMatrix) -> bool:
        x, y, z = coordinate.flatten()
        if 0 <= x < self.map.shape[0] and 0 <= y < self.map.shape[1] and z >= 0:
            return True
        return False
    
    def check_collision_with_terrain(self, coordinate: floatMatrix) -> bool:
        """Check if the given coordinate collides with the terrain"""
        x, y, z = coordinate.flatten()
        return z <= self.map[int(x), int(y)]

    def check_collision_with_target(self, distance_threshold: float = 2.0) -> bool:
        """Check if the given coordinate collides with the target"""
        return bool(np.linalg.norm(self.target.position - self.missile.position) <= distance_threshold)

    def create_3D_map(self, xDim: int, yDim: int, numMountains: int = 5, maxHeight: int = 100) -> floatMatrix:
        """Create a simple 3D map with random mountains"""
        map: floatMatrix = np.zeros((xDim, yDim))
        x = np.arange(xDim)
        y = np.arange(yDim)
        x, y = np.meshgrid(x, y)

        for _ in range(numMountains):
            # Random center
            cx = np.random.randint(0, xDim)
            cy = np.random.randint(0, yDim)
            height = np.random.randint(maxHeight // 2, maxHeight)
            width = np.random.randint(maxHeight // 10, maxHeight // 5)
            
            # Create a Gaussian mountain with O(n)
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            map += height * np.exp(-(dist ** 2) / (2 * (width ** 2)))
            # --- IGNORE ---

        return map
    
    def set_ax_right(self, att_matrix: floatMatrix):
        self.ax_right.set_title(f'Quaternion: {self.quaternion.flatten()}')
        # Re-draw body frame with new orientation
        x_dir = att_matrix @ np.array([[1], [0], [0]])
        y_dir = att_matrix @ np.array([[0], [1], [0]])
        z_dir = att_matrix @ np.array([[0], [0], [1]])
        self.quiver_artists.append(self.ax_right.quiver(0, 0, 0, x_dir[0,0], x_dir[1,0], x_dir[2,0], color='r', label='X (Tip)'))
        self.quiver_artists.append(self.ax_right.quiver(0, 0, 0, y_dir[0,0], y_dir[1,0], y_dir[2,0], color='g', label='Y (Right)'))
        self.quiver_artists.append(self.ax_right.quiver(0, 0, 0, z_dir[0,0], z_dir[1,0], z_dir[2,0], color='b', label='Z (Down)'))
        self.update_queue.put(1)  # Request visualization update
    
    def init_plot(self):
        plt.switch_backend('TkAgg')

        # Create a figure with two subplots side-by-side, left larger than right.
        # Left: 3D environment surface. Right: placeholder 2D axis for additional plots.
        self.fig = plt.figure(figsize=(14, 6))
        # Use GridSpec to control the relative width of the two subplots (left wider)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 2, width_ratios=[3, 1], figure=self.fig)
        self.ax_left = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax_right = self.fig.add_subplot(gs[0, 1], projection='3d')

        # Configure left (3D) axis
        self.ax_left.set_xlabel('X axis')
        self.ax_left.set_ylabel('Y axis')
        self.ax_left.set_zlabel('Z axis')
        self.ax_left.set_title('3D Environment Map')
        # make sure z starts at 0, and x y z axes are in correct ratio
        self.ax_left.set_zlim(0, max(500, np.max(self.map) + 10))
        self.ax_left.set_box_aspect([1, 1, 0.5])  # Aspect ratio 1:1:0.5 for x:y:z
        self.ax_left.view_init(elev=30, azim=45)

        x = np.arange(self.map.shape[0])
        y = np.arange(self.map.shape[1])
        x, y = np.meshgrid(x, y)
        z = self.map.T  # Transpose to align axes correctly

        # Plot surface on left axis
        self.surf = self.ax_left.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.7)
        self.fig.colorbar(self.surf, ax=self.ax_left, shrink=0.5, aspect=5)

        # Optional: legend/grid for left
        try:
            self.ax_left.legend()
        except Exception:
            # If nothing to legend, ignore
            pass
        self.ax_left.grid(True)

        # Add point coordinate marker
        self.point_marker = self.ax_left.scatter(self.point_coordinate[0], self.point_coordinate[1], self.point_coordinate[2], color='g', s=75, label='Missile')
        self.target_marker = self.ax_left.scatter(self.target_coordinate[0], self.target_coordinate[1], self.target_coordinate[2], color='r', s=75, label='Target')

        # Configure right (2D) axis - placeholder for user content
        self.ax_right.set_xlabel('X axis')
        self.ax_right.set_ylabel('Y axis')
        self.ax_right.set_zlabel('Z axis')
        self.ax_right.set_zlim(-2, 2)
        self.ax_right.set_xlim(-2, 2)
        self.ax_right.set_ylim(-2, 2)

        self.ax_right.set_box_aspect([1, 1, 0.5])
        self.ax_right.view_init(elev=30, azim=45)
        self.ax_right.grid(True)

        self.set_ax_right(np.eye(3))

        self.ax_right.legend()

    def update_visualization(self):
        """Update the visualization with current data"""
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            return False
            
        try:
            # Redraw
            # Update point marker position
            self.point_marker._offsets3d = (np.array([self.point_coordinate[0]]), 
                                            np.array([self.point_coordinate[1]]),
                                            np.array([self.point_coordinate[2]]))
            
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            
            return True
            
        except Exception as e:
            print(f"Visualization update error: {e}")
            return False
    
    def updater_thread(self):
        while self.working == True:
            time.sleep(0.1)  # Wait until the visualization thread is ready

        while self.running:
            try:                    
                self.update_queue.put(1)

                time.sleep(0.01)

            except Exception as e:
                print(f"Rotation thread error: {e}")
                break

    def quaternion_thread(self):
        while self.working == True:
            time.sleep(0.1)  # Wait until the visualization thread is ready

        while self.running:
            try:
                if not self.quaternion_queue.empty():
                    self.quaternion = self.quaternion_queue.get()
                    att_matrix = calculateAttitudeMatrixFromQuaternion(self.quaternion)
                    print(f"Received Quaternion:\n{self.quaternion.flatten()}")

                    with self.lock:
                        for artist in self.quiver_artists:
                            artist.remove()
                        self.quiver_artists.clear()
                        self.set_ax_right(att_matrix)

                time.sleep(0.01)
                
            except Exception as e:
                print(f"Quaternion thread error: {e}")
                break

    def map_thread(self):
        # Initialize plot in this thread
        self.init_plot()
        self.working = False
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)

        print("Visualization thread started")
        
        last_update = 0
        update_interval = 0.05  # 20 FPS
        
        while self.running:
            try:
                # Check for update requests
                if not self.update_queue.empty():
                    self.update_queue.get()
                    current_time = time.time()
                    if current_time - last_update >= update_interval:
                        self.update_visualization()
                        last_update = current_time
                
                # Check if window is still open
                if not plt.fignum_exists(self.fig.number):
                    print("Plot window closed")
                    self.running = False
                    break
                    
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Visualization thread error: {e}")
                break
        
        plt.ioff()
    
    def run(self):
        if self.running:
            print("Visualization is already running!")
            return
        
        self.running = True

        if self.visualize:
            self.threadVisT = threading.Thread(target=self.map_thread)
            self.threadVisT.daemon = True
            self.threadVisT.start()

            self.threadUpdaterT = threading.Thread(target=self.updater_thread)
            self.threadUpdaterT.daemon = True
            self.threadUpdaterT.start()

            self.threadQuaternionT = threading.Thread(target=self.quaternion_thread)
            self.threadQuaternionT.daemon = True
            self.threadQuaternionT.start()

    def stop(self):
        self.running = False
        time.sleep(1)  # Give threads time to exit
        plt.close('all')

if __name__ == "__main__":
    env: Enviroment = Enviroment(500, 500)
    env.run()

    new_quat: floatMatrix = np.array([[0.0], [math.cos(math.pi/4)], [0.0], [math.cos(math.pi/4)]]) 

    time.sleep(10)
    env.quaternion_queue.put(new_quat)
    env.point_coordinate = np.array([100.0, 200.0, env.map[100,200]+10])

    time.sleep(100)  # Let it run for a bit
    env.stop()