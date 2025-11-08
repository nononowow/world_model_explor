"""
Driving environment for autonomous driving world models.

A simplified driving environment for testing world models.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame


class SimpleDrivingEnv(gym.Env):
    """
    Simple 2D driving environment.
    
    A car drives on a road and must avoid obstacles and stay on the road.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None, width=800, height=600):
        """
        Args:
            render_mode: 'human' or 'rgb_array'
            width: Screen width
            height: Screen height
        """
        super(SimpleDrivingEnv, self).__init__()
        
        self.width = width
        self.height = height
        self.render_mode = render_mode
        
        # Action space: [steering, throttle, brake]
        # steering: [-1, 1] (left to right)
        # throttle: [0, 1] (no throttle to full throttle)
        # brake: [0, 1] (no brake to full brake)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: RGB image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8
        )
        
        # State
        self.car_x = width // 2
        self.car_y = height - 50
        self.car_angle = 0.0  # Angle in degrees
        self.car_speed = 0.0
        
        # Road parameters
        self.road_width = 200
        self.road_center = width // 2
        
        # Obstacles
        self.obstacles = []
        
        # Rendering
        self.screen = None
        self.clock = None
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset car state
        self.car_x = self.width // 2
        self.car_y = self.height - 50
        self.car_angle = 0.0
        self.car_speed = 0.0
        
        # Reset obstacles
        self.obstacles = []
        for i in range(3):
            self.obstacles.append({
                'x': np.random.randint(100, self.width - 100),
                'y': np.random.randint(0, self.height // 2),
                'size': 30
            })
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Step the environment.
        
        Args:
            action: [steering, throttle, brake]
        """
        steering, throttle, brake = action
        
        # Update car state
        # Steering affects angle
        self.car_angle += steering * 5.0  # Max 5 degrees per step
        self.car_angle = np.clip(self.car_angle, -45, 45)
        
        # Throttle and brake affect speed
        acceleration = throttle * 0.5 - brake * 1.0
        self.car_speed += acceleration
        self.car_speed = np.clip(self.car_speed, 0, 10)
        
        # Update position based on speed and angle
        angle_rad = np.radians(self.car_angle)
        self.car_x += self.car_speed * np.sin(angle_rad)
        self.car_y -= self.car_speed * np.cos(angle_rad)
        
        # Boundary checks
        self.car_x = np.clip(self.car_x, 0, self.width)
        self.car_y = np.clip(self.car_y, 0, self.height)
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination
        terminated = self._check_termination()
        truncated = False
        
        observation = self._get_observation()
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def _compute_reward(self):
        """Compute reward based on current state."""
        reward = 0.0
        
        # Reward for staying on road
        road_left = self.road_center - self.road_width // 2
        road_right = self.road_center + self.road_width // 2
        
        if road_left <= self.car_x <= road_right:
            reward += 0.1
        else:
            reward -= 0.5  # Penalty for going off road
        
        # Reward for forward progress
        reward += 0.01 * self.car_speed
        
        # Penalty for collisions
        for obstacle in self.obstacles:
            dist = np.sqrt((self.car_x - obstacle['x'])**2 + (self.car_y - obstacle['y'])**2)
            if dist < obstacle['size']:
                reward -= 10.0
        
        return reward
    
    def _check_termination(self):
        """Check if episode should terminate."""
        # Terminate if off road
        road_left = self.road_center - self.road_width // 2
        road_right = self.road_center + self.road_width // 2
        
        if not (road_left <= self.car_x <= road_right):
            return True
        
        # Terminate if collision
        for obstacle in self.obstacles:
            dist = np.sqrt((self.car_x - obstacle['x'])**2 + (self.car_y - obstacle['y'])**2)
            if dist < obstacle['size']:
                return True
        
        return False
    
    def _get_observation(self):
        """Get current observation (RGB image)."""
        if self.render_mode == 'rgb_array' or self.screen is not None:
            return self.render()
        else:
            # Return a simple observation (could be improved)
            obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Draw road
            road_left = self.road_center - self.road_width // 2
            road_right = self.road_center + self.road_width // 2
            obs[:, road_left:road_right, :] = [100, 100, 100]  # Gray road
            return obs
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == 'human':
                self.screen = pygame.display.set_mode((self.width, self.height))
            else:
                self.screen = pygame.Surface((self.width, self.height))
            pygame.display.set_caption('Simple Driving Environment')
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.screen.fill((50, 50, 50))  # Dark background
        
        # Draw road
        road_left = self.road_center - self.road_width // 2
        road_right = self.road_center + self.road_width // 2
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (road_left, 0, self.road_width, self.height))
        
        # Draw road markings
        center_line_y = 0
        while center_line_y < self.height:
            pygame.draw.rect(self.screen, (255, 255, 255),
                           (self.road_center - 2, center_line_y, 4, 20))
            center_line_y += 40
        
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.circle(self.screen, (255, 0, 0),
                             (int(obstacle['x']), int(obstacle['y'])),
                             obstacle['size'])
        
        # Draw car
        car_points = [
            (self.car_x, self.car_y - 10),
            (self.car_x - 5, self.car_y + 10),
            (self.car_x + 5, self.car_y + 10)
        ]
        # Rotate car points based on angle
        angle_rad = np.radians(self.car_angle)
        rotated_points = []
        for x, y in car_points:
            dx = x - self.car_x
            dy = y - self.car_y
            new_x = self.car_x + dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
            new_y = self.car_y + dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
            rotated_points.append((int(new_x), int(new_y)))
        pygame.draw.polygon(self.screen, (0, 0, 255), rotated_points)
        
        if self.render_mode == 'human':
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Close the environment."""
        if self.screen is not None:
            pygame.quit()

