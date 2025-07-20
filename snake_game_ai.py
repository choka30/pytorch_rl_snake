import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
#font = pygame.font.Font('/home/dhuencho/dev_py/0001_start_pytorch/utils/0001_RL_snake/arial.ttf', 25)
font = pygame.font.SysFont('arial', 25)


# reset 

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 80

class SnakeGameAI:
    
    def __init__(self, w=840, h=680):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        # Track previous head position for distance-based reward calculation
        # This enables potential-based reward shaping (Ng et al., 1999)
        self.prev_head = self.head
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Store previous head position before moving for reward calculation
        self.prev_head = self.head
            
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        game_over = False
        reward = 0
        
        # Enhanced reward calculation using multiple factors
        reward = self._calculate_enhanced_reward()
        
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            # Severe penalty for collision/death to discourage risky behavior
            # Increased from -10 to -15 based on reward balancing principles
            reward = -15
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            # Major positive reward for achieving the primary objective
            # Increased from +10 to +20 to emphasize food collection
            reward += 20
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def _calculate_enhanced_reward(self):
        """
        Enhanced reward function implementing multiple reward components:
        1. Distance-based reward shaping (Ng et al., 1999)
        2. Survival bonus for longevity
        3. Proximity awareness to encourage food-seeking behavior
        
        This dense reward structure provides more frequent feedback to the agent
        compared to the original sparse reward system (only food/death rewards).
        """
        reward = 0
        
        # 1. Distance-based reward shaping using Manhattan distance
        # Manhattan distance is computationally efficient and works well for grid-based games
        prev_distance = abs(self.prev_head.x - self.food.x) + abs(self.prev_head.y - self.food.y)
        current_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        # Potential-based reward shaping: R'(s,a,s') = R(s,a,s') + γΦ(s') - Φ(s)
        # Where Φ(s) = -distance_to_food, ensuring policy invariance (Ng et al., 1999)
        if current_distance < prev_distance:
            # Moving closer to food - positive reinforcement
            reward += 1.0
        elif current_distance > prev_distance:
            # Moving away from food - mild negative reinforcement
            # Kept small to avoid overly constraining exploration
            reward -= 0.5
        # If distance unchanged (current_distance == prev_distance), reward = 0
        
        # 2. Small survival bonus to encourage longevity and exploration
        # This prevents the agent from getting stuck in local minima
        # Small value (0.1) ensures it doesn't override other reward signals
        reward += 0.1
        
        # 3. Proximity bonus - additional reward when very close to food
        # Encourages the final approach to food when within striking distance
        # Grid size is BLOCK_SIZE (20), so distance <= 40 means within 2 blocks
        if current_distance <= 2 * BLOCK_SIZE:
            # Exponential bonus based on proximity (closer = higher reward)
            proximity_bonus = 2.0 * (2 * BLOCK_SIZE - current_distance) / (2 * BLOCK_SIZE)
            reward += proximity_bonus
        
        # 4. Efficiency bonus - reward shorter paths to food
        # Normalized by maximum possible distance to keep rewards bounded
        max_distance = self.w + self.h  # Maximum Manhattan distance in the game
        efficiency_bonus = 0.5 * (1.0 - current_distance / max_distance)
        reward += efficiency_bonus
        
        return reward
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # [0, 0, 1]
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            


