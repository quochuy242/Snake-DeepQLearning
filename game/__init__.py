import random
from collections import namedtuple
from enum import Enum

import numpy as np
import pygame

# Initialize PyGame
pygame.init()

# Constant
## Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

## Game dimensions
BLOCK_SIZE = 20
WIDTH = 800
HEIGHT = 600
SPEED = 40


## Direction
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Point class
Point = namedtuple("Point", ["x", "y"])


class SnakeGame:
    def __init__(self, width=WIDTH, height=HEIGHT, block_size=BLOCK_SIZE) -> None:
        self.width = width
        self.height = height
        self.block_size = block_size

        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Game - Deep Q Learning")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self) -> None:
        # Initialize game state
        self.head: Point = Point(self.width // 2, self.height // 2)
        self.direction = Direction.RIGHT

        self.snake = [
            self.head,
            Point(self.head.x - self.block_size, self.head.y),
            Point(self.head.x - 2 * self.block_size, self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self) -> None:
        foodx = (
            round(random.randrange(0, self.width - self.block_size) / self.block_size)
            * self.block_size
        )
        foody = (
            round(random.randrange(0, self.height - self.block_size) / self.block_size)
            * self.block_size
        )
        self.food: Point = Point(foodx, foody)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move
        self._move(action)
        self.snake.insert(0, self.head)

        # Check if game over
        reward = 0
        game_over = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return game_over, self.score

        # Place new food or move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)  # adjust for faster/slower game speed

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # Hit boundary
        if (
            pt.x > self.width - self.block_size
            or pt.x < 0
            or pt.y > self.height - self.block_size
            or pt.y < 0
        ):
            return True

        # Hit itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(
                self.display,
                BLUE,
                pygame.Rect(pt.x, pt.y, self.block_size, self.block_size),
            )
            pygame.draw.rect(
                self.display, WHITE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )

        # TODO: Change shape of food from rectangle to circle
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size),
        )

        font = pygame.font.SysFont(None, 25)
        text = font.render(f"Score: {str(self.score)}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # Action: [straight, right, left]
        clock_wise = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Straight
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4  # Left turn
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.UP:
            y -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.RIGHT:
            x += self.block_size

        self.head = Point(x, y)
