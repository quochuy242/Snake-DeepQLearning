import random
from collections import namedtuple
from enum import Enum
from typing import Tuple

import numpy as np
import pygame

from utils.constant import (
    BLACK,
    BLOCK_SIZE,
    BLUE,
    HEIGHT,
    RED,
    SPEED,
    LIGHT_BLUE,
    WIDTH,
    WHITE,
    RADIUS,
    REWARD,
)

# Initialize PyGame
pygame.init()


# Direction
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
        border = self.block_size + RADIUS
        foodx = (
            round(random.randrange(border, self.width - border) / self.block_size)
            * self.block_size
        )
        foody = (
            round(random.randrange(border, self.height - border) / self.block_size)
            * self.block_size
        )
        self.food: Point = Point(foodx, foody)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action) -> Tuple[int, bool, int]:
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move
        self._move(action)
        self.snake.insert(0, self.head)

        # Initial reward and flag game_over
        reward = 0
        game_over = False

        # Check if game over
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -REWARD
            return reward, game_over, self.score

        # Place new food or move
        if self.head == self.food:
            self.score += 1
            reward = REWARD
            self._place_food()
        else:
            self.snake.pop()

        # Update UI and clock
        self._update_ui()

        # Adjust for faster/slower game speed
        self.clock.tick(SPEED)

        # Return game over and score
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
                self.display, LIGHT_BLUE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )

        # // TODO: Change shape of food from rectangle to circle
        pygame.draw.circle(self.display, RED, self.food, radius=RADIUS)

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
