import pygame
import random
import numpy as np
import time

pygame.init()

# Constants
width, height = 480, 480
block_size = 20
clock = pygame.time.Clock()
black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)
red = (213, 50, 80)
blue = (50, 153, 213)

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake AI - Enhanced")

font = pygame.font.SysFont("Arial", 25)

UP = (0, -block_size)
DOWN = (0, block_size)
LEFT = (-block_size, 0)
RIGHT = (block_size, 0)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]
ACTIONS = ['STRAIGHT', 'RIGHT', 'LEFT']


class Snake:
    def __init__(self):
        self.reset()

    def reset(self):
        self.body = [(width // 2, height // 2)]
        self.direction = random.choice(DIRECTIONS)
        self.length = 1

    def move(self, action):
        idx = DIRECTIONS.index(self.direction)
        if action == 'RIGHT':
            self.direction = DIRECTIONS[(idx + 1) % 4]
        elif action == 'LEFT':
            self.direction = DIRECTIONS[(idx - 1) % 4]
        # else STRAIGHT = no change

        new_head = (self.body[0][0] + self.direction[0],
                    self.body[0][1] + self.direction[1])
        self.body = [new_head] + self.body[:-1]

    def grow(self):
        tail = self.body[-1]
        self.body.append(tail)

    def collide(self):
        head = self.body[0]
        x, y = head
        return (x < 0 or x >= width or y < 0 or y >= height or head in self.body[1:])

    def get_state(self, food_pos):
        head_x, head_y = self.body[0]
        food_x, food_y = food_pos
        dir_x, dir_y = self.direction

        def danger_in_dir(dx, dy):
            next_pos = (head_x + dx, head_y + dy)
            return (next_pos[0] < 0 or next_pos[0] >= width or
                    next_pos[1] < 0 or next_pos[1] >= height or next_pos in self.body)

        left_dir = DIRECTIONS[(DIRECTIONS.index(self.direction) - 1) % 4]
        right_dir = DIRECTIONS[(DIRECTIONS.index(self.direction) + 1) % 4]

        state = (
            danger_in_dir(*self.direction),     # danger ahead
            danger_in_dir(*right_dir),          # danger right
            danger_in_dir(*left_dir),           # danger left
            dir_x == -block_size,  # moving left
            dir_x == block_size,   # moving right
            dir_y == -block_size,  # moving up
            dir_y == block_size,   # moving down
            food_x < head_x,  # food left
            food_x > head_x,  # food right
            food_y < head_y,  # food up
            food_y > head_y   # food down
        )
        return tuple(int(x) for x in state)


class Food:
    def __init__(self):
        self.position = (0, 0)
        self.spawn([])

    def spawn(self, snake_body):
        while True:
            x = random.randrange(0, width, block_size)
            y = random.randrange(0, height, block_size)
            if (x, y) not in snake_body:
                self.position = (x, y)
                break


class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return max(ACTIONS, key=lambda a: self.get_q(state, a))

    def learn(self, state, action, reward, next_state):
        max_q_next = max([self.get_q(next_state, a) for a in ACTIONS])
        current_q = self.get_q(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * max_q_next - current_q)
        self.q_table[(state, action)] = new_q

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def draw(snake, food, score, deaths):
    screen.fill(blue)
    for segment in snake.body:
        pygame.draw.rect(screen, green, (*segment, block_size, block_size))
    pygame.draw.rect(screen, red, (*food.position, block_size, block_size))

    score_text = font.render(f"Score: {score}", True, white)
    death_text = font.render(f"Deaths: {deaths}", True, white)
    screen.blit(score_text, (10, 10))
    screen.blit(death_text, (width - 150, 10))
    pygame.display.flip()


def main():
    agent = QLearningAgent()
    snake = Snake()
    food = Food()
    score = 0
    deaths = 0

    while True:
        state = snake.get_state(food.position)
        action = agent.choose_action(state)

        snake.move(action)

        reward = 0
        game_over = False

        if snake.collide():
            reward = -100
            game_over = True
        elif snake.body[0] == food.position:
            reward = 50
            snake.grow()
            food.spawn(snake.body)
            score += 1
        else:
            reward = -1

        next_state = snake.get_state(food.position)
        agent.learn(state, action, reward, next_state)

        draw(snake, food, score, deaths)

        if game_over:
            snake.reset()
            food.spawn(snake.body)
            deaths += 1
            score = 0
            agent.decay()
            time.sleep(0.2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        clock.tick(90) #change for faster training

if __name__ == "__main__":
    main()
