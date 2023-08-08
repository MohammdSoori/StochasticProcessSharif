import numpy as np

def check_win_large_grid(board):
    # Define the winning length
    win_length = 10

    # Check rows
    for row in board:
        for i in range(len(row) - win_length + 1):
            if len(set(row[i:i+win_length])) == 1 and row[i] != 'E':
                return row[i]

    # Check columns
    for col in range(len(board[0])):
        column = [row[col] for row in board]
        for i in range(len(column) - win_length + 1):
            if len(set(column[i:i+win_length])) == 1 and column[i] != 'E':
                return column[i]

    # Check main diagonals (top-left to bottom-right)
    for i in range(len(board) - win_length + 1):
        for j in range(len(board[0]) - win_length + 1):
            diagonal = [board[i+k][j+k] for k in range(win_length)]
            if len(set(diagonal)) == 1 and diagonal[0] != 'E':
                return diagonal[0]

    # Check anti-diagonals (top-right to bottom-left)
    for i in range(len(board) - win_length + 1):
        for j in range(win_length - 1, len(board[0])):
            diagonal = [board[i+k][j-k] for k in range(win_length)]
            if len(set(diagonal)) == 1 and diagonal[0] != 'E':
                return diagonal[0]

    # If no winner is found and the board is full, the game is a draw
    if 'E' not in np.array(board).flatten():
        return 'Draw'

    # If no player has won and the board is not full, continue the game
    return 'Continue'


import numpy as np
import torch.nn.functional as F
def get_empty_cells(self):
    return [(i, j) for i in range(100) for j in range(100) if self.board[i, j] == 'E']

def encode_state(state):
    """
    Encode a tic-tac-toe state into a numeric array.
    'X' -> 1
    'O' -> -1
    'E' -> 0
    """
    encoded_state = np.where(state=='X', 1, np.where(state=='O', -1, 0))
    return encoded_state.reshape(1, 1, 100, 100)

class TicTacToe:
    def __init__(self):
        self.board = np.full((100, 100), 'E')
        self.current_player = 'X'

    def reset(self):
        self.board = np.full((100, 100), 'E')
        self.current_player = 'X'

    def perform_action(self, action):
        # action is a tuple (i, j) representing a cell on the board
        i, j = action
        if self.board[i, j] != 'E':
            return self.board, -1, False  # invalid action, return a penalty and continue the game


        self.board[i, j] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'

        winner = check_win_large_grid(self.board)
        if winner == 'X':
            return self.board, 1, True  # X wins, return positive reward and end the game
        elif winner == 'O':
            return self.board, -1, True  # O wins, return negative reward and end the game
        elif winner == 'Draw':
            return self.board, 0, True  # game is a draw, return no reward and end the game

        return self.board, 0, False  # game continues, return no reward

    def get_empty_cells(self):
        return [(i, j) for i in range(100) for j in range(100) if self.board[i, j] == 'E']


    def get_status(self):
        # Use the check_win_large_grid function provided in the question
        result = check_win_large_grid(self.board)
        if result == 'X':
            return 1  # X wins
        elif result == 'O':
            return -1  # O wins
        elif result == 'Draw':
            return 0  # Draw
        else:
            return None  # Game continues

    def get_state(self):
        # Return a numerical representation of the board
        state = np.where(self.board == 'X', 1, self.board)
        state = np.where(state == 'O', -1, state)
        state = np.where(state == 'E', 0, state)
        return state.astype(float).reshape(1, 1, 100, 100)


import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=4, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(100)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(100)))
        linear_input_size = convw * convh * 64
        self.fc = nn.Linear(linear_input_size, action_dim)  # Action dimension

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.fc(x.view(x.size(0), -1))

from collections import deque
import random

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        self.model = DQN(state_dim, action_dim, hidden_dim)
        self.target = DQN(state_dim, action_dim, hidden_dim)
        self.target.load_state_dict(self.model.state_dict())
        self.memory = ReplayMemory()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, state, epsilon, empty_cells):
        if random.random() < epsilon or len(empty_cells) == 0:
            return random.choice(empty_cells)
        else:
            with torch.no_grad():
                q_values = self.model(state)
                empty_cell_indices = [i*100 + j for i, j in empty_cells]
                empty_cell_q_values = q_values[0, 0, empty_cell_indices]
                return empty_cells[empty_cell_q_values.argmax()]


    def update(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        batch_state = torch.tensor([encode_state(state)[0] for state in batch_state], dtype=torch.float32)
        batch_action = torch.tensor([i*100 + j for i, j in batch_action], dtype=torch.int64).unsqueeze(1)
        batch_next_state = torch.tensor([encode_state(state)[0] for state in batch_next_state], dtype=torch.float32)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32)

        current_q_values = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze()
        next_q_values = self.target(batch_next_state).max(1)[0].detach()
        target_q_values = batch_reward + gamma * next_q_values

        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train(agent, environment, episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay):
    scores = []
    epsilon = epsilon_start

    for episode in range(episodes):
        state = environment.reset()
        done = False
        score = 0

        state = environment.reset()
        encoded_state = encode_state(state)

        while not done:
            action = agent.get_action(encoded_state, epsilon, environment.get_empty_cells())
            next_state, reward, done = environment.perform_action(action)
            encoded_next_state = encode_state(next_state)
            agent.memory.push((encoded_state, action, encoded_next_state, reward))
            agent.update(batch_size, gamma)
            state = next_state
            encoded_state = encoded_next_state
            score += reward

        scores.append(score)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode: {episode}, Epsilon: {epsilon:.2f}, Average Score: {avg_score:.2f}")

        if episode % 10 == 0:
            agent.target.load_state_dict(agent.model.state_dict())

    return scores



class RandomAgent:
    def get_action(self, state, empty_cells):
        return random.choice(empty_cells)

def compete(dqn_agent, random_agent, environment, episodes):
    wins, losses, draws = 0, 0, 0

    for _ in range(episodes):
        state = environment.reset()
        done = False

        while not done:
            empty_cells = environment.get_empty_cells()

            # DQN agent takes its turn
            action = dqn_agent.get_action(state, 0, empty_cells)  # epsilon = 0 for the trained agent
            _, _, done = environment.perform_action(action)
            if done:
                wins += 1
                break

            # Random agent takes its turn
            action = random_agent.get_action(state, empty_cells)
            _, reward, done = environment.perform_action(action)
            if done:
                if reward == 1:
                    losses += 1
                else:
                    draws += 1
                break

    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Instantiate environment
    environment = TicTacToe()

    # Instantiate DQN agent
    state_dim = 10000
    action_dim = 10000
    hidden_dim = 128
    lr = 0.001

    dqn_agent = DQNAgent(state_dim, action_dim, hidden_dim, lr)

    # Train DQN agent
    episodes = 100
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.999
    train(dqn_agent, environment, episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay)

    # Instantiate random agent
    random_agent = RandomAgent()

    # Have the DQN agent compete with the random agent
    compete(dqn_agent, random_agent, environment, 1000)
