# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

def board_printer_3d(board):
    # Verify if the matrix has the correct size
    if len(board) != 3 or len(board[0]) != 9:
        print("Invalid board size!")
        return

    # Create empty lists to store the x, y, z coordinates, color, and text of each point
    x, y, z, color, text = [], [], [], [], []

    # Color map
    color_map = {'X': 'red', 'O': 'blue', 'E': 'black'}

    # Iterate over the board
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Add the coordinates, color, and text of the point to the lists
                x.append(i)
                y.append(j)
                z.append(k)
                color.append(color_map[board[i][j*3+k]])
                text.append(f'({i},{j*3+k})')

    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, text=text,
        mode='markers+text',
        marker=dict(
            size=12,
            color=color,
            opacity=0.8
        ),
        textposition='top center'
    )])

    # Update layout
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    width=500,  # Adjust the width of the plot
    height=500  # Adjust the height of the plot
    )

    fig.show()
def check_win_3d(board):
    # Define all possible winning combinations on each layer
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]               # Diagonals
    ]

    # Check if any player has won on any layer
    for i in range(3):
        for combination in winning_combinations:
            if board[i][combination[0]] == board[i][combination[1]] == board[i][combination[2]] != 'E':
                return board[i][combination[0]]  # Return the winning player ('X' or 'O')

    # Check for winning combinations that span layers
    for i in range(9):
        # Columns that span layers
        if board[0][i] == board[1][i] == board[2][i] != 'E':
            return board[0][i]
    for i in range(3):
      if board[i][0]==board[i][1]==board[i][2]!='E':
        return board[i][0]
      if board[i][3]==board[i][4]==board[i][5]!='E':
        return board[i][3]
      if board[i][6]==board[i][7]==board[i][8]!='E':
        return board[i][6]

    # Check for diagonals that span layers
    if board[0][0] == board[1][4] == board[2][8] != 'E':
        return board[0][0]
    if board[0][2] == board[1][4] == board[2][6] != 'E':
        return board[0][2]

   #Check for rows that span layers

    if board[0][0] == board[1][1] == board[2][2] != 'E':
        return board[0][0]

    if board[0][3] == board[1][4] == board[2][5] != 'E':
        return board[0][3]

    if board[0][6] == board[1][7] == board[2][8] != 'E':
        return board[0][6]


    # Check if the board is full (draw)
    if 'E' not in np.array(board).flatten():
        return 'Draw'

    # If no player has won and the board is not full, continue the game
    return 'Continue'

def next_turn(turn):
  #Gets a turn and returns its erverse
  if turn=='X':
    return 'O'
  else:
    return 'X'

def A(board,action):
  #Checks if an action is possible and legal on a board
  layer=action[0]
  square=action[1]

  if board[layer][square]!='E':
    return False

  if check_win_3d(board)!= 'Continue':
    return False

  return True

def reward(board):
  #rewards +1 if X wins, -1 if O wins and 0 otherwise
  if  check_win_3d(board)=='X':
    return 1
  elif check_win_3d(board)=='O':
    return -1
  else:
    return 0

def next_state_same(board,turn,action):
  #Gets a board, an action indice and a turn and returns a new board with that player's mark in that action
  new_board=[]
  layer=action[0]
  square=action[1]
  for j in range(3):
    new_layer=[]
    for i in range(9):
      if i==square and j==layer:
        new_layer.append(turn)
      else:
        new_layer.append(board[j][i])
    new_board.append(new_layer)

  return new_board

def next_state_reverse(board,turn,action):
  #Gets a board, an action indice and a turn and returns a new board with that player's mark in that action
  new_board=[]
  layer=action[0]
  square=action[1]
  for j in range(3):
    new_layer=[]
    for i in range(9):
      if i==square and j==layer:
        new_layer.append(next_turn(turn))
      else:
        new_layer.append(board[j][i])
    new_board.append(new_layer)

  return new_board

def v_encode_state(board, turn):
    # Define a mapping from cell state to code
    cell_to_code = {'X': 3, 'O': 1, 'E': 2}

    # Flatten the board and convert each cell to code
    flat_board = [cell_to_code[cell] for layer in board for cell in layer]

    # Convert turn to code and append it to the encoded state
    turn_code = 1 if turn == 'X' else -1
    flat_board.append(turn_code)

    return flat_board
def board_transformer(board):
    # Define a mapping from code to cell state
    code_to_cell = {1: 'O', 2: 'E', 3: 'X'}

    # Transform the board
    transformed_board = [[code_to_cell[code] for code in layer] for layer in board]

    return transformed_board


def value_function(x, coef):
    # Initialize the value to the constant term
    value = coef['constant']

    # Add the linear terms
    for i in range(len(x)):
        value += coef['linear'][i] * x[i]

    # Add the quadratic terms
    for i in range(len(x)):
        for j in range(len(x)):
            value += coef['quadratic'][(i, j)] * x[i] * x[j]

    return value

def value_function_gradient(x, coef):
    # Initialize the gradient to an empty dictionary
    gradient = {'constant': 0, 'linear': [], 'quadratic': {}}

    # Compute the gradient for the linear terms
    for i in range(len(x)):
        gradient['linear'].append(x[i])

    # Compute the gradient for the quadratic terms
    for i in range(len(x)):
        for j in range(len(x)):
            gradient['quadratic'][(i, j)] = x[i] * x[j]

    # The gradient for the constant term is just 1
    gradient['constant'] = 1
    return gradient

def td_error_value_function(reward, old_state, new_state, coef):
    # Calculate the TD error
    V_old = value_function(old_state, coef)
    V_new = value_function(new_state, coef)

    # The discount factor is set to 1 as the game horizon is 27 moves
    gamma = 1

    TD_error = reward + gamma * V_new - V_old
    return TD_error

def value_function_update_coefficients(coef, td_error, gradient, learning_rate):
    # Update the coefficients of the value function using the TD error and the learning rate
    coef['constant'] += learning_rate * td_error * gradient['constant']

    for i in range(len(coef['linear'])):
        coef['linear'][i] += learning_rate * td_error * gradient['linear'][i]

    for key in coef['quadratic'].keys():
        coef['quadratic'][key] += learning_rate * td_error * gradient['quadratic'][key]

    return coef

def value_greedy_action(board, turn, coef, danger_prob):
    # Initialize the best value and action
    if turn == 'X':
        best_value = -float('inf')
    else:  # turn == 'O'
        best_value = float('inf')
    best_action = None

    # Iterate over all possible actions
    for layer in range(3):
        for square in range(9):
            action = (layer, square)

            # Check if the action is legal
            if A(board, action):
                # Take the action and encode the next state
                prob = danger_prob[layer][square]
                next_board = np.random.choice([next_state_same, next_state_reverse], p=[1-prob, prob])(board, turn, action)
                next_state = v_encode_state(next_board, next_turn(turn))

                # Calculate the value of the next state
                value = value_function(next_state, coef)

                # If this value is the best so far, store the value and action
                if turn == 'X' and value > best_value:
                    best_value = value
                    best_action = action
                elif turn == 'O' and value < best_value:
                    best_value = value
                    best_action = action

    # Return the action that leads to the highest (or lowest) value state
    return best_action

import random

def random_action(board, *args):
    # Initialize a list to store all legal actions
    legal_actions = []
    action=None
    # Iterate over all possible actions
    for layer in range(3):
        for square in range(9):
            action = (layer, square)

            # Check if the action is legal
            if Q_A(board, action):
                # If the action is legal, add it to the list of legal actions
                legal_actions.append(action)

    # Choose a legal action at random

    action = random.choice(legal_actions)

    return action

import numpy as np
def td_learning(board, turn, coef, danger_prob, learning_rate, epsilon):
    # Initialize the state
    state = v_encode_state(board, turn)

    # Play the game until it's over
    while True:
        if np.random.rand() < epsilon:  # epsilon is a small positive number
            action = random_action(board)
        else:
            action = value_greedy_action(board, turn, coef, danger_prob)

        # Decay epsilon

        # If the action is None, the game has ended, so we break the loop
        if action is None:
            break

        # Take the action and observe the new state and reward
        # Use the danger probability to decide whether to place the same or the opposite mark
        layer, square = action
        prob = danger_prob[layer][square]
        next_board = np.random.choice([next_state_same, next_state_reverse], p=[1-prob, prob])(board, turn, action)

        next_state = v_encode_state(next_board, next_turn(turn))
        r = reward(next_board)

        # Calculate the TD error
        td_err = td_error_value_function(r, state, next_state, coef)

        # Calculate the gradient of the value function
        gradient = value_function_gradient(state, coef)

        # Update the coefficients of the value function
        coef = value_function_update_coefficients(coef, td_err, gradient, learning_rate)

        # Update the state and turn
        board = next_board
        turn = next_turn(turn)
        state = next_state

        # Check if the game has ended
        if check_win_3d(board) != 'Continue':
            break

    print(td_err)
    return coef

def play_game(agent1, agent2, coef,danger_prob):
    # Initialize the game board and the turn
    board = [['E' for _ in range(9)] for _ in range(3)]
    turn = 'X'  # The X player starts

    # Play the game until it's over
    while check_win_3d(board) == 'Continue':
        # Choose an action based on the current turn
        if turn == 'X':
            action = agent1(board, turn, coef,danger_prob) if agent1 == value_greedy_action or agent1==q_greedy_action else agent1(board)
        else:
            action = agent2(board, turn, coef,danger_prob) if agent2 == value_greedy_action or agent2==q_greedy_action else agent2(board)

        # Take the action
        layer, square = action
        prob = danger_prob[layer][square]
        next_board = np.random.choice([next_state_same, next_state_reverse], p=[1-prob, prob])(board, turn, action)

        # Switch turns
        turn = next_turn(turn)
        board=next_board
    # Return the result of the game
    return check_win_3d(board)

def evaluate_agents(agent1, agent2, coef, simulations, danger_prob):
        # Initialize a dictionary to store the results
        results = defaultdict(int)
        i = 0
        # Play the games
        for i in range(simulations):
            result = play_game(agent1, agent2, coef, danger_prob)
            results[result] += 1
        # Return the results
        return results

def eligibity_update_coefficients(coef, td_error, gradient, learning_rate, eligibility):
    # Update the coefficients of the value function using the TD error, the learning rate, and the eligibility traces
    coef['constant'] += learning_rate * td_error * eligibility['constant']

    for i in range(len(coef['linear'])):
        coef['linear'][i] += learning_rate * td_error * eligibility['linear'][i]

    for key in coef['quadratic'].keys():
        coef['quadratic'][key] += learning_rate * td_error * eligibility['quadratic'][key]

    return coef

def td_lambda_learning(board, turn, coef, danger_prob, learning_rate, lambda_=0.9, epsilon=0.05):
    # Initialize the state
    state = v_encode_state(board, turn)

    # Initialize the eligibility traces
    eligibility = {'constant': 0, 'linear': [0]*len(state), 'quadratic': {(i, j): 0 for i in range(len(state)) for j in range(len(state))}}

    # Play the game until it's over
    while True:
        if np.random.rand() < epsilon:  # epsilon is a small positive number, e.g., 0.1
            action = random_action(board)
        else:
            action = value_greedy_action(board, turn, coef , danger_prob)

        # If the action is None, the game has ended, so we break the loop
        if action is None:
            break

        # Take the action and observe the new state and reward
        # Use the danger probability to decide whether to place the same or the opposite mark
        layer, square = action
        prob = danger_prob[layer][square]
        next_board = np.random.choice([next_state_same, next_state_reverse], p=[1-prob, prob])(board, turn, action)

        next_state = v_encode_state(next_board, next_turn(turn))
        r = reward(next_board)

        # Calculate the TD error
        td_err = td_error_value_function(r, state, next_state, coef)

        # Calculate the gradient of the value function
        gradient = value_function_gradient(state, coef)

        # Update the eligibility traces
        eligibility['constant'] = lambda_ * eligibility['constant'] + gradient['constant']
        for i in range(len(eligibility['linear'])):
            eligibility['linear'][i] = lambda_ * eligibility['linear'][i] + gradient['linear'][i]
        for key in eligibility['quadratic'].keys():
            eligibility['quadratic'][key] = lambda_ * eligibility['quadratic'][key] + gradient['quadratic'][key]

        # Update the coefficients of the value function
        coef = eligibity_update_coefficients(coef, td_err, gradient, learning_rate, eligibility)

        # Update the state and turn
        board = next_board
        turn = next_turn(turn)
        state = next_state

        # Check if the game has ended
        if check_win_3d(board) != 'Continue':
            break
        print(td_err)
    return coef
def q_encode_state_action(board, turn, action):
    # Define a mapping from cell state to code
    cell_to_code = {'X': 3, 'O': 1, 'E': 2}

    # Flatten the board and convert each cell to code
    flat_board = [cell_to_code[cell] for layer in board for cell in layer]

    # Convert turn to code and append it to the encoded state-action
    turn_code = 1 if turn == 'X' else -1
    flat_board.append(turn_code)

    # Convert action to code and append it to the encoded state-action
    if action is not None:
        action_code = [i for i in action]
    else:
        action_code = [0, 0]  # Encode 'None' action as [0, 0]

    flat_board.extend(action_code)

    return flat_board
def Q_A(board, action):
    # Checks if an action is possible and legal on a board
    layer = action[0]
    square = action[1]

    if board[layer][square] != 'E':
        return False

    if check_win_3d(board) != 'Continue':
        return False

    return True

def q_function(x, coef):
    # Check if the game has ended
    if check_win_3d(board_transformer(np.array(x[:-3]).reshape(3, 9))) != 'Continue':
        return 0

    # Initialize the value to the constant term
    value = coef['constant']

    # Add the linear terms
    for i in range(len(x)):
        value += coef['linear'][i] * x[i]

    # Add the quadratic terms
    for i in range(len(x)):
        for j in range(len(x)):
            value += coef['quadratic'][(i, j)] * x[i] * x[j]

    return value

def q_function_gradient(x, coef):
    # Initialize the gradient to an empty dictionary
    gradient = {'constant': 0, 'linear': [], 'quadratic': {}}

    # Compute the gradient for the linear terms
    for i in range(len(x)):
        gradient['linear'].append(x[i])

    # Compute the gradient for the quadratic terms
    for i in range(len(x)):
        for j in range(len(x)):
            gradient['quadratic'][(i, j)] = x[i] * x[j]

    # The gradient for the constant term is just 1
    gradient['constant'] = 1
    return gradient

def td_error_q_function(old_state_action,coef,danger_prob,board,turn,action):
    # Calculate the TD error

    Q_old = q_function(old_state_action, coef)
    layer, square = action  # Corrected line
    prob = danger_prob[layer][square]
    next_board = np.random.choice([next_state_same, next_state_reverse], p=[1 - prob, prob])(board, turn, action)
    r = reward(next_board)

    next_best_action= q_greedy_action(next_board,next_turn(turn),coef,danger_prob)

    new_state_action=q_encode_state_action(next_board,next_turn(turn),next_best_action)

    Q_new= q_function(new_state_action,coef)

    # The discount factor is set to 1 as the game horizon is 27 moves
    gamma = 1

    TD_error = r + gamma * Q_new - Q_old
    return TD_error,next_board

def q_function_update_coefficients(coef, td_error, gradient, learning_rate):
    # Update the coefficients of the q function using the TD error and the learning rate
    coef['constant'] += learning_rate * td_error * gradient['constant']

    for i in range(len(coef['linear'])):
        coef['linear'][i] += learning_rate * td_error * gradient['linear'][i]

    for key in coef['quadratic'].keys():
        coef['quadratic'][key] += learning_rate * td_error * gradient['quadratic'][key]

    return coef

def q_greedy_action(board, turn, coef, danger_prob):
    # Initialize the best value and action
    if turn == 'X':
        best_value = -float('inf')
    else:  # turn == 'O'
        best_value = float('inf')
    best_action = None

    # Iterate over all possible actions
    for layer in range(3):
        for square in range(9):
            action = (layer, square)

            # Check if the action is legal
            if Q_A(board, action):
                # Take the action and encode the next state
                prob = danger_prob[layer][square]
                next_board = np.random.choice([next_state_same, next_state_reverse], p=[1-prob, prob])(board, turn, action)
                next_state_action = q_encode_state_action(next_board, next_turn(turn), action)

                # Calculate the value of the next state
                value = q_function(next_state_action, coef)

                # If this value is the best so far, store the value and action
                if turn == 'X' and value > best_value:
                    best_value = value
                    best_action = action
                elif turn == 'O' and value < best_value:
                    best_value = value
                    best_action = action

    # Return the action that leads to the highest (or lowest) value state
    return best_action

import time
import os

def play_game_animate(coef, danger_prob, pause_time=1):
    # Initialize the board and the turn
    board = np.full((3, 9), 'E')

    turn = 'X'

    # Play the game until it's over
    while True:
        # Decide on an action
        if turn == 'X':  # The trained agent
            action = q_greedy_action(board, turn, coef, danger_prob)
        else:  # The random agent
            action = random_action(board)

        # If the action is None, the game has ended, so we break the loop
        if action is None:
            break

        # Take the action and observe the new state
        layer, square = action
        prob = danger_prob[layer][square]
        next_board = np.random.choice([next_state_same, next_state_reverse], p=[1-prob, prob])(board, turn, action)

        # Update the state and turn
        board = next_board
        turn = next_turn(turn)

        # Print the board
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears the console
        board_printer_3d(board)

        # Pause for a while
        time.sleep(pause_time)

        # Check if the game has ended
        if check_win_3d(board) != 'Continue':
            break

    # Print the final result
    result = check_win_3d(board)
    if result == 'Draw':
        print("The game ended in a draw.")
    else:
        print(f"The winner is {result}!")

def q_learning(board, turn, coef, danger_prob, learning_rate, epsilon):
    # Initialize the state-action
    state_action = None

    # Play the game until it's over
    while True:
        if np.random.rand() < epsilon:  # epsilon is a small positive number
            action = random_action(board)
        else:
            action = q_greedy_action(board, turn, coef, danger_prob)

        # If the action is None, the game has ended, so we break the loop
        if action is None:
            break

        # Take the action and observe the new state and reward
        layer, square = action  # Corrected line
        prob = danger_prob[layer][square]
        state_action=q_encode_state_action(board,turn,action)

        # Calculate the TD error
        if state_action is not None:  # Only compute TD error and update when state_action is not None
            temp= td_error_q_function(state_action,coef,danger_prob,board,turn,action)
            td_err = temp[0]


            # Calculate the gradient of the Q-function
            gradient = q_function_gradient(state_action, coef)

            # Update the coefficients of the Q-function
            coef = q_function_update_coefficients(coef, td_err, gradient, learning_rate)
            print(td_err)
            board = temp[1]
            turn = next_turn(turn)
        else:
            break
        # Update the state, state-action, and turn


        # Check if the game has ended
        if check_win_3d(board) != 'Continue':
            break

    return coef


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Initialize the coefficients of the value function
    coef = {
        'constant': 0,
        'linear': [0 for _ in range(30)],  # 27 cells in the board plus the turn
        'quadratic': {(i, j): 0 for i in range(30) for j in range(30)}
    }

    # Set the learning rate
    learning_rate = 0.00001
    epsilon=1
    epsilon_decay=0.9995
    min_epsilon=0.1

    # Set the number of games to train on
    num_games = 1000
    simulations=1000
    # Initialize the danger probability matrix
    danger_prob = [[random.random() for _ in range(9)] for _ in
                   range(3)]  # for example, all cells have a 0.1 danger probability


    # Train the agent using Q-learning
    for i in range(num_games):
        board = [['E' for _ in range(9)] for _ in range(3)]
        turn = 'X'
        coef = q_learning(board, turn, coef, danger_prob, learning_rate, epsilon)
        epsilon = min(epsilon_decay*epsilon, min_epsilon)

    # play_game_animate(coef, danger_prob)

    # # Rest of the code remains the same...
    # results1 = evaluate_agents(q_greedy_action, random_action, coef, simulations, danger_prob)
    # results2 = evaluate_agents(random_action, q_greedy_action, coef, simulations, danger_prob)
    # results3 = evaluate_agents(q_greedy_action, q_greedy_action, coef, simulations, danger_prob)
    # results4 = evaluate_agents(random_action, random_action, coef, simulations, danger_prob)
    #
    # labels = ['Q-learning first', 'Random first', 'Q-learning vs Q-learning', 'Random vs Random']
    # first_player_wins = [results1['X'], results2['X'], results3['X'], results4['X']]
    # second_player_wins = [results1['O'], results2['O'], results3['O'], results4['O']]
    # draws = [results1['Draw'], results2['Draw'], results3['Draw'], results4['Draw']]
    #
    # x = range(len(labels))
    #
    # plt.figure(figsize=(10, 6))
    # plt.bar(x, first_player_wins, width=0.3, align='center', color='b', label='First Player Wins')
    # plt.bar(x, second_player_wins, width=0.3, bottom=first_player_wins, align='center', color='r', label='Second Player Wins')
    # plt.bar(x, draws, width=0.3, bottom=[i + j for i, j in zip(first_player_wins, second_player_wins)], align='center', color='g', label='Draws')
    # plt.xticks(x, labels)
    # plt.xlabel('Scenarios')
    # plt.ylabel('Number of Games')
    # plt.title('Tic Tac Toe Results using Q-learning')
    # plt.legend()
    # plt.show()

    # # Train the agent using TD learning
    # for i in range(num_games):
    #     # Initialize the game board and the turn
    #     board = [['E' for _ in range(9)] for _ in range(3)]
    #     turn = 'X'  # The X player starts
    #
    #     # Update the coefficients using TD learning
    #     coef = td_lambda_learning(board, turn, coef, danger_prob, learning_rate)
    #
    #     epsilon= min(epsilon_decay*epsilon,min_epsilon)
    # # Print the learned coefficients of the value function
    # print(coef)
    #
    # from collections import defaultdict
    # import matplotlib.pyplot as plt
    #
    # simulations=1000
    #
    # # Play games and collect results
    #
    # # TD learning agent vs Random agent (TD first)
    # results1 = evaluate_agents(value_greedy_action, random_action, coef, simulations, danger_prob)
    #
    # # TD learning agent vs Random agent (Random first)
    # results2 = evaluate_agents(random_action, value_greedy_action, coef, simulations, danger_prob)
    #
    # # TD learning agent vs TD learning agent
    # results3 = evaluate_agents(value_greedy_action, value_greedy_action, coef, simulations, danger_prob)
    #
    # # Random agent vs Random agent
    # results4 = evaluate_agents(random_action, random_action, coef, simulations, danger_prob)
    #
    # # Create lists for the bar chart
    # labels = ['TD first', 'Random first', 'TD vs TD', 'Random vs Random']
    # first_player_wins = [results1['X'], results2['X'], results3['X'], results4['X']]
    # second_player_wins = [results1['O'], results2['O'], results3['O'], results4['O']]
    # draws = [results1['Draw'], results2['Draw'], results3['Draw'], results4['Draw']]
    #
    # x = range(len(labels))
    #
    # # Create the bar chart
    # plt.figure(figsize=(10, 6))
    # plt.bar(x, first_player_wins, width=0.3, align='center', color='b', label='First Player Wins')
    # plt.bar(x, second_player_wins, width=0.3, bottom=first_player_wins, align='center', color='r',
    #         label='Second Player Wins')
    # plt.bar(x, draws, width=0.3, bottom=[i + j for i, j in zip(first_player_wins, second_player_wins)], align='center',
    #         color='g', label='Draws')
    #
    # # Add labels and title
    # plt.xticks(x, labels)
    # plt.xlabel('Scenarios')
    # plt.ylabel('Number of Games')
    # plt.title('Tic Tac Toe Results')
    # plt.legend()
    #
    # # Show the plot
    # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
