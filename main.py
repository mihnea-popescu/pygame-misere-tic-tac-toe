import itertools
import time

import pygame, sys, copy, math
from itertools import chain
import numpy as np
import statistics

MAX_DEPTH = 0

class Game:
    """
    State defining the game.
    JMIN - human player
    JMAX - computer
    """
    NO_COLUMNS = 3
    JMIN = None
    JMAX = None
    EMPTY = '#'
    JMAX_TIMES = []
    JMAX_NODES = []
    JMIN_moves = 0
    JMAX_moves = 0
    nodes_no = 0

    @classmethod
    def start(cls, display, NO_COLUMNS=3, cell_size=100):
        cls.display = display
        cls.cell_size = cell_size
        cls.x_img = pygame.image.load('x.png')
        cls.x_img = pygame.transform.scale(cls.x_img, (
        cell_size, math.floor(cell_size * cls.x_img.get_height() / cls.x_img.get_width())))
        cls.zero_img = pygame.image.load('zero.png')
        cls.zero_img = pygame.transform.scale(cls.zero_img, (
        cell_size, math.floor(cell_size * cls.zero_img.get_height() / cls.zero_img.get_width())))
        cls.gridCells = []
        for line in range(NO_COLUMNS):
            cls.gridCells.append([])
            for column in range(NO_COLUMNS):
                patr = pygame.Rect(column * (cell_size + 1), line * (cell_size + 1), cell_size, cell_size)
                cls.gridCells[line].append(patr)

    """
        Method that updates the Pygame interface
    """
    def draw_grid(self, mark=None): 
        possibilities = []
        if mark is None and self.final() is False:
            table = list(chain.from_iterable(self.matrix))
            if table.count(self.EMPTY) == len(table):
                for i in range(self.NO_COLUMNS):
                    for j in range(self.NO_COLUMNS):
                        possibilities.append((i, j))
            elif (table.count(self.JMIN) == 1 and table.count(self.JMAX) == 0) or (table.count(self.JMIN) == 0 and table.count(self.JMAX) == 1):
                for i in range(self.NO_COLUMNS):
                    for j in range(self.NO_COLUMNS):
                        if self.matrix[i][j] != self.JMIN and self.matrix[i][j] != self.JMAX:
                            possibilities.append((i, j))
            elif table.count(self.JMIN) == table.count(self.JMAX):
                # It's the starting player's turn (x)
                possibilities = self.moving_possibilities('x')
            else:
                # It's the second player's turn (0)
                possibilities = self.moving_possibilities('0')

        for line in range(Game.NO_COLUMNS):
            for column in range(Game.NO_COLUMNS):
                if mark is not None and (line, column) in mark:
                    # if the cell is marked, draw it with red
                    color = (255, 0, 0)
                elif (line, column) in possibilities:
                    # if it is a possible move, draw it in green
                    color = (115, 218, 104)
                else:
                    # otherwise draw it with white
                    color = (255, 255, 255)
                pygame.draw.rect(self.__class__.display, color,
                                 self.__class__.gridCells[line][column])
                if self.matrix[line][column] == 'x':
                    self.__class__.display.blit(self.__class__.x_img, (column * (self.__class__.cell_size + 1),
                                                                       line * (self.__class__.cell_size + 1) + (
                                                                                   self.__class__.cell_size - self.__class__.x_img.get_height()) // 2))
                elif self.matrix[line][column] == '0':
                    self.__class__.display.blit(self.__class__.zero_img, (column * (self.__class__.cell_size + 1),
                                                                          line * (self.__class__.cell_size + 1) + (
                                                                                      self.__class__.cell_size - self.__class__.zero_img.get_height()) // 2))
        # update the interface
        pygame.display.update()

    def __init__(self, table=None):
        if table:
            self.matrix = table
        else:
            self.matrix = []
            for i in range(self.__class__.NO_COLUMNS):
                self.matrix.append([self.__class__.EMPTY] * self.__class__.NO_COLUMNS)

    """
        Shows statistics about the computer's average time to select a move and the number
        of nodes generated
    """
    @classmethod
    def show_computer_times(cls):
        times = cls.JMAX_TIMES
        nodes = cls.JMAX_NODES
        if len(times) == 0 or len(nodes) == 0:
            return
        print('Minimum time for computer to think the next move was ' + str(min(times)) + " ms")
        print('Maximum time for computer to think the next move was ' + str(max(times)) + " ms")
        print('Average time for computer to think the next move was ' + str(sum(times) / len(times)) + " ms")
        print('Median time for computer to think the next move was ' + str(statistics.median(times)) + " ms")
        print('Minimum number of nodes generated by the computer was ' + str(min(nodes)))
        print('Maximum number of nodes generated by the computer was ' + str(max(nodes)))
        print('Average number of nodes generated by the computer was ' + str(sum(nodes) / len(nodes)))
        print('Median number of nodes generated by the computer was ' + str(statistics.median(nodes)))

    @classmethod
    def opposing_player(cls, player):
        return cls.JMAX if player == cls.JMIN else cls.JMIN

    """
        Returns all the possibilites that a player has
        The rule is that (besides the first move), each player can only place a symbol
        In the adjacent cells
    """
    def moving_possibilities(self, player):
        possibilities = []
        for i in range(self.NO_COLUMNS):
            for j in range(self.NO_COLUMNS):
                if self.matrix[i][j] == player:
                    # upper left
                    if self.exists_and_it_is_empty(i - 1, j - 1) and (i - 1, j - 1) not in possibilities:
                        possibilities.append((i - 1, j - 1))

                    # up
                    if self.exists_and_it_is_empty(i - 1, j) and (i - 1, j) not in possibilities:
                        possibilities.append((i - 1, j))

                    # upper right
                    if self.exists_and_it_is_empty(i - 1, j + 1) and (i - 1, j + 1) not in possibilities:
                        possibilities.append((i - 1, j + 1))

                    # right
                    if self.exists_and_it_is_empty(i, j + 1) and (i, j + 1) not in possibilities:
                        possibilities.append((i, j + 1))

                    # lower right
                    if self.exists_and_it_is_empty(i + 1, j + 1) and (i + 1, j + 1) not in possibilities:
                        possibilities.append((i + 1, j + 1))

                    # down
                    if self.exists_and_it_is_empty(i + 1, j) and (i + 1, j) not in possibilities:
                        possibilities.append((i + 1, j))

                    # lower left
                    if self.exists_and_it_is_empty(i + 1, j - 1) and (i + 1, j - 1) not in possibilities:
                        possibilities.append((i + 1, j - 1))

                    # left
                    if self.exists_and_it_is_empty(i, j - 1) and (i, j - 1) not in possibilities:
                        possibilities.append((i, j - 1))
        return possibilities

    """
        Return the number of moves which a player has that would not
        Make him lose
    """
    def possible_moves_no(self, player):
        possibilities = self.moving_possibilities(player)
        nr_moves = len(possibilities)
        for possibility in possibilities:
            i = possibility[0]
            j = possibility[1]

            # Upper left
            if (i - 2) >= 0 and (j - 2) >= 0:
                if self.matrix[i - 2][j - 2] == self.matrix[i - 1][j - 1] == player:
                    nr_moves = nr_moves - 1

            # Up
            if (i - 2) >= 0:
                if self.matrix[i-2][j] == self.matrix[i - 1][j] == player:
                    nr_moves = nr_moves - 1

            # Upper right
            if (i - 2) >= 0 and (j + 2) < self.NO_COLUMNS:
                if self.matrix[i - 2][j + 2] == self.matrix[i - 1][j + 1] == player:
                    nr_moves = nr_moves - 1

            # Right
            if (j + 2) < self.NO_COLUMNS:
                if self.matrix[i][j + 1] == self.matrix[i][j + 2] == player:
                    nr_moves = nr_moves - 1

            # Lower right
            if (i + 2) < self.NO_COLUMNS and (j + 2) < self.NO_COLUMNS:
                if self.matrix[i + 1][j + 1] == self.matrix[i + 2][j + 2] == player:
                    nr_moves = nr_moves - 1

            # Down
            if (i + 2) < self.NO_COLUMNS:
                if self.matrix[i + 1][j] == self.matrix[i + 2][j] == player:
                    nr_moves = nr_moves - 1

            # Lower left
            if (i + 2) < self.NO_COLUMNS and (j - 2) >= 0:
                if self.matrix[i + 1][j - 1] == self.matrix[i + 2][j - 2] == player:
                    nr_moves = nr_moves - 1

            # Left
            if (j - 2) >= 0:
                if self.matrix[i][j - 1] == self.matrix[i][j - 2] == player:
                    nr_moves = nr_moves - 1

        return nr_moves

    """
        Verify if the specified position exists in the matrix
        And it is not occupied by any player
    """
    def exists_and_it_is_empty(self, i, j):
        if i < 0 or i >= self.NO_COLUMNS:
            return False
        if j < 0 or j >= self.NO_COLUMNS:
            return False
        if self.matrix[i][j] != self.EMPTY:
            return False
        return True

    """
        Shows the final grid, highlighting the finishing formation
    """
    def show_final(self):
        for i in range(self.NO_COLUMNS):
            for j in range(self.NO_COLUMNS):
                if self.matrix[i][j] == Game.EMPTY:
                    continue

                # Line
                if (j + 2) < self.NO_COLUMNS:
                    if self.matrix[i][j] == self.matrix[i][j + 1] == self.matrix[i][j + 2]:
                        return self.draw_grid(list([(i, j), (i, j + 1), (i, j + 2)]))

                # Column
                if (i + 2) < self.NO_COLUMNS:
                    if self.matrix[i][j] == self.matrix[i + 1][j] == self.matrix[i + 2][j]:
                        return self.draw_grid(list([(i, j), (i + 1, j), (i + 2, j)]))

                # Diagonal
                if (i - 1) >= 0 and (j - 1) >= 0 and (i + 1) < self.NO_COLUMNS and (j + 1) < self.NO_COLUMNS:
                    # Main diagonal
                    if self.matrix[i][j] == self.matrix[i-1][j-1] == self.matrix[i+1][j+1]:
                        return self.draw_grid(list([(i, j), (i + 1, j + 1), (i - 1, j - 1)]))
                    # Secondary diagonal
                    if self.matrix[i][j] ==  self.matrix[i-1][j+1] == self.matrix[i+1][j-1]:
                        return self.draw_grid(list([(i, j), (i - 1, j + 1), (i + 1, j - 1)]))

    """
        Verify if there is a winner in the curent game state
        A player wins when the opposing player places 3 symbols consecutively on a row, column or diagonal
        If there aren't any spaces to place, then the match ends in a draw
    """
    def final(self):
        for i in range(self.NO_COLUMNS):
            for j in range(self.NO_COLUMNS):
                if self.matrix[i][j] == Game.EMPTY:
                    continue

                # Row
                if (j + 2) < self.NO_COLUMNS:
                    if self.matrix[i][j] == self.matrix[i][j + 1] == self.matrix[i][j + 2]:
                        return self.__class__.opposing_player(self.matrix[i][j])

                # Column
                if (i + 2) < self.NO_COLUMNS:
                    if self.matrix[i][j] == self.matrix[i + 1][j] == self.matrix[i + 2][j]:
                        return self.__class__.opposing_player(self.matrix[i][j])

                # Diagonal
                if (i - 1) >= 0 and (j - 1) >= 0 and (i + 1) < self.NO_COLUMNS and (j + 1) < self.NO_COLUMNS:
                    # Main diagonal
                    if self.matrix[i][j] == self.matrix[i-1][j-1] == self.matrix[i+1][j+1]:
                        return self.__class__.opposing_player(self.matrix[i][j])
                    # Secondary diagonal
                    if self.matrix[i][j] ==  self.matrix[i-1][j+1] == self.matrix[i+1][j-1]:
                        return self.__class__.opposing_player(self.matrix[i][j])

        if self.__class__.EMPTY not in list(chain.from_iterable(self.matrix)):
            return 'draw'
        else:
            return False

    """
        Returns the possible moves of a player
    """
    def moves(self, player):
        l_moves = [] # moves list
        for i in range(self.__class__.NO_COLUMNS):
            for j in range(self.__class__.NO_COLUMNS):
                if self.can_move(self.matrix, i, j, player):
                    matrix_copy = copy.deepcopy(self.matrix)
                    matrix_copy[i][j] = player
                    l_moves.append(Game(matrix_copy))
        return l_moves

    """
        Returns wheter the player can place as symbol in a (i,j) position
    """
    @staticmethod
    def can_move(matrix, i, j, player):
        # Verify if it is the initial move
        if list(itertools.chain(*matrix)).count(player) == 0:
            return True

        if matrix[i][j] != Game.EMPTY:
            return False

        for rand in range(i-1, i+2):
            for column in range (j-1, j+2):
                if (rand, column) != (i, j) and rand in range(Game.NO_COLUMNS) and column in range(Game.NO_COLUMNS):
                    if matrix[rand][column] == player:
                        return True

        return False

    """
        An open line is a line where the player can form a losing formation
        Technically it's a line without any symbols of the opposing player
    """
    def open_line(self, line, player):
        opponent = self.opposing_player(player)
        # verify if this line contains any symbols of the opposing player
        if not opponent in line:
            return 1
        return 0

    def open_lines(self, player):
        open_lines_count = 0

        # lines
        for i in range(self.NO_COLUMNS):
            open_lines_count = open_lines_count + self.open_line(self.matrix[i], player)
        # columns
        for i in range(self.NO_COLUMNS):
            open_lines_count = open_lines_count + self.open_line([rand[i] for rand in self.matrix], player)
        # diagonals
        matrix_np = np.array(self.matrix)
        diags = [matrix_np[::-1, :].diagonal(i) for i in range(-self.NO_COLUMNS+1, self.NO_COLUMNS)]
        diags.extend(matrix_np.diagonal(i) for i in range(self.NO_COLUMNS))
        for i in range(len(diags)):
            open_lines_count = open_lines_count + self.open_line(diags[i], player)

        return open_lines_count

    """
        Score estimation is done in the following manner:
        1. Calculation of the number of open lines the computer has minus the open lines of the player
            Each open line represents a "space of maneuver" for the computer because it allows him to prevent
            making new formations
        
        2. Number of moves possible (without losing) by the computer minus the number of moves possible by the player
            The more possible moves the computer has, the more he chances he has at winning
    """
    def estimate_score(self, depth):
        t_final = self.final()

        if t_final == self.__class__.JMAX:
            return (99 + depth)
        elif t_final == self.__class__.JMIN:
            return (-99 - depth)
        elif t_final == 'draw':
            return 0
        else:
            return self.estimate_score_player(self.__class__.JMAX) - self.estimate_score_player(self.__class__.JMIN)

    """
        Estimate the score of a player
    """
    def estimate_score_player(self, player):
        return self.possible_moves_no(player)

    """
        Get the game matrix string output
    """
    def displayOutput(self):
        sir = "  |"
        sir += " ".join([str(i) for i in range(self.NO_COLUMNS)]) + "\n"
        sir += "-" * (self.NO_COLUMNS + 1) * 2 + "\n"
        for i in range(self.NO_COLUMNS):  # itereaza prin linii
            sir += str(i) + " |" + " ".join([str(x) for x in self.matrix[i]]) + "\n"
        return sir

    def __str__(self):
        return self.displayOutput()

    def __repr__(self):
        return self.displayOutput()


class State:
    """
    Class used by the minimax and alpha-beta algorithms
    Has the game matrix as property
    Functions based on the condition that the Game class has JMIN and JMAX defined (the two players, human and computer)
    Also requires that the game Class has a moves() method that returns all the possible moves for a player
    """

    def __init__(self, game_matrix, current_player, depth, parent=None, estimate=None):
        self.game_matrix = game_matrix
        self.current_player = current_player

        # current game state's depth
        self.depth = depth

        # estimates how advantageous is the current state
        self.estimate = estimate

        # possible moves list from the current state
        self.moves_possible = []

        # best move for the current player
        self.chosen_state = None

    def moves(self):
        l_moves = self.game_matrix.moves(self.current_player)
        opponent = Game.opposing_player(self.current_player)
        state_moves_list = [State(move, opponent, self.depth - 1, parent=self) for move in l_moves]

        return state_moves_list

    def __str__(self):
        sir = str(self.game_matrix) + "(Current player:" + self.current_player + ")\n"
        return sir


""" MinMax Algorithm """


def min_max(state):
    Game.nodes_no = Game.nodes_no + 1
    if state.depth == 0 or state.game_matrix.final():
        state.estimate = state.game_matrix.estimate_score(state.depth)
        return state
    # compute the current state's possible moves
    state.moves_possible = state.moves()
    # apply the minimax algorithms on all possible moves
    movesWithEstimates = [min_max(move) for move in state.moves_possible]
    if state.current_player == Game.JMAX:
        # if the current player is JMAX then choose the child-state with the maximum estimate
        state.chosen_state = max(movesWithEstimates, key=lambda x: x.estimate)
    else:
        # if the current player is JMIN then choose the child-state with the minimum estimate
        state.chosen_state = min(movesWithEstimates, key=lambda x: x.estimate)
    state.estimate = state.chosen_state.estimate
    return state


def alpha_beta(alpha, beta, state):
    Game.nodes_no = Game.nodes_no + 1
    if state.depth == 0 or state.game_matrix.final():
        state.estimate = state.game_matrix.estimate_score(state.depth)
        return state

    if alpha > beta:
        return state  # it's in an invalid interval so stop processing

    state.moves_possible = state.moves()

    if state.current_player == Game.JMAX:
        current_estimate = float('-inf')

        for move in state.moves_possible:
            # calculate estimate for the new state by expanding the subtree
            new_state = alpha_beta(alpha, beta, move)

            if (current_estimate < new_state.estimate):
                state.chosen_state = new_state
                current_estimate = new_state.estimate
            if (alpha < new_state.estimate):
                alpha = new_state.estimate
                if alpha >= beta:
                    break

    elif state.current_player == Game.JMIN:
        current_estimate = float('inf')

        for move in state.moves_possible:

            new_state = alpha_beta(alpha, beta, move)

            if (current_estimate > new_state.estimate):
                state.chosen_state = new_state
                current_estimate = new_state.estimate

            if (beta > new_state.estimate):
                beta = new_state.estimate
                if alpha >= beta:
                    break
    state.estimate = state.chosen_state.estimate

    return state


def show_if_final(current_state):
    final = current_state.game_matrix.final()
    if (final):
        if (final == "draw"):
            print("Draw!")
        else:
            print(final + " has won")
            pygame.display.set_caption('Popescu Mihnea - Misere Tic Tac Toe - Winner ' + final)
        return True

    return False


def main():
    valid_response = False
    while not valid_response:
        algorithm_type = input("What algorithm do you want to use? (respond with 1 or 2)\n 1.Minimax\n 2.Alpha-beta\n ")
        if algorithm_type in ['1', '2']:
            valid_response = True
        else:
            print("You haven't chosen a valid respponse.")
    # player initialization
    valid_response = False
    while not valid_response:
        Game.JMIN = input("Do you want to play with x or 0? ").lower()
        if (Game.JMIN in ['x', '0']):
            valid_response = True
        else:
            print("Response has to be x or 0..")
    Game.JMAX = '0' if Game.JMIN == 'x' else 'x'
    # table dimensions initialization
    valid_response = False
    while not valid_response:
        try:
            N = int(input("What should be the dimension of the game table? (minimum 4 maximum 10): "))
            if N < 4 or N > 10:
                raise ValueError
            Game.NO_COLUMNS = N
            valid_response = True
        except ValueError:
            print("The response must be in the [4,10] interval!")
    # game difficulty initialization
    valid_response = False
    while not valid_response:
        difficulty = input("Choose game difficulty: (easy, medium, hard): ")
        if difficulty in ['easy', 'medium', 'hard']:
            valid_response = True
        else:
            print("The response must be one of the following: easy, medium, hard.")
    if difficulty == 'easy':
        MAX_DEPTH = 6
    if difficulty == 'medium':
        MAX_DEPTH = 7
    if difficulty == 'hard':
        MAX_DEPTH = 8

    # Save the game start time
    time_start = int(round(time.time() * 1000))

    # game matrix initialization
    current_table = Game();
    print("Initial game table")
    print(str(current_table))

    # initial state initialization
    current_state = State(current_table, 'x', MAX_DEPTH)

    # graphical interface initialization
    pygame.init()
    pygame.display.set_caption('Popescu Mihnea - Misere Tic Tac Toe')

    window_size = Game.NO_COLUMNS*100 + (Game.NO_COLUMNS - 1)*1
    screen = pygame.display.set_mode(size=(window_size, window_size))
    Game.start(screen, Game.NO_COLUMNS)

    has_to_move = False
    current_table.draw_grid()
    t_start = int(round(time.time() * 1000))
    while True:
        if (current_state.current_player == Game.JMIN):
            # the human player moves
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Show game data
                    if current_state.game_matrix.final() is False:
                        print('Game was interrupted.')
                        time_end = int(round(time.time() * 1000))
                        print("Game ran for the duration of " + str(time_end - time_start) + " ms")
                        Game.show_computer_times()
                        print("Player moved " + str(Game.JMIN_moves) + " times.")
                        print("Computer moved " + str(Game.JMAX_moves) + " times.")
                        print("Player score: " + str(current_state.game_matrix.estimate_score_player(Game.JMIN)))
                        print("Computer score: " + str(current_state.game_matrix.estimate_score_player(Game.JMAX)))

                    pygame.quit()  # close the window
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Reset the game
                        Game.JMAX_TIMES = []
                        Game.JMAX_NODES = []
                        Game.JMIN_moves = 0
                        Game.JMAX_moves = 0
                        Game.nodes_no = 0
                        t_start = int(round(time.time() * 1000))
                        time_start = int(round(time.time() * 1000))
                        print('Game was resetted!')
                        current_table = Game();
                        print("Initial table")
                        print(str(current_table))
                        current_state = State(current_table, 'x', MAX_DEPTH)
                        pygame.display.set_caption('Popescu Mihnea - Misere Tic Tac Toe')
                        has_to_move = False
                        current_table.draw_grid()

                elif event.type == pygame.MOUSEBUTTONDOWN:  # click

                    pos = pygame.mouse.get_pos()  # click's coordinates

                    for line in range(Game.NO_COLUMNS):
                        for column in range(Game.NO_COLUMNS):

                            if Game.gridCells[line][column].collidepoint(
                                    pos):  # verify if the point is inside a cell
                                ###############################
                                if not Game.can_move(current_state.game_matrix.matrix, line, column, Game.JMIN):
                                    continue
                                if current_state.game_matrix.matrix[line][column] == Game.JMIN:
                                    if (has_to_move and line == has_to_move[0] and column == has_to_move[1]):
                                        has_to_move = False
                                        current_state.game_matrix.draw_grid()
                                    else:
                                        has_to_move = (line, column)
                                        # draw the grid with the clicked cell highlighted
                                        current_state.game_matrix.draw_grid(list([has_to_move]))
                                elif current_state.game_matrix.matrix[line][column] == Game.EMPTY:
                                    if has_to_move:
                                        current_state.game_matrix.matrix[has_to_move[0]][has_to_move[1]] = Game.EMPTY
                                        has_to_move = False
                                    # place the symbol on the game matrix
                                    current_state.game_matrix.matrix[line][column] = Game.JMIN
                                    current_state.game_matrix.draw_grid()
                                    # show the game state after the player's move
                                    print("\nGame table after player's move")
                                    print(str(current_state))
                                    Game.JMIN_moves = Game.JMIN_moves + 1
                                    t_after = int(round(time.time() * 1000))
                                    print("Duration for player's move: " + str(
                                        t_after - t_start) + " miliseconds.")
                                    # verify if game is in a final state
                                    if (show_if_final(current_state)):
                                        current_state.game_matrix.show_final()
                                        time_end = int(round(time.time() * 1000))
                                        print("Game ran for the duration of " + str(time_end - time_start) + " seconds")
                                        Game.show_computer_times()
                                        print("Player moved " + str(Game.JMIN_moves) + " times.")
                                        print("Computer moved " + str(Game.JMAX_moves) + " times.")
                                        break
                                    # Change player
                                    current_state.current_player = Game.opposing_player(current_state.current_player)


        # --------------------------------
        else:
            # current player is JMAX (the computer)
            t_before = int(round(time.time() * 1000))
            Game.nodes_no = 0
            if algorithm_type == '1':
                updated_state = min_max(current_state)
            else:  # algorithm_type==2
                updated_state = alpha_beta(-500, 500, current_state)
            current_state.game_matrix = updated_state.chosen_state.game_matrix
            print("Game matrix after computer's move")
            print(str(current_state))
            print("Nodes generated: " + str(Game.nodes_no))
            Game.JMAX_moves = Game.JMAX_moves + 1

            current_state.game_matrix.draw_grid()
            
            t_after = int(round(time.time() * 1000))
            print("Computer took " + str(t_after - t_before) + " miliseconds to think.")

            Game.JMAX_TIMES.append(t_after - t_before)
            Game.JMAX_NODES.append(Game.nodes_no)

            if (show_if_final(current_state)):
                current_state.game_matrix.show_final()
                time_end = int(round(time.time() * 1000))
                print("Game took " + str(time_end - time_start) + " ms")
                Game.show_computer_times()
                print("Player moved " + str(Game.JMIN_moves) + " times.")
                print("Computer moved " + str(Game.JMAX_moves) + " times.")
                break

            # Change current player
            current_state.current_player = Game.opposing_player(current_state.current_player)
            t_start = int(round(time.time() * 1000)) # Reset time for current move

if __name__ == "__main__":
    main()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
