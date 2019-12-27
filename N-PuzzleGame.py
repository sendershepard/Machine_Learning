""" Sender Shepard. Creating a N-Puzzle board with a matrix of n*n
    which uses informed search algorithms, A* AND Best First Search, in order
    to arrive at the optimal solution without doing unecessary traverals.
    To determine the optimal path, a function f(n) is created and assigned
    to nodes in a priority queue, seeking to search the first best nodes to
    explore and determine if that path provides a solution. """

import random
import math


class Puzzle(object):
    """ Creates a new instance of the class Puzzle, given an input state.
        State is a list of lenght (width * height) and must contain all
        numbers between 1, (width * height)-1. Width and height must be > 1 """

    
    def __init__(self, width, height, state):
        """The input obtained will initialize the variables of the class"""
        assert width > 1
        assert height > 1
        assert math.sqrt(width * height) == (width or height)
        self._width = width
        self._height = height
        self._blank = 'b'
        self.matrix_board = []
        self.parent = None #need to move to a different class
        self.hval = 0
        self.depth = 0
        
        """Initializing the matrix board"""
        for rowz in range(height):
            row = state[:width]
            state = state[width:]
            self.matrix_board.append(row)
        self._goal_state = [row[:] for row in self.matrix_board]


    def __eq__(self, other):
        """This class is needed as it allows to check for custom equalities
            to check against the objects with equivalent contents. """
        if self._width != other._width:
            return False
        elif self._height != other._height:
            return False
        for i in range(self._height):
            if self.matrix_board[i] != other.matrix_board[i]:
                return False

        return True


    def __ne__(self, other):
        """This class does anything but return the opposite
            of the equivalent function. """
        return not self.__eq__(other)


    @property
    def height(self):
        """Function that returns the protected height. """
        return self._height


    @property
    def width(self):
        """Function that retruns the protected width. """
        return self._width

    
    @property 
    def goal_state(self):
        """Function returns the protected goal state"""
        return self._goal_state


    def print_board(self):
        """This funcktion prints the board in a matrix format"""
        for i in range(self._width):
            print(self.matrix_board[i], '\t')

        return None


    def copy_puzzle(self):
        """Performs a deep copy of self"""
        def _copy():
            """Protected function, passing emtpy list"""
            copy = Puzzle(self._width, self._height, [])
            return copy
        
        puzzle_copy = _copy()
        puzzle_copy.matrix_board = [row[:] for row in self.matrix_board]

        return puzzle_copy
    
    
    def shuffle_board(self):
        """ This fuction will shuffle the board's contents through
            the moves that are available and thus playing, fair. """
        for i in range(self._width * self._height):
            row, column = self.find_value('b')
            free_spots = self.available_moves()
            self.swap_values((row, column), random.choice(free_spots))
        
        return None


    def find_value(self, value):
        """Finds the row, column index of the blank in the board"""
        for row in range(self._width):
            for column in range (self._height):
                if self.matrix_board[row][column] == value:
                    return row, column
                
                
    def get_value(self, row, column):
        """Returns the board's value at row/column"""
        return self.matrix_board[row][column]


    def set_value(self, row, column, value):
        """Assigns the value on the board on row/column"""
        self.matrix_board[row][column] = value

        return None


    def swap_values(self, x, y): 
        """Swaps values"""
        temp = self.get_value(*x) # Splat (*) new takes list, creates flat tuple      
        self.set_value(x[0], x[1], self.get_value(*y))
        self.set_value(y[0], y[1], temp)
        
        return None


    def available_moves(self):
        """Returns a list of the possible moves around the blank."""
        row, column = self.find_value(self._blank) 
        open_spots = []

        if row > self._width - self._height:
            open_spots.append((row - 1, column))

        if column > self._height - self._width:
            open_spots.append((row, column - 1))

        if row < self._width - 1:
            open_spots.append((row + 1, column)) 

        if column < self._height - 1:
            open_spots.append((row, column + 1))
        
        return open_spots


    def iterable_moves(self):
        """ Performs all the available moves and returns a list with the
            iteration each move, attributes that are copied as well."""
        open_spots, blank = self.available_moves(), self.find_value('b')

        return [self._move_attributes(blank, x) for x in open_spots]


    def _move_attributes(self, blank, iterable):
        """ Protected move constructor and move helper function.
            It creates a class object, copies attributes and swaps values."""
        copie = self.copy_puzzle()
        copie.swap_values(blank, iterable)
        copie.depth = self.depth + 1
        copie.parent = self

        return copie


class A_Star(object):
    """ A* picks a node at each step according to a value f.
        At each step it picks the node with the lowest value f"""

    
    def __init__(self):
        self = self

        
    def reconstruct_path(self, state, came_from):
        """ A path is reconstructed to show from where the solution came from
            by reversing the path we can return the path in sorted order. """
        came_from.append(state)

        if state.parent is None:
            came_from.reverse()
            return came_from
        else:
            return self.reconstruct_path(state.parent, came_from)


    def A_Star_Algorithm(self, start, heuristic):
        """ A* Algorithm is an informed search algo that uses the a formula
            f(n) to decide which path is the best to take based on lowest f(n)
            f(n) = g(n) + h(n), where n is the last seen node in the path,
            g(n) is the cost of the path that comes from the start node to n,
            h(n) = is a heuristic that estimates the cost of the cheapest path
            from n to the target node"""
        open_list = [start] #list of currently discovered nodes not evaluated yet
        closed_list = [] #list of nodes already evaluated
        counter = 0 #initializing a counter to track number of attempts

        while open_list: #while the queue is not empty
            current_state = open_list.pop(0) #priority_queue, the node in the open_list having the lowest f score
            counter = counter + 1 #Adding one to counter, to keep track of moves

            if current_state.matrix_board == start.goal_state:
                return self.reconstruct_path(current_state, []), counter

            successive_move = current_state.iterable_moves()
            index_closed = index_open = -1
            
            for current_move in successive_move:
                try:
                    index_open = open_list.index(current_move)
                except ValueError:
                    index_open = -1
                try:
                    index_closed = closed_list.index(current_move)
                except ValueError:
                    index_closed = -1

                #The distance from start to the neighbor f(n) = g(n) + h(n)
                h_value = heuristic_calculation(current_move, heuristic)
                g_value = current_move.depth
                f_value = g_value + h_value
            
                if index_closed == -1 and index_open == -1: #if neighbor in closed_list:
                    current_move.hval = h_value
                    open_list.append(current_move)

            closed_list.append(current_state)
                
            open_list = sorted(open_list, key=lambda node: node.hval + node.depth)
            
        return 0 #return 0 for no found solution


class Best_First_Search(object):
    """ The Best First Search Agorithm uses a function f(n) in order to decide
        which adjacent node is the best suited to explore.
        Best First Search is a kind of Heuristic Search or Informed Search. """

    def __init__(self):
        self = self

    def rec_path(self, state, came_from):
        """ A path is reconstructed to show from where the solution came from
            by reversing the path we can return the path in sorted order. """
        came_from.append(state)

        if state.parent is None:
            came_from.reverse()
            return came_from
        else:
            return self.rec_path(state.parent, came_from)
        
    def BFS_Algorithm(self, start, heuristic):
        """ The BFS uses a priority queue to store nodes. """
        p_queue = [start]
        visited = []
        counter = 0

        while p_queue:
            current_state = p_queue.pop(0) #priority_queue
            counter = counter + 1

            if current_state.matrix_board == start.goal_state:
                return self.rec_path(current_state, []), counter

            successive_move = current_state.iterable_moves()
            for current_move in successive_move:
                target_index = 0
                for i in range(1, len(successive_move)):
                    #f_value = g_value + h_value
                    q_node = heuristic_calculation(successive_move[i], heuristic) + successive_move[i].depth
                    target_node = heuristic_calculation(successive_move[target_index], heuristic) + successive_move[target_index].depth
                    if q_node < target_node:
                        target_index = i #finding the best node to check first
        
                best = successive_move[target_index]
                if best not in visited:
                    p_queue.append(current_move)

            visited.append(current_state)
            p_queue = sorted(p_queue, key=lambda node: node.hval + node.depth)
                        
        return 0 #return 0 for no found solution


def heuristic_calculation(Puzzle, heuristic_kind):
    """This function provides a value on the puzzle state to help inform
        the algorithm of which moves are best based on lowest cost. """
    zero = Puzzle.find_value('b') #Let's give blank the value of 0 to do math
    Puzzle.set_value(*zero, 0) #Calculate the weight of blank as zero
    total_distance = 0 #Value given for total distance from goal state
    x = Puzzle.height #Column size
    y = Puzzle.width #Row size
    
    """Going through the whole matrix to give it a value to the state. """
    for row in range(y): #need to account for larger matrices 
        for column in range(x): #need to account for larger matrices
            val = Puzzle.get_value(row, column) - 1 #-1 to find the blank in values
            column_goal = val % x #Puzzle.height a 3x3 matrix
            row_goal = val / y #Puzzle.width could be a 4x4 matrix

            if row_goal < 0: #if we found the blank
                row_goal = 2 #row's goal state of blank

            if heuristic_kind == manhattan:
                total_distance += manhattan(row, row_goal, column, column_goal)
            if heuristic_kind == diagonal:
                total_distance += manhattan(row, row_goal, column, column_goal)
            if heuristic_kind == euclidean:
                total_distance += manhattan(row, row_goal, column, column_goal)

    blank = Puzzle.find_value(0)
    Puzzle.set_value(*blank, 'b')

    return total_distance


def manhattan(row, row_goal, column, column_goal):
    """Manhattan Heuristic"""
    return (abs(row - row_goal) + abs(column - column_goal))


def diagonal(row, row_goal, column, column_goal):
    """Diagonal Distance Heuristic"""
    x = abs(row - row_goal)
    y = abs(column - column_goal)
    
    return ((x + y ) * min(x,y))


def euclidean(row, row_goal, column, column_goal):
    """Euclidean Distance Heuristic"""
    x = abs(row - row_goal)
    y = abs(column - column_goal)

    return(math.sqrt(x*x + y*y))


class Puzzle_Main(object):
    """This class creates a board of matrix size width * height and
        plays the game until solved using the A* and Best-First Search"""
    def __init__(self):
        self.astar = A_Star()
        self.bfs = Best_First_Search()

    def run(self):
        print(" Welcome to the N-Puzzle Game ")
        user_choice = Board_Size_Prompt(prompt = "Please pick board size: ")
        if user_choice == "1":
            print("8 N")
            self.game = Puzzle(3, 3, [1,2,3,4,5,6,7,8,'b'])
        if user_choice == "2":
            print("15 N")
            self.game = Puzzle(4, 4, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,'b'])

        self.game.shuffle_board()

        print("For A* heuristic: ")
        print("Manhattan Distance: ")
        path, steps = self.astar.A_Star_Algorithm(self.game, manhattan)
        print("Manhattan Steps = ", steps)
        print("Manhattan Path:")
        for state in path:
            state.print_board()
            print()
            
        print("Diagonal Distance: ")
        path, steps = self.astar.A_Star_Algorithm(self.game, diagonal)
        print("Steps = ", steps)
        print("Diagonal Path:")
        for state in path:
            state.print_board()
            print()
            
        print("Euclidean Distance: ")
        path, steps = self.astar.A_Star_Algorithm(self.game, euclidean)
        print("Steps = ", steps)
        print("Euclidean Path:")
        for state in path:
            state.print_board()
            print()
  
        print("For Best First Search heuristic: ")
        print("Manhattan Distance: ")
        path, steps = self.bfs.BFS_Algorithm(self.game, manhattan)
        print("Manhattan Steps = ", steps)
        print("Manhattan Path:")
        for state in path:
            state.print_board()
            print()

        print("Diagonal Distance: ")
        path, steps = self.bfs.BFS_Algorithm(self.game, diagonal)
        print("Steps = ", steps)
        print("Diagonal Path:")
        for state in path:
            state.print_board()
            print()
        
        print("Euclidean Distance: ")
        path, steps = self.bfs.BFS_Algorithm(self.game, euclidean)
        print("Steps = ", steps)
        print("Euclidean Path:")
        for state in path:
            state.print_board()
            print()


def Prompt_User(prompt):
    """Helper function to prompt the user to pick a menu entry. """
    print("\nPlease pick from the Menu below: ")
    print("\nFor Manhattan Heuristic enter: 1")
    print("For Diagonal Heuristic enter: 2")
    print("For Euclidean Heuristic 3\n")
    reminder = "Input must match the options!"
    entries=["1", "2" ,"3"]
    while True:
        user_answer = input(prompt)
        if user_answer in entries:
            return user_answer
        else:
            print(reminder)
            return False   


def Board_Size_Prompt(prompt):
    """Helper function to prompt the user to pick a menu entry. """
    print("\nPlease pick from the Menu below: ")
    print("\nFor a 8-Puzzle Game enter: 1")
    print("For a 15-Puzzle Game enter: 2")
    reminder = "Input must match the options!"
    entries=["1", "2"]
    while True:
        user_answer = input(prompt)
        if user_answer in entries:
            return user_answer
        else:
            print(reminder)
            return False 


""" Main """
if __name__ == '__main__':

    if True:
        main = Puzzle_Main()
        main.run()
