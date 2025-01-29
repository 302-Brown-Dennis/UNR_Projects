#!/usr/bin/python3

import numpy as np
import argparse

class TicTacToe:
	def __init__(self, board=None, player=1) -> None:
		if board is None:
			self.board = self.init_board()
		else:
			self.board = board
		self.player = player
		self.depth = 9
		self.board_rows, self.board_cols = self.board.shape
		self.swapped = False

	def init_board(self):
		return np.array([[0,0,0],[0,0,0],[0,0,0]])

	def print_board(self):
		print (self.board)
		
	def check_win(self, players):
		for row in self.board:
			if np.all(row == players):
				return True
		for col in self.board.T:
			if np.all(col == players):
				return True
		if np.all(np.diag(self.board) == players) or np.all(np.diag(np.fliplr(self.board)) == players):
			return True
		return False
	
	def eval_win(self):
		if self.check_win(1):
			return 1
		elif self.check_win(-1):
			return -1
		elif self.board_full():
			return 0
		else:
			return 0
	def minimax(self, depth, is_maximizing):

		tmp_depth = depth
		if self.game_end():
			
			if self.check_win(1):
				if (depth-1 < self.depth):
					self.depth = depth
				return 1
			
			elif self.check_win(-1):
				if (depth-1 < self.depth):
					self.depth = depth
				return -1
			
			elif self.board_full():
				return 0
		
		if is_maximizing:
			best_score = -float('inf')
			
			for i in range(self.board_rows):
				for j in range(self.board_cols):
					if self.board[i, j] == 0:
						
						self.board[i, j] = self.player
						
						eval = self.minimax(depth + 1, False)
					
						self.board[i, j] = 0
						
						best_score = max(best_score,eval)

			return best_score

		else:
			best_score = float('inf')
			for i in range(self.board_rows):
				for j in range(self.board_cols):
					if self.board[i, j] == 0:
						
						self.board[i, j] = -self.player
						
						eval = self.minimax(depth + 1, True)
						
						self.board[i, j] = 0
						
						best_score = min(best_score, eval)
						
			return best_score
			
	def find_best(self):
		best_move = None
		if self.swapped:
			best_score = np.inf
		else:
			best_score = -np.inf
		
		for i in range(self.board_rows):
			for j in range(self.board_cols):
				if self.board[i, j] == 0:
					
					self.board[i, j] = self.player
					move_score = self.minimax(0, self.swapped)

					self.board[i, j] = 0
					
					if self.depth == 0:
						return i,j
					
					if ((move_score > best_score and not self.swapped) or (move_score < best_score and self.swapped)):

						best_score = move_score
						best_move = (i, j)
	
					self.depth = 9


		

		return best_move

	def board_full(self):
		if np.all(self.board != 0):
			return True
		
		return False
		
	def game_end(self):
		return self.check_win(1) or self.check_win(-1) != 0 or self.board_full()
	
	def play_game(self):

		while not self.game_end():
	
			row, col = self.find_best()
			self.board[row, col] = self.player
		
			self.player = -self.player
			self.swapped = not self.swapped
			
			
		return self.board, self.eval_win()

def load_board( filename ):
	return np.loadtxt( filename, dtype = int)

# def save_board( self, filename ):
# 	np.savetxt( filename, self.board, fmt='%d')

def main():
	parser = argparse.ArgumentParser(description='Play tic tac toe')
	parser.add_argument('-f', '--file', default=None, type=str ,help='load board from file')
	parser.add_argument('-p', '--player', default=1, type=int, choices=[1,-1] ,help='player that playes first, 1 or -1')
	args = parser.parse_args()

	board = load_board(args.file) if args.file else None
	testcase = np.array([[ 0,0,0],
                             [-1,1,0],
                             [-1,0,0]])
	
	test_arr_one = np.array([[ 1,0,0],
								 [-1,0,0],
								 [0,0,1]])
	
	test_arr_two = np.array([[ 1,0,0],
								 [-1,1,0],
								 [-1,0,-1]])

	board = np.array(board)
	ttt = TicTacToe(board, args.player)
	#ttt.print_board()
	b,p = ttt.play_game()
	print("final board: \n{}".format(b))
	print("winner: player {}".format(p))

if __name__ == '__main__':
	main()