import numpy as np
import pandas as pd
from tqdm import tqdm
from Solver import Board, Solver

game = '''
          0 0 0 7 0 0 0 9 6
          0 0 3 0 6 9 1 7 8
          0 0 7 2 0 0 5 0 0
          0 7 5 0 0 0 0 0 0
          9 0 1 0 0 0 3 0 0
          0 0 0 0 0 0 0 0 0
          0 0 9 0 0 0 0 0 1
          3 1 8 0 2 0 4 0 7
          2 4 0 0 0 5 0 0 0
      '''
game = game.strip().split("\n")
board = []
for i in game:
    t = i.replace(' ','').strip()
    t = list(t)
    t = list(map(int,t))
    board.append(t)

brd = Board(board)
print(brd)


# data = pd.read_csv("sudoku.csv")

# val_set = data.iloc[:1000]

# quiz_list = list(val_set['quizzes'])
# print(quiz_list[0])
# sol_list = list(val_set['solutions'])
# print(sol_list[0])
# val_quiz = []
# val_sol = []
# for i,j in tqdm(zip(quiz_list,sol_list)):
#     q = np.array(list(map(int,list(i)))).reshape(9,9)
#     s = np.array(list(map(int,list(j)))).reshape(9,9)
#     val_quiz.append(q)
#     val_sol.append(s)

# count = 0
# for i,j in tqdm(zip(val_quiz,val_sol)):
#     if solve(i):
#         if (i==j).all():
#             count+=1
#     else:
#         pass
    
# print("{}/1000 solved!! That's {}% accuracy.\n".format(count,(count/1000.0)*100))



