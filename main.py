from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku.puzzle import extract_digits, find_puzzle
from sudoku.Board import Board
from sudoku.Solver import Solver
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained classifier")
ap.add_argument("-i", "--image", required=True,
                help="path to Sudoku puzzle image")
ap.add_argument("-d", "--debug", default=False, type=bool,
                help="Display the image after every step of pipelining.")
args = vars(ap.parse_args())

# load the digit classifier from disk
if args['debug']: 
    print("[INFO] loading digit classifier...")
model = load_model(args['model'])
if args['debug']: 
    print("[INFO] model loaded...")

# load the input image from disk and resize it
if args['debug']: 
    print("[INFO] processing image...")
image = cv2.imread(args["image"])
scale_percent = 60  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
if args['debug']: 
    print("[INFO] image loaded...")

# find the puzzle in the image and then
if args['debug']: 
    print("[INFO] finding puzzle...")
(puzzleImage, warped) = find_puzzle(image, debug=args["debug"])

# initialize the board

board = np.zeros((9, 9), dtype='int')

step_x = warped.shape[1] // 9
step_y = warped.shape[0] // 9

# storing cell locations
celllocs = []
for y in range(0, 9):
    row = []
    for x in range(0, 9):
        startx = x * step_x
        starty = y * step_y
        endx = (x + 1) * step_x
        endy = (y + 1) * step_y

        row.append((startx, starty, endx, endy))
        cell = warped[starty:endy, startx:endx]
        digit = extract_digits(cell, debug=args['debug'])

        if digit is not None:
            roi = cv2.resize(digit, (28, 28))
            roi = roi.astype("float") / 255.
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            pred = model.predict(roi).argmax(axis=1)
            board[y, x] = pred
    celllocs.append(row)

if args['debug']: 
    print("[INFO] OCR'd Sudoku board:")
puzzle = Board(grid=board)
solver = Solver(puzzle)
if args['debug']:
    print(puzzle)

for (cellrow, boardrow) in zip(celllocs, solver._board._board):

    for (box, digit) in zip(cellrow, boardrow):
        startx, starty, endx, endy = box

        # compute coordinates where the text will be put on the image
        textx = int((endx-startx)*0.33)
        texty = int((endy-starty)*-0.2)
        textx+=startx
        texty+=endy
        # draw result in the image using coordinates calculated above
        cv2.putText(puzzleImage, str(digit), (textx, texty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

cv2.imshow("Sudoku Result", puzzleImage)
cv2.waitKey(0)

