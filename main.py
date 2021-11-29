from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku.puzzle import extract_digits, find_puzzle
from sudoku import Board, Solver
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained classifier")
ap.add_argument("-i", "--image", required=True,
                help="path to Sudoku puzzle image")
ap.add_argument("-d", "--debug", default=True, type=bool,
                help="Display the image after every step of pipelining.")
args = vars(ap.parse_args())

# load the digit classifier from disk
print("[INFO] loading digit classifier...")
model = load_model(args["model"])
print("[INFO] model loaded...")

# load the input image from disk and resize it
print("[INFO] processing image...")
image = cv2.imread(args["image"])
scale_percent = 60 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# find the puzzle in the image and then
(puzzleImage, warped[np.ndarray]) = find_puzzle(image, debug=args["debug"] > 0)

# initialize the board
board = np.zeros((9, 9), dtype='int')

step_x = warped.shape()