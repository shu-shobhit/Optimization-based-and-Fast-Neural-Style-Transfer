import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE_VALUE = 512
NUM_STEPS = 75
LEARNING_RATE = 1
ALPHA = 1
BETA = 1000
OPTIMIZER = "LBFGS"

CONTENT_LAYER_NAME = "11"
STYLE_LAYERS_LIST = [str(i) for i in [1, 6, 11, 20, 29]] 
STYLE_LAYER_WEIGHTS = [1e3 / n**2 for n in [64,128,256,512,512,]]
