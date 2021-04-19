# common
RATE = 16000
WINDOW_TIME = 0.01
WINDOW_LENGTH = int(RATE * WINDOW_TIME)
WINDOW_HOP = WINDOW_LENGTH // 2
CHANNEL_NUMBER = 1

# model
INPUT_SIZE = 128
HIDDEN_SIZE = 128
OUT_SIZE = 2

# live
CHUNK_SIZE = 1024
MAX_QUEUE_LENGTH = 8
MODEL_FILE = "./model.0.93.97.pt"  # TODO: move to args
SENSITIVITY = 13  # TODO: move to args
INPUT_DEVICE_INDEX = 0 # TODO: move to args
DEVICE = 'cpu'

# train
EPOCH = 10
LR = 0.001
TRAIN_DATA_PERCENTAGE = 0.9 # 90%
TRAIN_KEYWORD_MARKER = 30 # check last n elements
TRAINED_MODEL_NAME = "trained model" # TODO: move to args

# data
NEG_CLEAN_ROOT = './input/neg_clean'
NEG_NOISY_ROOT = './input/neg_noisy'
POS_CLEAN_ROOT = './input/pos_clean'
POS_NOISY_ROOT = './input/pos_noisy'
NEG_RAND_ROOT =  './input/neg_random'
RANDOM_DATASET_SIZE = 2000
MIN_RANDOM_AUDIO_SIZE = RATE * 0.1
MAX_RANDOM_AUDIO_SIZE = RATE * 5
KEYWORD_MARKER_NUMBER = 23