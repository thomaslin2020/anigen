import os

AVAIL_GPUS = 1
BATCH_SIZE = 64
INITIAL_RESOLUTION = 4
FINAL_RESOLUTION = 64
NUM_WORKERS = int(os.cpu_count() / 4)
PRECISION = 32 # Set to 16 for mixed precision
FIG_SIZE = 12

ROOT_DIR = '/mnt/d/Datasets/anime-faces'
CHECKPOINT_DIR = 'checkpoints'
IMAGE_DIR = 'images'
USE_WANDB = True

Z_DIM = 64
MAP_HIDDEN_DIM = 64
W_DIM = 64
IN_CHANNELS = 64
KERNEL_SIZE = 3
HIDDEN_CHANNELS = 64
GENERATOR_LR = 1e-4
GENERATOR_MAP_LR = 1e-5
DISCRIMINATOR_LR = 4e-4
B1 = 0.5
B2 = 0.99
C_LAMBDA = 10
CRITIC_REPEATS = 3
IMAGE_CHANNELS = 3

NETWORK_NAME = 'wgan_gp_small'
NUM_EPOCHS = 100
EPOCHS_TILL_MAX_ALPHA = 16
EPOCHS_PER_RESOLUTION = 20
LOG_RESULTS_EVERY_N_STEPS = 10
TRUNCATION_VALUE = 0.7

DISABLE_PROGAN = False
IMAGES_PER_EPOCH = -1 # -1 to indicate only show images for each epoch
SAVE_MODEL = True
SAVE_IMAGES = True
SHOW_IMAGES = False
NUM_VALIDATION_IMAGES = 8
NUM_ROWS_IN_GRID = 8

assert EPOCHS_PER_RESOLUTION * 4 + 1 <= NUM_EPOCHS, f"Please train to full resolution ({FINAL_RESOLUTION} x {FINAL_RESOLUTION})!"
