import torch
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from model import StyleGAN
from data import AnimeFaceDataModule
import pytorch_lightning as pl
from utils import generate_images
import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.rcParams["figure.figsize"] = (config.FIG_SIZE, config.FIG_SIZE)

dm = AnimeFaceDataModule()

model_config = {
    "z_dim": config.Z_DIM,
    "map_hidden_dim": config.MAP_HIDDEN_DIM,
    "w_dim": config.W_DIM,
    "in_chan": config.IN_CHANNELS,
    "kernel_size": config.KERNEL_SIZE,
    "hidden_chan": config.HIDDEN_CHANNELS,
    "generator_lr": config.GENERATOR_LR,
    "generator_map_lr": config.GENERATOR_MAP_LR,
    "discriminator_lr": config.DISCRIMINATOR_LR,
    "b1": config.B1,
    "b2": config.B2,
    "c_lambda": config.C_LAMBDA
}

model = StyleGAN(**model_config)
model.set_datamodule(dm)

wandb_logger = WandbLogger() if config.USE_WANDB else True # Use default loader

trainer = pl.Trainer(max_epochs = config.NUM_EPOCHS, gpus=config.AVAIL_GPUS, precision=config.PRECISION, \
    logger=wandb_logger, log_every_n_steps=config.LOG_RESULTS_EVERY_N_STEPS, default_root_dir=config.CHECKPOINT_DIR)
trainer.fit(model, dm)


generate_images(model.to(device), 25, 5, device)
