import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from torchvision import transforms
from math import ceil
from utils import *
import config
from pathlib import Path

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class WSConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2,
    ):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class WSLinear(nn.Module):
    def __init__(
        self, in_features, out_features, gain=2,
    ):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (gain / in_features)**0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        # initialize linear layer
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias


class MappingLayers(nn.Module):
    '''
    Mapping Layers Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        hidden_dim: the inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''
    # Use 3 for fast calculation (8 in original paper)

    def __init__(self, z_dim, hidden_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            PixelNorm(),
            WSLinear(z_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(z_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(hidden_dim, w_dim)
        )

    def forward(self, noise):
        return self.mapping(noise)


class InjectNoise(nn.Module):
    '''
    Inject Noise Class
    Values:
        channels: the number of channels the image has, a scalar
    '''

    def __init__(self, channels):
        super().__init__()
        # You use nn.Parameter so that these weights can be optimized
        self.weight = nn.Parameter(
            torch.randn(1, channels, 1, 1)
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of InjectNoise: Given an image, 
        returns the image with random noise added.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
        '''
        n_samples, _, width, height = image.size()
        noise_shape = (n_samples, 1, width, height)

        # Creates the random noise
        noise = torch.randn(noise_shape, device=image.device)
        # Applies to image after multiplying by the weight for each channel
        return image + self.weight * noise


class AdaIN(nn.Module):
    '''
    AdaIN Class
    Values:
        channels: the number of channels the image has, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''

    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale_transform = WSLinear(w_dim, channels)
        self.style_shift_transform = WSLinear(w_dim, channels)

    def forward(self, image, w):
        '''
        Function for completing a forward pass of AdaIN: Given an image and intermediate noise vector w, 
        returns the normalized image that has been scaled and shifted by the style.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]

        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image

    def get_style_scale_transform(self):
        return self.style_scale_transform

    def get_style_shift_transform(self):
        return self.style_shift_transform


class StyleGANGeneratorBlock(nn.Module):
    '''
    Micro StyleGAN Generator Block Class
    Values:
        in_chan: the number of channels in the input, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        kernel_size: the size of the convolving kernel
        starting_size: the size of the starting image
    '''

    def __init__(self, in_chan, out_chan, w_dim, kernel_size, use_upsample=True):
        super().__init__()
        # Padding is used to maintain the image size
        self.conv1 = WSConv2d(in_chan, out_chan, kernel_size, padding=1)
        self.conv2 = WSConv2d(in_chan, out_chan, kernel_size, padding=1)
        self.inject_noise1 = InjectNoise(out_chan)
        self.inject_noise2 = InjectNoise(out_chan)
        self.adain1 = AdaIN(out_chan, w_dim)
        self.adain2 = AdaIN(out_chan, w_dim)
        self.activation = nn.LeakyReLU(0.2)

    # Original model has more layers in each generator block
    def forward(self, x, w):
        '''
        Function for completing a forward pass of StyleGANGeneratorBlock: Given an x and w, 
        computes a StyleGAN generator block.
        Parameters:
            x: the input into the generator, feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        x = self.adain1(self.activation(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.activation(self.inject_noise2(self.conv2(x))), w)
        return x


class StyleGANGenerator(nn.Module):
    '''
    StyleGAN Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        map_hidden_dim: the mapping inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        in_chan: the dimension of the constant input, usually w_dim, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        kernel_size: the size of the convolving kernel
        hidden_chan: the inner dimension, a scalar
    '''

    def __init__(self, z_dim, map_hidden_dim, w_dim,
                 in_chan, kernel_size, hidden_chan, img_channels=3):
        super().__init__()
        self.map = MappingLayers(z_dim, map_hidden_dim, w_dim)

        self.starting_constant = nn.Parameter(torch.ones(1, in_chan, 4, 4))
        self.initial_block = StyleGANGeneratorBlock(
            in_chan, hidden_chan, w_dim, kernel_size)

        self.initial_rgb = WSConv2d(
            in_chan, img_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]),
        )

        for _ in range(get_num_steps(config.INITIAL_RESOLUTION, config.FINAL_RESOLUTION)):
            self.prog_blocks.append(
                StyleGANGeneratorBlock(
                    hidden_chan, hidden_chan, w_dim, kernel_size)
            )
            self.rgb_layers.append(
                WSConv2d(hidden_chan, img_channels,
                         kernel_size=1, stride=1, padding=0)
            )

    def forward(self, noise, alpha, current_layer):
        '''
        Function for completing a forward pass of StyleGANGenerator: Given noise, 
        computes a StyleGAN iteration.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''

        x = self.starting_constant
        w = self.map(noise)

        x = self.initial_block(x, w)  # initial block

        if current_layer == 0:
            return self.initial_rgb(x)

        for layer in range(current_layer):
            upscaled = F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True)
            x = self.prog_blocks[layer](upscaled, w)

        upscaled = self.rgb_layers[current_layer - 1](upscaled)
        x = self.rgb_layers[current_layer](x)
        return self.fade_in(upscaled, x, alpha)

    def interpolate(self, upscaled, generated, alpha):
        return generated * alpha + (1 - alpha) * upscaled

    def fade_in(self, upscaled, generated, alpha):
        return torch.tanh(self.interpolate(upscaled, generated, alpha))


class StyleGANDiscriminatorBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv1 = WSConv2d(in_chan, out_chan, kernel_size,
                              stride=stride, padding=padding)
        self.conv2 = WSConv2d(in_chan, out_chan, kernel_size,
                              stride=stride, padding=padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x


class StyleGANDiscriminator(nn.Module):
    '''
    Values:
        in_chan: same as the dimension of the constant inputto the generator class
        kernel_size: the size of the convolving kernel
        hidden_chan: the inner dimension, a scalar
    '''

    def __init__(self,
                 in_chan,
                 kernel_size,
                 hidden_chan,
                 img_channels=3):
        super().__init__()

        self.prog_blocks, self.rgb_layers = nn.ModuleList(
            []), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # For the rgb layer, we are doing the reverse of the genrator
        # (taking an image <img_channels> channels, and passing it through
        # the Conv2d to get an output with <hidden_chan> channels)
        for _ in range(get_num_steps(config.INITIAL_RESOLUTION, config.FINAL_RESOLUTION)):
            self.prog_blocks.append(
                StyleGANDiscriminatorBlock(
                    hidden_chan, hidden_chan, kernel_size)
            )
            self.rgb_layers.append(
                WSConv2d(img_channels, hidden_chan,
                         kernel_size=1, stride=1, padding=0)
            )

        # rgb layer for the 4x4 input size
        self.final_rgb = WSConv2d(
            img_channels, in_chan, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.final_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.final_block = nn.Sequential(
            WSConv2d(in_chan + 1, in_chan, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_chan, in_chan, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_chan, 1, kernel_size=1, padding=0, stride=1)
        )

    def forward(self, x, alpha, current_layer):
        # (current_layer = 0 means we are at the 4x4 block. = 4 means we are at the 64x64 block)
        cur_step = len(self.prog_blocks) - current_layer

        # convert from rgb as initial step, this will depend on
        # the image size (each will have it's on rgb layer)
        out = self.leaky(self.rgb_layers[cur_step](x))
        if current_layer == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        # the following two images are interpolated during training
        # downscaled image using average pooling
        downscaled = self.leaky(
            self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        # output image from the block (current one that we are training)
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(downscaled, out, alpha)  # interpolation

        # for the rest of the blocks, pass in the image and downscale it, and repeat
        for layer in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[layer](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(
                x.shape[0], 1, x.shape[2], x.shape[3])
        )
        # we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        # will get information about the variation in the batch/image
        return torch.cat([x, batch_statistics], dim=1)

    def fade_in(self, downscaled, generated, alpha):
        return generated * alpha + (1 - alpha) * downscaled


class StyleGAN(pl.LightningModule):  # TODO: Add val loop, spectral norm
    def __init__(self, z_dim, map_hidden_dim, w_dim, in_chan, kernel_size, hidden_chan, generator_lr,
                 generator_map_lr, discriminator_lr, b1, b2, c_lambda, crit_repeats=config.CRITIC_REPEATS, image_channels=config.IMAGE_CHANNELS):
        super().__init__()
        self.save_hyperparameters()
        self.generator = StyleGANGenerator(
            z_dim, map_hidden_dim, w_dim, in_chan, kernel_size, hidden_chan, image_channels)
        self.critic = StyleGANDiscriminator(in_chan, kernel_size, hidden_chan)

        self.alpha = 0
        self.current_layer = 0
        self.dm = None
        
        self.current_resolution = config.INITIAL_RESOLUTION
        self.epoch_layer_count = 0
        self.step_alpha_count = 0
        self.max_layer = get_num_steps(
            config.INITIAL_RESOLUTION, config.FINAL_RESOLUTION)

        self.validation_z = torch.randn(
            config.NUM_VALIDATION_IMAGES, hidden_chan)
        
        self.steps_per_epoch = 0
        self.steps_till_max_alpha = 0
        self.alpha_step = 0

        path = Path(f'{config.IMAGE_DIR}/{config.NETWORK_NAME}')
        path.mkdir(parents=True, exist_ok=True)
    
    def set_datamodule(self, dm):
        self.dm = dm
        total_images = len(self.dm)
        self.steps_per_epoch = ceil(total_images / config.BATCH_SIZE)
        self.steps_till_max_alpha = self.steps_per_epoch * config.EPOCHS_TILL_MAX_ALPHA
        self.alpha_step = 1 / self.steps_till_max_alpha
        assert config.EPOCHS_PER_RESOLUTION * \
            self.steps_per_epoch >= self.steps_till_max_alpha, "Increase alpha to 1 before moving on to next resolution"

    def forward(self, z):
        return self.generator(z, self.alpha, self.current_layer)

    def get_crit_loss(self, fake_pred, real_pred, gp, c_lambda):
        return - real_pred.mean() + fake_pred.mean() + c_lambda * gp

    def get_gen_loss(self, crit_fake_pred):
        return - crit_fake_pred.float().mean()

    def training_step(self, batch, batch_idx, optimizer_idx):
        z = torch.randn(batch.size(0), self.hparams.z_dim).type_as(batch)
        if optimizer_idx < self.hparams.crit_repeats:  # train discriminator
            fake = self(z)
            crit_fake_pred = self.critic(
                fake.detach(), self.alpha, self.current_layer)
            crit_real_pred = self.critic(batch, self.alpha, self.current_layer)
            # gradient penalty
            epsilon = torch.rand(len(batch), 1, 1, 1,
                                 requires_grad=True).type_as(batch)
            gradient = get_gradient(self.critic, batch, fake.detach(
            ), epsilon, self.alpha, self.current_layer)
            gp = gradient_penalty(gradient)
            crit_loss = self.get_crit_loss(
                crit_fake_pred, crit_real_pred, gp, self.hparams.c_lambda)

            if optimizer_idx == self.hparams.crit_repeats - 1:  # log last critic loss
                self.log('crit_loss', crit_loss)

            return crit_loss

        else:  # train generator
            fake = self(z)
            crit_fake_pred = self.critic(fake, self.alpha, self.current_layer)
            gen_loss = self.get_gen_loss(crit_fake_pred)
            if config.STEPS_PER_IMAGE > 0:
                if batch_idx % config.STEPS_PER_IMAGE == 0:
                    show_tensor_images(fake.detach(), config.NUM_ROWS_IN_GRID,
                           image_name=get_image_name(f'{self.current_epoch}-{batch_idx}'))
            self.log('gen_loss', gen_loss)
            return gen_loss

    def configure_optimizers(self):  # Add LR Scheduler here
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_gen = optim.Adam([{"params": [param for name, param in self.generator.named_parameters() if "map" not in name]},
                              {"params": self.generator.map.parameters(), "lr": self.hparams.generator_map_lr}],
                             lr=self.hparams.generator_lr, betas=(b1, b2))
        opt_critic = optim.Adam(self.critic.parameters(
        ), lr=self.hparams.discriminator_lr, betas=(b1, b2))

        return ([opt_critic] * self.hparams.crit_repeats) + [opt_gen]

    def on_epoch_end(self):
        z = self.validation_z.to(self.device)
        sample_images = self(z)
        show_tensor_images(sample_images, config.NUM_VALIDATION_IMAGES,
                           image_name=get_image_name(self.current_epoch))
        # grid = make_grid(sample_images)
        # self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

    def get_resolution(self, initial_resolution):
        return initial_resolution * 2 ** self.current_layer

    def change_resolution(self, resolution):
        # Trick to change the resolution during training
        mean, std = self.dm.get_data_mean_std()
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.Normalize(mean, std)
        ])
        self.dm.set_transform(transform)

    def on_train_epoch_start(self):
        if self.update_layer():
            self.change_resolution(self.get_resolution(config.INITIAL_RESOLUTION))
        print(
            f'Now training in {self.current_resolution} x {self.current_resolution} resolution')

    def on_train_epoch_end(self):
        self.epoch_layer_count += 1

    def on_train_batch_start(self, batch, batch_idx):
        self.update_alpha()

    def update_alpha(self):
        if self.step_alpha_count < self.steps_till_max_alpha:
            self.step_alpha_count += 1
            self.alpha = min(
                round(self.alpha + self.alpha_step, 8), 1.0)

    def update_layer(self):
        if self.current_layer == self.max_layer:
            return False  # skip layer updates (already in full resolution)

        if self.epoch_layer_count == config.EPOCHS_PER_RESOLUTION:
            self.current_layer += 1
            self.epoch_layer_count = 0
            self.step_alpha_count = 0
            self.alpha = 0
        return True  # updated layer
