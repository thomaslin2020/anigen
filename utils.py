import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
from math import log2
from scipy.stats import truncnorm
import config


def get_image_name(image_name, step = None):
    if step:
            return f'{config.IMAGE_DIR}/{config.NETWORK_NAME}/{image_name}-{step}.png'
    return f'{config.IMAGE_DIR}/{config.NETWORK_NAME}/{image_name}.png'

def show_tensor_images(image_tensor, rows, image_name=None, stats=None, device='cpu'):
    size = image_tensor.shape[1:]
    num_images = image_tensor.shape[0]
    if stats is None: # Image has not been normalizled
        image_tensor = (image_tensor + 1) / 2
    else: # Image normalized with mean, std
        mean, std = stats
        image_tensor = torch.tensor(std).view(size[0], 1, 1).to(
            device) * image_tensor + torch.tensor(mean).view(size[0], 1, 1).to(device)

    image_unflat = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = make_grid(image_unflat[:num_images], nrow=rows, padding=0)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    if config.SAVE_IMAGES:
        plt.savefig(image_name)
    if config.SHOW_IMAGES:
        plt.show()


def get_num_steps(initial_resolution, full_resolution):
    assert log2(full_resolution) == int(log2(full_resolution)
                                        ), "Please use a power of 2 as the resolution"
    return int(log2(full_resolution / initial_resolution))


def get_truncated_noise(n_samples, z_dim, truncation):
    '''
    Function for creating truncated noise vectors: Given the dimensions (n_samples, z_dim)
    and truncation value, creates a tensor of that shape filled with random
    numbers from the truncated normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        truncation: the truncation value, a non-negative scalar
    '''
    truncated_noise = truncnorm.rvs(-truncation,
                                    truncation, size=(n_samples, z_dim))
    return torch.Tensor(truncated_noise)


def get_gradient(crit, real, fake, epsilon, alpha, current_layer):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images, alpha, current_layer)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

def generate_images(model, n_samples, rows, device, image_name = None, truncation=config.TRUNCATION_VALUE):
    z = get_truncated_noise(n_samples, config.Z_DIM, truncation).to(device)
    image_tensor = model(z)
    show_tensor_images(image_tensor, rows, image_name = image_name, device = device)
