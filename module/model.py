import jax
import jax.numpy as jnp
import haiku as hk

from .layers import ConvNextBlock, PreNormAttention


class Discriminator(hk.Module):
    def __init__(
        self,
        dim=32,
        n_layers=3,
        channels=3
    ):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.n_layers = n_layers
        dims = [channels, *map(lambda m: dim * m, [2 ** i for i in range(self.n_layers)])]
        self.in_out = list(zip(dims[:-1], dims[1:]))
        self.channels = channels

    def __call__(self, x):
        h = list()
        for i, (in_dim, out_dim) in enumerate(self.in_out):
            x = ConvNextBlock(in_dim, out_dim, norm=i != 0)(x)
            x = ConvNextBlock(out_dim, out_dim)(x)
            h.append(x)
            if i < len(self.in_out) - 1:
                x = hk.Conv2D(out_dim, kernel_shape=4, stride=2)(x)
        x = jnp.mean(x, axis=(1, 2))
        x = hk.Linear(2)(x)
        return x, h


class UNet(hk.Module):
    def __init__(
        self,
        dim=32,
        n_layers=4,
        channels=3
    ):
        super(UNet, self).__init__()
        self.dim = dim
        self.n_layers = n_layers
        dims = [channels, *map(lambda m: dim * m, [2 ** i for i in range(self.n_layers)])]
        self.in_out = list(zip(dims[:-1], dims[1:]))
        self.mid_dim = self.in_out[-1][1]
        self.channels = channels

    def __call__(self, x):
        h = list()
        for i, (in_dim, out_dim) in enumerate(self.in_out):
            x = ConvNextBlock(in_dim, out_dim, norm=i != 0)(x)
            x = ConvNextBlock(out_dim, out_dim)(x)
            x = PreNormAttention(out_dim)(x)
            h.append(x)
            if i < len(self.in_out) - 1:
                x = hk.Conv2D(out_dim, kernel_shape=4, stride=2)(x)
        x = ConvNextBlock(self.mid_dim, self.mid_dim)(x)
        x = PreNormAttention(self.mid_dim)(x)
        x = ConvNextBlock(self.mid_dim, self.mid_dim)(x)
        for i, (in_dim, out_dim) in enumerate(reversed(self.in_out[1:])):
            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = ConvNextBlock(out_dim * 2, in_dim, norm=i != 0)(x)
            x = ConvNextBlock(in_dim, in_dim)(x)
            x = PreNormAttention(in_dim)(x)
            if i < len(self.in_out) - 1:
                x = hk.Conv2DTranspose(in_dim, kernel_shape=4, stride=2)(x)
        x = ConvNextBlock(self.dim, self.dim)(x)
        x = hk.Conv2D(self.channels, kernel_shape=1)(x)
        x = jax.nn.tanh(x)
        return x
