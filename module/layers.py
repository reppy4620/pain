import jax
import jax.numpy as jnp
import haiku as hk

from einops import rearrange


class ConvNextBlock(hk.Module):
    def __init__(self, dim, dim_out, mult=2, norm=True):
        super(ConvNextBlock, self).__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.mult = mult
        self.norm = norm

    def __call__(self, x):
        h = hk.Conv2D(self.dim, kernel_shape=7, feature_group_count=self.dim)(x)

        if self.norm:
            h = hk.LayerNorm(axis=-1, param_axis=-1,
                             create_scale=True, create_offset=True)(h)
        h = hk.Conv2D(self.dim_out * self.mult, kernel_shape=3)(h)
        h = jax.nn.gelu(h)
        h = hk.Conv2D(self.dim_out, kernel_shape=3)(h)
        if self.dim != self.dim_out:
            x = hk.Conv2D(self.dim_out, kernel_shape=1)(x)
        return h + x


class LinearAttention(hk.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.dim = dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads

    def __call__(self, x):
        b, h, w, c = x.shape
        qkv = hk.Conv2D(self.hidden_dim * 3, kernel_shape=1, with_bias=False)(x)
        q, k, v = jnp.split(qkv, indices_or_sections=3, axis=-1)
        q = rearrange(q, 'b y x (h c) -> b h c (y x)', h=self.heads)
        k = rearrange(k, 'b y x (h c) -> b h c (y x)', h=self.heads)
        v = rearrange(v, 'b y x (h c) -> b h c (y x)', h=self.heads)

        k = jax.nn.softmax(k)
        context = jnp.einsum('b h d n, b h e n -> b h d e', k, v)
        out = jnp.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b y x (h c)', h=self.heads, x=h, y=w)
        out = hk.Conv2D(self.dim, kernel_shape=1)(out)
        return out


class PreNormAttention(hk.Module):
    def __init__(self, dim):
        super(PreNormAttention, self).__init__()
        self.dim = dim

    def __call__(self, x):
        y = hk.LayerNorm(axis=-1, param_axis=-1,
                         create_scale=True, create_offset=True)(x)
        y = LinearAttention(self.dim)(y)
        return x + y
