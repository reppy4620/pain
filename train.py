import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import functools
import pickle

from omegaconf import OmegaConf
from pathlib import Path
from typing import NamedTuple
from argparse import ArgumentParser
from tqdm import tqdm

from module.model import UNet, Discriminator
from module.dataset import load_dataset


def sparse_softmax_cross_entropy(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1)


class State(NamedTuple):
    step: int
    rng: jax.random.PRNGKey
    g_opt_state: optax.OptState
    d_opt_state: optax.OptState
    g_params: hk.Params
    d_params: hk.Params

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'step': self.step,
                'rng': self.rng,
                'g_opt_state': self.g_opt_state,
                'd_opt_state': self.d_opt_state,
                'g_params': self.g_params,
                'd_params': self.d_params
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return State(
            step=obj['step'],
            rng=obj['rng'],
            g_opt_state=obj['g_opt_state'],
            d_opt_state=obj['d_opt_state'],
            g_params=obj['g_params'],
            d_params=obj['d_params'],
        )


class Updater:
    def __init__(self, g, d, g_opt, d_opt):
        self.g: hk.Transformed = g
        self.d: hk.Transformed = d
        self.g_opt: optax.GradientTransformation = g_opt
        self.d_opt: optax.GradientTransformation = d_opt

    def g_loss(self, g_params, d_params, batch, hiddens):
        x, y = batch['x'], batch['y']
        fake = self.g.apply(g_params, x)
        fake_logits, _ = self.d.apply(d_params, fake)
        fake_probs = jax.nn.softmax(fake_logits)[:, 1]
        real_hidden, fake_hidden = hiddens[:len(hiddens)], hiddens[len(hiddens):]
        fm_loss = 0
        for r_h, f_h in zip(real_hidden, fake_hidden):
            fm_loss += jnp.abs(r_h - f_h)
        fm_loss /= len(real_hidden)
        gan_loss = jnp.mean(-jnp.log(fake_probs))
        recon_loss = jnp.mean((fake - y) ** 2)
        return gan_loss + recon_loss + fm_loss

    def d_loss(self, d_params, g_params, batch):
        x, y = batch['x'], batch['y']
        fake = self.g.apply(g_params, x)

        real_and_fake = jnp.concatenate([y, fake], axis=0)
        real_and_fake_logits, hiddens = self.d.apply(d_params, real_and_fake)
        real_logits, fake_logits = jnp.split(real_and_fake_logits, 2, axis=0)

        # Class 1 is real.
        real_labels = jnp.ones((real_logits.shape[0],), dtype=jnp.int32)
        real_loss = sparse_softmax_cross_entropy(real_logits, real_labels)

        # Class 0 is fake.
        fake_labels = jnp.zeros((fake_logits.shape[0],), dtype=jnp.int32)
        fake_loss = sparse_softmax_cross_entropy(fake_logits, fake_labels)

        return jnp.mean(real_loss + fake_loss), hiddens

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, rng, batch):
        x, y = batch['x'], batch['y']
        rng, step_rng = jax.random.split(rng)
        g_params = self.g.init(step_rng, x)
        g_opt_state = self.g_opt.init(g_params)

        rng, step_rng = jax.random.split(rng)
        inputs = jnp.concatenate([y, x], axis=0)
        d_params = self.d.init(step_rng, inputs)
        d_opt_state = self.d_opt.init(d_params)
        return State(
            step=1,
            rng=rng,
            g_opt_state=g_opt_state,
            d_opt_state=d_opt_state,
            g_params=g_params,
            d_params=d_params
        )

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: State, batch: dict):
        (loss_d, hiddens), grads = jax.value_and_grad(self.d_loss, has_aux=True)(state.d_params, state.g_params, batch)
        updates, d_opt_state = self.d_opt.update(grads, state.d_opt_state)
        d_params = optax.apply_updates(state.d_params, updates)

        loss_g, grads = jax.value_and_grad(self.g_loss)(state.g_params, state.d_params, batch, hiddens)
        updates, g_opt_state = self.g_opt.update(grads, state.g_opt_state)
        g_params = optax.apply_updates(state.g_params, updates)
        state = State(
            step=state.step + 1,
            rng=state.rng,
            g_opt_state=g_opt_state,
            d_opt_state=d_opt_state,
            g_params=g_params,
            d_params=d_params
        )
        loss_dict = dict(
            loss=loss_d + loss_g,
            loss_g=loss_g,
            loss_d=loss_d
        )
        return loss_dict, state


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train.yaml')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    g = hk.without_apply_rng(hk.transform(lambda x: UNet()(x)))
    d = hk.without_apply_rng(hk.transform(lambda x: Discriminator()(x)))

    g_opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=1e-4, b1=0.5, b2=0.9)
    )
    d_opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=1e-4, b1=0.5, b2=0.9)
    )

    ds = load_dataset(config.data_dir, batch_size=config.batch_size)
    updater = Updater(g, d, g_opt, d_opt)

    rng = jax.random.PRNGKey(config.seed)
    batch = next(ds)
    rng, step_rng = jax.random.split(rng)
    state = updater.init(rng, batch)
    ckpts = list(sorted(output_dir.glob('*.ckpt')))
    if len(ckpts) > 0:
        state = state.load(ckpts[-1])
        print(f'Loaded {state.step} checkpoint')

    print('Starting training loop')
    bar = tqdm(total=config.n_steps + 1 - int(state.step))
    for step in range(state.step, config.n_steps + 1):
        bar.set_description_str(f'Step: {step}')
        data = next(ds)
        loss_dict, state = updater.update(state, data)
        bar.update()
        bar.set_postfix_str(f'{", ".join([f"{k}: {v:.6f}" for k, v in loss_dict.items()])}')

        if (step + 1) % config.save_interval == 0:
            state.save(output_dir / f'n_{step + 1:07d}.ckpt')


if __name__ == '__main__':
    main()
