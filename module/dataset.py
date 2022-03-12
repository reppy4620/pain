import random
import tensorflow as tf
from pathlib import Path
from typing import Union, Tuple

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_dataset(data_dir: Union[Path, str], batch_size: int, crop_size: Tuple[int] = (256, 256)):
    def preprocess(path: str):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False, dtype=tf.float32)
        img = 2 * img - 1
        color, gray = tf.split(img, num_or_size_splits=2, axis=-2)
        box = tf.random.uniform(shape=(1, 4))
        gray = tf.image.crop_and_resize(gray[None, ...], boxes=box, box_indices=(0,), crop_size=crop_size)[0]
        color = tf.image.crop_and_resize(color[None, ...], boxes=box, box_indices=(0,), crop_size=crop_size)[0]
        data = dict(
            x=gray,
            y=color,
        )
        return data

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    file_path_list = list(sorted(data_dir.glob('*')))
    file_path_list = [str(p) for p in file_path_list]
    ds = tf.data.Dataset.from_tensor_slices(file_path_list)
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(buffer_size=1024)
    ds = ds.cache()
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds.as_numpy_iterator()
