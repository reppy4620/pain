import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE


def test1():
    path = './dataset/gray_color_0.jpg'
    crop_size = (512, 512)

    def preprocess(path: str):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False, dtype=tf.float32)
        color, gray = tf.split(img, num_or_size_splits=2, axis=-2)
        box = tf.random.uniform(shape=(1, 4))
        gray = tf.image.crop_and_resize(gray[None, ...], boxes=box, box_indices=(0,), crop_size=crop_size)[0]
        color = tf.image.crop_and_resize(color[None, ...], boxes=box, box_indices=(0,), crop_size=crop_size)[0]
        data = dict(
            x=gray,
            y=color,
        )
        return data

    data = preprocess(path)
    plt.subplot(1, 2, 1)
    plt.imshow(np.asarray(data['x']), aspect='auto')
    plt.subplot(1, 2, 2)
    plt.imshow(np.asarray(data['y']), aspect='auto')
    plt.show()


if __name__ == '__main__':
    test1()
