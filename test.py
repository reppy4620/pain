import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def test1():
    path = './dataset/gray_color_0.jpg'
    crop_size = (512, 512)

    def preprocess(path: str):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False, dtype=tf.float32)
        color, gray = tf.split(img, num_or_size_splits=2, axis=-2)
        box = tf.random.uniform(shape=(1, 4))
        gray = tf.image.crop_and_resize(gray[None, ...], boxes=box, box_indices=(0,), crop_size=crop_size)
        color = tf.image.crop_and_resize(color[None, ...], boxes=box, box_indices=(0,), crop_size=crop_size)
        data = dict(
            x=gray,
            y=color,
        )
        return data

    print(preprocess(path))


if __name__ == '__main__':
    test1()
