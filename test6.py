import tensorflow as tf
import tensorflow_datasets as tfds

mnist_data = tfds.load("fashion_mnist")
for item in mnist_data:
    print(item)

mnist_train = tfds.load(name="fashion_mnist", split="train")
assert isinstance(mnist_train, tf.data.Dataset)
print(type(mnist_train))

for item in mnist_train.take(1):
    print(type(item))
    print(item.keys())
    print(item["image"])
    print(item["label"])

mnist_test, info = tfds.load(name="fashion_mnist", with_info="true")
print(info)

filename = "C:\\Users\\lnmcc\\tensorflow_datasets\\fashion_mnist\\3.0.1\\fashion_mnist-train.tfrecord-00000-of-00001"
raw_dataset = tf.data.TFRecordDataset(filename)
#for raw_record in raw_dataset.take(1):
    #print(repr(raw_record))

feature_description = {
    "image": tf.io.FixedLenFeature([], dtype=tf.string),
    "label": tf.io.FixedLenFeature([], dtype=tf.int64),
}


def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


parsed_dataset = raw_dataset.map(_parse_function)
for parsed_record in parsed_dataset.take(1):
    print((parsed_record))
