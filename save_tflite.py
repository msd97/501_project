import tensorflow as tf
from keras.models import load_model

m_path = './weights/weights_ResNet18_drop_200.hdf5'
model = load_model(m_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('resnet18_artists.tflite', 'wb') as f:
  f.write(tflite_model)
