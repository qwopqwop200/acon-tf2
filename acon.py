import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)
    
class AconC(layers.Layer):
    def __init__(self):
        super().__init__()
        
    def build(self,shape):
        self.p1 = self.add_weight(shape=(1, 1, 1, shape[-1]),initializer='uniform',trainable=True)
        self.p2 = self.add_weight(shape=(1, 1, 1, shape[-1]),initializer='uniform',trainable=True)
        self.beta = self.add_weight(shape=(1, 1, 1, shape[-1]),initializer='ones',trainable=True)
        
    def call(self, x):
        return (self.p1 * x - self.p2 * x) * keras.activations.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

class MetaAconC(layers.Layer):
    def __init__(self, r=16):
        super().__init__()
        self.r = r
        
    def build(self,shape):
        self.fc1 = layers.Conv2D(max(self.r, shape[-1] // self.r), 1)
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Conv2D(shape[-1],1)
        self.bn2 = layers.BatchNormalization()
        self.p1 = self.add_weight(shape=(1, 1, 1, shape[-1]),initializer='uniform',trainable=True)
        self.p2 = self.add_weight(shape=(1, 1, 1, shape[-1]),initializer='uniform',trainable=True)

    def call(self, x):
        beta = keras.activations.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(tf.reduce_mean(tf.reduce_mean(x,axis=1, keepdims=True),axis=2, keepdims=True))))))
        return (self.p1 * x - self.p2 * x) * keras.activations.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x
      
def test():
  img = tf.ones(2,128,128,32)
  act = AconC()
  act(img)
  act = MetaAconC()
  act(img)
