import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Dense, Flatten

def get_model():
    x_in = keras.Input(shape=(100))
    #x_out = Flatten()(x_in)
    x_out = x_in
    x_out = Dense(10, activation='softmax')(x_out)
    return tf.keras.Model(inputs=x_in, outputs=x_out)

if __name__ == "__main__":
    model=get_model()

    print(model.summary())