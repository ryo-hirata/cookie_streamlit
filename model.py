import matplotlib.pyplot as plt
import torch
import cv2
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, LeakyReLU
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from natsort import natsorted
import glob
import numpy as np
from keras.models import load_model

# 学習データの読み込み＆前処理
train_images = sorted(glob.glob('/Users/hirataryo/Desktop/cookie_app/test_resized_400x300_images/*.jpg'))
train = []
for i in train_images:
    image = cv2.imread(i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    train.append(image)

train = np.array(train)
train = train.astype('float32') / 255

# 学習用ハイパーパラメータ
LEARNING_RATE = 0.0005
BATCH_SIZE = 8
Z_DIM = 50
EPOCHS = 1

# エンコーダ
encoder_input = Input(shape=(300,400,3), name='encoder_input')
x = encoder_input
x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', name='encoder_conv_0')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', name='encoder_conv_0_1')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', name='encoder_conv_1')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', name='encoder_conv_2')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='encoder_conv_3')(x)
x = LeakyReLU()(x)
shape_before_flattening = K.int_shape(x)[1:]
x = Flatten()(x)
encoder_output = Dense(Z_DIM, name='encoder_output')(x)
encoder = Model(encoder_input, encoder_output)

# デコーダ
decoder_input = Input(shape=(Z_DIM,), name='decoder_input')
x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same', name='decoder_conv_t_0')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_1')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_2')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='same', name='decoder_conv_t_2_5')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', name='decoder_conv_t_3')(x)
x = Activation('sigmoid')(x)
decoder_output = x
decoder = Model(decoder_input, decoder_output)

# エンコーダ/デコーダ連結
model_input = encoder_input
model_output = decoder(encoder_output)
model = Model(model_input, model_output)

# 学習用設定設定（最適化関数、損失関数）
optimizer = Adam(learning_rate=LEARNING_RATE)

def r_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=[1,2,3])

model.compile(optimizer=optimizer, loss=r_loss)

# 学習実行
model.fit(
    train,
    train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# 学習実行
history = model.fit(
    train,
    train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# 学習曲線の可視化
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Save the entire model to a HDF5 file
model.save('modeltest.h5')