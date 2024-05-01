# Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
Autoencoder is an unsupervised artificial neural network that is trained to copy its input to output.
An autoencoder will first encode the image into a lower-dimensional representation, then decodes the representation back to the image.
The goal of an autoencoder is to get an output that is identical to the input. Autoencoders uses MaxPooling, convolutional and upsampling layers to denoise the image.
![img2](https://github.com/lokeshrahulv/convolutional-denoising-autoencoder/assets/118423842/32977471-6ecb-43f4-bec7-56456e4374ce)

## Convolution Autoencoder Network Model
![Screenshot 2024-05-01 130949](https://github.com/lokeshrahulv/convolutional-denoising-autoencoder/assets/118423842/1baa931b-81e6-4a19-867b-57a0154e2fa9)

## DESIGN STEPS
## STEP 1:
Download and split the dataset into training and testing datasets

## STEP 2:
rescale the data as that the training is made easy

## STEP 3:
create the model for the program , in this experiment we create to networks , one for encoding and one for decoding Write your own steps

Write your own steps

## PROGRAM

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers, utils, models
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train.shape

x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape)
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

print("LOKESH RAHUL V V \n 212222100024")
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))
x=layers.Conv2D(16,(5,5),activation='relu',padding='same')(input_img)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(7,7),activation='relu',padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(encoded)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(5,5),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(5,5),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(5,5),activation='relu')(x)
x=layers.UpSampling2D((1,1))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)

print("LOKESH RAHUL V V \n 212222100024")
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))

print("LOKESH RAHUL V V \n 212222100024")
metrics = pd.DataFrame(autoencoder.history.history)
metrics[['loss','val_loss']].plot()

decoded_imgs = autoencoder.predict(x_test_noisy)
n=10

print("LOKESH RAHUL V V \n 212222100024")
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
![Screenshot 2024-05-01 123805](https://github.com/lokeshrahulv/convolutional-denoising-autoencoder/assets/118423842/47ad87c1-d124-4cf8-8c59-f3c8c0c47e88)

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-05-01 131025](https://github.com/lokeshrahulv/convolutional-denoising-autoencoder/assets/118423842/ba58a63b-0887-4ba2-af72-f8287facede7)

## Model Summary:
![Screenshot 2024-05-01 125127](https://github.com/lokeshrahulv/convolutional-denoising-autoencoder/assets/118423842/b38ee659-c3a0-4eec-8b21-f5a00de6ced6)

### Original vs Noisy Vs Reconstructed Image
![Screenshot 2024-05-01 125249](https://github.com/lokeshrahulv/convolutional-denoising-autoencoder/assets/118423842/752b83fb-b2d4-4311-a595-b9deced95497)

## RESULT
Thus we have successfully developed a convolutional autoencoder for image denoising application.
