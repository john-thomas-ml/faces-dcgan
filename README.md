# Celebrity Face Generation using DCGAN

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic celebrity faces using the CelebA dataset. The model is trained on a subset of the dataset and saved for generating new images.

## Project Structure

- `img_align_celeba/`: Directory containing the CelebA dataset images (aligned and cropped version).
- `faces_dcgan_model.h5`: Saved trained GAN model for generating celebrity faces.
- `faces.ipynb`: Jupyter notebook containing the code for data preprocessing, model architecture, training, and image generation.

## Dataset

- **CelebA Dataset**: Uses the aligned and cropped version of the CelebA dataset. Only the first 10,000 images are used to reduce training time.
- **Preprocessing**:
  - Images are cropped to a 128x128 box (coordinates: (26, 51, 154, 179)) and resized to 64x64 pixels.
  - Pixel values are normalized to [-1, 1] using `(X_train / 127.5) - 1.0`.

## Model Architecture

### Generator

- **Input**: 100-dimensional random noise vector.
- **Architecture**:
  - Dense layer (4x4x512 units), reshaped to [4, 4, 512].
  - Conv2DTranspose layers for upsampling:
    - 256 filters, 4x4 kernel, strides=2 → [8, 8, 256]
    - LeakyReLU(alpha=0.2) + BatchNormalization
    - 128 filters, 4x4 kernel, strides=2 → [16, 16, 128]
    - LeakyReLU(alpha=0.2) + BatchNormalization
    - 64 filters, 4x4 kernel, strides=2 → [32, 32, 64]
    - LeakyReLU(alpha=0.2) + BatchNormalization
    - 3 filters, 4x4 kernel, strides=2, tanh activation → [64, 64, 3]
- **Output**: 64x64x3 image with pixel values in [-1, 1].

### Discriminator

- **Input**: 64x64x3 image.
- **Architecture**:
  - Conv2D(32, kernel_size=4, strides=2) → [32, 32, 32]
  - Conv2D(64, kernel_size=3, strides=2) → [16, 16, 64]
  - LeakyReLU(alpha=0.2) + Dropout(0.5)
  - Conv2D(128, kernel_size=3, strides=2) → [8, 8, 128]
  - LeakyReLU(alpha=0.2)
  - Conv2D(256, kernel_size=4, strides=2) → [4, 4, 256]
  - LeakyReLU(alpha=0.2)
  - Flatten()
  - Dropout(0.5)
  - Dense(1, sigmoid activation)
- **Output**: Probability (0 to 1) that the input image is real.

### GAN

- Combines generator and discriminator.
- Discriminator is non-trainable during generator training.

## Training

- **Epochs**: 100
- **Batch Size**: 128
- **Optimizers**:
  - Discriminator: Adam (learning_rate=0.0002, beta_1=0.5)
  - Generator (via GAN): Adam (learning_rate=0.0002, beta_1=0.5)
- **Loss**: Binary Crossentropy
- **Process**:
  - Generate fake images from noise.
  - Train discriminator on real images (labeled 0.9) and fake images (labeled 0).
  - Train generator via GAN twice per batch, each time with a new noise batch and real labels (1).
- **Monitoring**: Displays 10 generated images every 10 epochs and plots generator/discriminator losses.
- **Note**: Training is computationally intensive. A GPU is recommended.

## Results

The trained model generates realistic celebrity faces, saved as `faces_dcgan_model.h5`. The `faces.ipynb` notebook includes visualizations of generated images and loss curves.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- Jupyter Notebook

Install dependencies using:
```bash
pip install tensorflow keras numpy pandas matplotlib pillow notebook
```

## Setup and Running

1. **Download CelebA Dataset**:
   - Get the aligned and cropped CelebA dataset from [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
   - Place images in `img_align_celeba/img_align_celeba/`.

2. **Run the Notebook**:
   - Open `faces.ipynb` in Jupyter Notebook.
   - Update the dataset path in the notebook if necessary.
   - Run cells sequentially to preprocess data, train the model, and generate images.

3. **Generate Images with Saved Model**:
   Use the following code to load the model and generate new faces:
   ```python
   from tensorflow.keras.models import load_model
   import numpy as np
   import matplotlib.pyplot as plt

   gan = load_model('faces_dcgan_model.h5')
   noise = np.random.uniform(-1, 1, size=[10, 100])
   generated_images = gan.layers[0].predict(noise)

   plt.figure(figsize=(10, 8))
   for i in range(10):
       plt.subplot(2, 5, i+1)
       plt.imshow((generated_images[i] + 1) / 2)
       plt.axis('off')
   plt.tight_layout()
   plt.show()
   ```

## License

This project is licensed under the MIT License. The CelebA dataset is for non-commercial research and educational use only. See the [CelebA license](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for details.