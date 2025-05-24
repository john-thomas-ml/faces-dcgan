# Celebrity Face Generation using DCGAN

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic celebrity faces using the CelebA dataset. The model is trained on a subset of the dataset and saved for generating new images. The project structure has been updated to include a dedicated script for image generation, along with a requirements file for easier dependency management.

## Project Structure

- `generated/`: Directory containing the generated face images (e.g., `face_000.png`, `face_001.png`, ..., `face_009.png`).
- `img_align_celeba/`: Directory containing the CelebA dataset images (aligned and cropped version).
- `scripts/`: Directory containing the script for generating images.
  - `generate.py`: Python script to generate new celebrity faces using the trained model.
- `faces_dcgan_model.h5`: Saved trained GAN model for generating celebrity faces.
- `faces.ipynb`: Jupyter notebook containing the code for data preprocessing, model architecture, training, and image generation.
- `README.md`: Project documentation file.
- `requirements.txt`: File listing the required Python packages and their minimum versions.

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

The trained model generates realistic celebrity faces, saved as `faces_dcgan_model.h5`. The `generated/` folder contains the output images (e.g., `face_000.png` to `face_009.png`) produced by the `generate.py` script. The `faces.ipynb` notebook includes visualizations of generated images and loss curves.

## Requirements

Install the required Python packages using the provided `requirements.txt` file:
```
tensorflow>=2.9.0
numpy>=1.19.5
pandas>=1.3.0
matplotlib>=3.4.0
Pillow>=8.0.0
notebook>=6.0.0
```

Install dependencies with:
```bash
pip install -r requirements.txt
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
   - Use the `generate.py` script to generate new faces:
     ```bash
     python scripts/generate.py --model faces_dcgan_model.h5 --num 10 --out generated/
     ```
   - This script generates 10 faces by default and saves them as `face_000.png` to `face_009.png` in the `generated/` directory. Adjust the `--num` and `--out` arguments as needed.

## License

This project is licensed under the MIT License. The CelebA dataset is for non-commercial research and educational use only. See the [CelebA license](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for details.