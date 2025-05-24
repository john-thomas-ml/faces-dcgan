from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image

def generate_faces(model_path, out_dir, n_faces, noise_dim=100):
    gan = load_model(model_path)
    gen = gan.layers[0]
    os.makedirs(out_dir, exist_ok=True)
    noise = np.random.uniform(-1,1,size=(n_faces, noise_dim))
    faces = gen.predict(noise)
    for i, face in enumerate(faces):
        # rescale from [-1,1] to [0,255]
        img = ((face + 1) * 127.5).astype('uint8')
        Image.fromarray(img).save(os.path.join(out_dir, f'face_{i:03d}.png'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate faces with DCGAN')
    parser.add_argument('--model', default='faces_dcgan_model.h5')
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--out', default='generated/')
    args = parser.parse_args()
    generate_faces(args.model, args.out, args.num)
