import torch
from PIL import Image
from py_real_esrgan.model import RealESRGAN


def upscale(mdl):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)

    if mdl=='xdeepcat':
        path_to_image = 'generated_images/generated_image_512x512_xdeepcat.png'
    else:
        path_to_image = 'generated_images/generated_image_256x256_deepcat.png'
    image = Image.open(path_to_image).convert('RGB')

    sr_image = model.predict(image)

    sr_image.save('results/sr_image.png')
