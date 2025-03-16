import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from PIL import Image
import upscl
import sys
os.makedirs('generated_images',exist_ok=True)
os.makedirs('results',exist_ok=True)
print('[*DEEPCAT-X*] Your using XDeepCat V0.83-XS+G2V')
print('[*DEEPCAT-X*] XDeepCat Resolution Latest Support - 512x512')
print('[*DEEPCAT-X*] XDeepCat-G variant image size - 2048x2048')

# Define the dataset path
dataset_path = './dataset/'

# Load image paths and corresponding text descriptions
image_paths = []
text_descriptions = []

for image_name in os.listdir(dataset_path):
    image_folder = os.path.join(dataset_path, image_name)
    if os.path.isdir(image_folder):
        for img_file in os.listdir(image_folder):
            if img_file.endswith('.jpg'):
                image_paths.append(os.path.join(image_folder, img_file))
                text_descriptions.append(image_name)  # Assuming the folder name is the description

# Load and preprocess images
images = []
for img_path in image_paths:
    img = load_img(img_path, target_size=(512, 512))  # Resize images to 256x256
    img = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    images.append(np.round(img,21))

images = np.array(images)

# Preprocess text descriptions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_descriptions)
text_sequences = tokenizer.texts_to_sequences(text_descriptions)
text_sequences = pad_sequences(text_sequences, maxlen=20)  # Pad sequences to a fixed length

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(text_sequences, images, test_size=0.2, random_state=42)

# Define the text encoder
text_input = layers.Input(shape=(20,))
text_embedding = layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(text_input)
text_embedding = layers.LSTM(128)(text_embedding)

# Define the image generator for 256x256 output
image_generator_input = layers.Input(shape=(128,))
x = layers.concatenate([text_embedding, image_generator_input])  # Concatenate text embedding and noise
x = layers.Dense(16 * 16 * 128)(x)  # Project to the required size
x = layers.Reshape((16, 16, 128))(x)
x = layers.Dense(512)(x)
x = layers.Dense(256)(x)
x = layers.Dense(128)(x)
x = layers.Dense(64)(x)
x = layers.Dense(32)(x)
x = layers.Dense(16)(x)
x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)  # Upsample to 32x32
x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)  # Upsample to 64x64
x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same')(x)  # Upsample to 128x128
x = layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same')(x)  # Upsample to 256x256
x = layers.Conv2DTranspose(8, (4, 4), strides=(2, 2), padding='same')(x)  # Upsample to 512x512
x = layers.Conv2DTranspose(3, (4, 4), strides=(1, 1), padding='same', activation='sigmoid')(x)  # Output 256x256

# Define the full model
text_to_image_model = models.Model([text_input, image_generator_input], x)

# Compile the model
text_to_image_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = text_to_image_model.fit(
    [X_train, np.random.normal(0, 1, (len(X_train), 128))],  # Random noise as initial input for the generator
    y_train,
    validation_data=([X_val, np.random.normal(0, 1, (len(X_val), 128))], y_val),
    epochs=312,
    batch_size=32
)

text_to_image_model.save('DeepCat.keras')

def save_img(path,arr):
    img=Image.fromarray(arr)
    img.save(path)

# Function to generate and save images from text descriptions
def generate_and_save_image_from_text(text_description, save_path):
    # Generate the image
    text_sequence = tokenizer.texts_to_sequences([text_description])
    text_sequence = pad_sequences(text_sequence, maxlen=20)
    noise = np.random.normal(0, 1, (1, 128))
    generated_image = text_to_image_model.predict([text_sequence, noise])

    # Rescale the image from [0, 1] to [0, 255]
    generated_image = generated_image[0] * 255.0
    generated_image = np.clip(generated_image, 0, 255).astype('uint8')

    # Save the image
    save_img(save_path, generated_image)
    print(f"Image saved to {save_path}")

# Example usage
output_folder = "./generated_images"

# Generate and save an image
text_description = sys.argv[1]
save_path = os.path.join(output_folder, "generated_image_512x512_xdeepcat.png")
generate_and_save_image_from_text(text_description, save_path)
upscl.upscale('xdeepcat')
wmarker.watermark_with_transparency('./results/sr_image.png','./results/sr_image.png','wmark.png',(0,0))
print('[*DEEPCAT-X*] Model training successful! Open DeepCat-Console?')
print('[*DEEPCAT-X-TIPS*] Always images with extra-size (2048x2048) saving in /results and normal images (512x512) in /generated_images')
answer=input('No/Yes(N/Y)>> ')
if answer.lower()=='n':
    print('[*DEEPCAT-X*] Ok. Closing...')
    exit()
elif answer.lower()=='y':
    print('[*DEEPCAT-X*] Opening DeepCat-Console ...')
    while True:
        input_of_user = input('[-=User=-] >> ')
        generate_and_save_image_from_text(text_description, save_path)
        upscl.upscale('xdeepcat')
        wmarker.watermark_with_transparency('./results/sr_image.png','./results/sr_image.png','wmark.png',(0,0))
        print('[-=DeepCatX Console=-] Generation success!')
