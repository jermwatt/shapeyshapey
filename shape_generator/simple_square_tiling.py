import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import idx2numpy
import random


# split array pairs list into train and test sets
def train_test_split_image_pairs(image_pairs, test_ratio=0.2):
    # Calculate the number of images for the test set
    num_test = int(len(image_pairs) * test_ratio)

    # Split the image pairs into training and test sets
    test_set = image_pairs[:num_test]
    train_set = image_pairs[num_test:]

    return train_set, test_set


# pair off input images at random
def get_random_image_pairs(images):
    # Shuffle the list of images randomly
    random.shuffle(images)

    # Create pairs of images
    image_pairs = []
    for i in range(0, len(images), 2):
        # Ensure the last pair has two elements
        if i + 1 < len(images):
            image_pairs.append((images[i], images[i + 1]))

    return image_pairs


def save_image_pairs_as_idx(image_pairs, output_file):
    # Convert the image pairs to a numpy array
    image_pairs_array = np.array(image_pairs)

    # Save the image pairs as an IDX file
    idx2numpy.convert_to_file(output_file, image_pairs_array)


def save_images_as_idx(images, output_file):
    # Convert the images and labels to numpy arrays
    images_array = np.array(images)
    # labels_array = np.array(labels)

    # Save the images and labels as IDX files
    idx2numpy.convert_to_file(output_file, images_array)
    # idx2numpy.convert_to_file(output_file.replace('.ubyte', '-labels.ubyte'), labels_array)


def read_all_image_pairs_from_idx(file_path):
    # Load the IDX file and convert it to a numpy array
    image_pairs_array = idx2numpy.convert_from_file(file_path)

    # Split the array into individual pairs
    num_pairs = image_pairs_array.shape[0]
    image_pairs = np.split(image_pairs_array, num_pairs, axis=0)

    return image_pairs


### visualization ###
def display_random_instance(tilings):
    # random number between 0 and length of tilings -1
    random_index = np.random.randint(0, tilings.shape[0])

    # get the random tiling
    random_tiling = tilings[random_index]

    # get dimensions of tile
    # tile_width = int(np.sqrt(random_tiling.shape[0]))

    # plot random tiling
    plt.imshow(random_tiling, cmap='gray')


def create_gif(tilings, num_examples=100, gif_path="animation.gif"):   
    # example tiling for width
    # example_tiling = tilings[0]

    # get dimensions of tile
    # tile_width = int(np.sqrt(example_tiling.shape[0]))

    images = []
    for i in range(num_examples):
        random_index = np.random.randint(0, tilings.shape[0])
        random_tiling = tilings[random_index]
        image_data = random_tiling * 255  # .reshape(tile_width, tile_width)
        image = Image.fromarray(np.uint8(image_data))
        image = image.convert("RGB")  # Convert to RGB mode
        images.append(image)

    # Save the images as a GIF
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=100,  # Delay between frames in milliseconds
        loop=0  # Loop forever
    )

    print(f"Saved GIF at: {gif_path}")


def create_complete_tiling(img_width=28, square_width=10):
    if square_width > img_width-1:
        raise ValueError("square_width cannot be greater than img_width")

    base_image = np.zeros((img_width, img_width))
    tensor = []

    for i in range(img_width - square_width + 1):
        for j in range(img_width - square_width + 1):
            image = base_image.copy()
            image[i:i+square_width, j:j+square_width] = 1
            tensor.append(image)  # .flatten())
    tensor = np.array(tensor)
    return tensor
