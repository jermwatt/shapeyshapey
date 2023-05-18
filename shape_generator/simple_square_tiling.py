import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 


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
