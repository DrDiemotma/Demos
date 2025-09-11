import os
from PIL import Image
import numpy as np


def crop_to_margin(image: Image.Image, border_left: float, border_right: float, border_top: float, border_bottom: float) -> Image.Image:
    image_width, image_height = image.size
    left = int(image_width * border_left)
    right = image_width - int(image_width * border_right)
    top = int(image_height * border_top)
    bottom = image_height - int(image_height * border_bottom)
    return image.crop((left, top, right, bottom))

def get_tile(image: Image.Image, tile_width: int, tile_height: int, tile_index_x: int, tile_index_y: int):
    """
    Returns a tile from an image.
    :param image: The image to get the tile from.
    :param tile_width: The width of the tile.
    :param tile_height: The height of the tile.
    :param tile_index_x: The x index of the tile.
    :param tile_index_y: The y index of the tile.
    :return: The tile as a numpy array.
    """
    image_width, image_height = image.size
    offset_x = (image_width % tile_width) // 2
    offset_y = (image_height % tile_height) // 2
    tile = image.crop((tile_index_x * tile_width + offset_x, tile_index_y * tile_height + offset_y,
                       (tile_index_x + 1) * tile_width + offset_x, (tile_index_y + 1) * tile_height + offset_y))
    return np.array(tile, dtype=np.float32) / 255.0


def load_and_tile(directory_pathes: list[str], labels: list[int | float], tile_width: int, tile_height: int, extension: str = ".jpg", convert_bw: bool = True,
    border_left: float = 0.2, border_right: float = 0.2, border_top: float = 0.2, border_bottom: float = 0.2):
    """
    Loads and tiles images from a directory.
    :param directory_pathes: The paths to the directories containing the images.
    :param labels: The labels corresponding to the image folders.
    :param tile_width: The width of the tiles.
    :param tile_height: The height of the tiles.
    :param extension: The extension of the images.
    :param convert_bw: Whether to convert the images to black and white.
    :param border_left: The left border of the image to be left blank.
    :param border_right: The right border of the image to be left blank.
    :param border_top: The top border of the image to be left blank.
    :param border_bottom: The bottom border of the image to be left blank.
    :return: An iterator of tiles as numpy arrays and labels.
    """
    for label, directory_path in zip(labels, directory_pathes):
        files = [x for x in os.listdir(directory_path) if x.endswith(extension)]
        for file in files:
            image = Image.open(os.path.join(directory_path, file)) if not convert_bw else Image.open(os.path.join(directory_path, file)).convert('L')
            image = crop_to_margin(image, border_left, border_right, border_top, border_bottom)
            image_width, image_height = image.size
            tile_count_x = image_width // tile_width
            tile_count_y = image_height // tile_height
            for x_tile in range(0, tile_count_x):
                for y_tile in range(0, tile_count_y):
                    tile = get_tile(image, tile_width, tile_height, x_tile, y_tile)
                    expanded_tile = np.expand_dims(tile, axis=-1)
                    yield expanded_tile, label





def get_number_of_tiles(directories: list[str], tile_width: int, tile_height: int, extension: str = ".jpg",
    border_left: float = 0.2, border_right: float = 0.2, border_top: float = 0.2, border_bottom: float = 0.2) -> int:
    total_number = 0
    for directory_path in directories:
        files = [x for x in os.listdir(directory_path) if x.endswith(extension)]
        for file in files:
            image = crop_to_margin(Image.open(os.path.join(directory_path, file)), border_left, border_right, border_top, border_bottom)
            image_width, image_height = image.size

            tile_count_x = image_width // tile_width
            tile_count_y = image_height // tile_height
            total_number += tile_count_x * tile_count_y
    return total_number
