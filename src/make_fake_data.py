"""
make_fake_data.py: creates a tiny fake dataset for quick local tests.

Where this fits in the pipeline:
    (fake) data -> model -> train -> evaluate -> predict
The real dataset (27,558 images) only lives on Kaggle. To check on the laptop
that our code runs without crashing, we draw a few dozen cartoon "cells":
  - Uninfected: a pink circle on a black background.
  - Parasitized: the same, plus a dark purple blob (the "parasite").

It copies the real dataset's quirks on purpose, so our data code is tested
against them:
  - one folder per class, named after the class
  - filenames that contain a patient/slide code, like the real ones
  - a nested duplicate folder (cell_images/cell_images/) with the same images
  - a junk Thumbs.db file in each class folder

Run it from the project folder:
    python -m src.make_fake_data
"""

import math
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from src import config

# Colours as (red, green, blue), each from 0 to 255.
BACKGROUND_COLOUR = (0, 0, 0)        # real cell images have a black background
CELL_COLOUR = (220, 150, 160)        # pinkish, like a stained red blood cell
PARASITE_COLOUR = (90, 40, 110)      # dark purple, like a stained parasite

# Each fake cell's colour is nudged by up to this much per channel, like stain
# colour varying between slides. Without it, two plain healthy cells of the same
# size would be pixel-identical, i.e. accidental duplicates.
MAX_COLOUR_SHIFT = 20

# How much of the image the cell fills, as a fraction of the image's width.
CELL_FRACTION_OF_IMAGE = 0.8
# How big the parasite is, as a fraction of the cell's width.
PARASITE_FRACTION_OF_CELL = 0.25


def draw_fake_cell(image_size: int, is_infected: bool, rng: random.Random) -> Image.Image:
    """Draw one cartoon cell. Infected cells get a purple blob inside."""
    image = Image.new("RGB", (image_size, image_size), BACKGROUND_COLOUR)
    draw = ImageDraw.Draw(image)

    # The cell: a circle centred in the image, in a slightly random shade of pink.
    red, green, blue = CELL_COLOUR
    cell_colour = (
        red + rng.randint(-MAX_COLOUR_SHIFT, MAX_COLOUR_SHIFT),
        green + rng.randint(-MAX_COLOUR_SHIFT, MAX_COLOUR_SHIFT),
        blue + rng.randint(-MAX_COLOUR_SHIFT, MAX_COLOUR_SHIFT),
    )
    cell_diameter = image_size * CELL_FRACTION_OF_IMAGE
    margin = (image_size - cell_diameter) / 2
    cell_box = (margin, margin, margin + cell_diameter, margin + cell_diameter)
    draw.ellipse(cell_box, fill=cell_colour)

    if is_infected:
        # The parasite: a small circle at a random spot inside the cell.
        # We pick a random direction and distance from the cell's centre.
        # The distance is capped so the parasite never pokes outside the cell.
        parasite_diameter = cell_diameter * PARASITE_FRACTION_OF_CELL
        parasite_radius = parasite_diameter / 2
        max_distance = cell_diameter / 2 - parasite_radius

        direction = rng.uniform(0, 2 * math.pi)  # angle in radians
        distance = rng.uniform(0, max_distance)
        centre = image_size / 2
        parasite_x = centre + distance * math.cos(direction)
        parasite_y = centre + distance * math.sin(direction)

        parasite_box = (
            parasite_x - parasite_radius,
            parasite_y - parasite_radius,
            parasite_x + parasite_radius,
            parasite_y + parasite_radius,
        )
        draw.ellipse(parasite_box, fill=PARASITE_COLOUR)

    return image


def make_fake_filename(patient_number: int, cell_number: int) -> str:
    """
    Build a filename in the same style as the real dataset.

    Real example: C33P1thinF_IMG_20150619_114756a_cell_179.png
    The start ("C33P1") is the patient/slide code. Phase 1 checks this pattern
    on the real filenames before we rely on it.
    """
    patient_code = f"C{patient_number}P{patient_number}"
    return f"{patient_code}thinF_IMG_20150101_120000_cell_{cell_number}.png"


def make_class_folder(class_folder: Path, is_infected: bool, rng: random.Random) -> None:
    """Fill one class folder (e.g. .../Parasitized) with fake cell images."""
    class_folder.mkdir(parents=True, exist_ok=True)

    for cell_number in range(config.FAKE_IMAGES_PER_CLASS):
        # Hand out cells to patients in turn (1, 2, ..., 8, 1, 2, ...), so every
        # patient has cells of both classes, like real infected patients' slides.
        patient_number = cell_number % config.FAKE_NUM_PATIENTS + 1

        image_size = rng.randint(config.FAKE_IMAGE_MIN_SIZE, config.FAKE_IMAGE_MAX_SIZE)
        image = draw_fake_cell(image_size, is_infected, rng)
        image.save(class_folder / make_fake_filename(patient_number, cell_number))

    # A junk file, like the Thumbs.db files Windows leaves in the real dataset.
    (class_folder / "Thumbs.db").write_bytes(b"not an image")


def main() -> None:
    """Delete any old fake dataset and create a fresh one."""
    # Same seed every run, so the fake images are identical every time.
    rng = random.Random(config.RANDOM_SEED)

    if config.FAKE_DATA_DIR.exists():
        shutil.rmtree(config.FAKE_DATA_DIR)

    top_folder = config.FAKE_DATA_DIR / "cell_images"
    make_class_folder(top_folder / "Uninfected", is_infected=False, rng=rng)
    make_class_folder(top_folder / "Parasitized", is_infected=True, rng=rng)

    # Copy everything into a nested duplicate folder, like some Kaggle versions.
    nested_duplicate = top_folder / "cell_images"
    shutil.copytree(top_folder / "Uninfected", nested_duplicate / "Uninfected")
    shutil.copytree(top_folder / "Parasitized", nested_duplicate / "Parasitized")

    total_images = 2 * config.FAKE_IMAGES_PER_CLASS
    print(f"Created {total_images} fake images in {top_folder}")
    print(f"(plus a nested duplicate copy in {nested_duplicate} and Thumbs.db junk files)")


if __name__ == "__main__":
    main()
