import numpy as np

from part1 import load_images, run_part1

images = load_images("data/images.npy")


def value_to_text(value: int) -> str:
    """Divide the 0-255 interval in 5 parts using   , ░░, ▒▒, ▓▓and ██"""
    if value < 51:
        return "  "
    elif value < 102:
        return "░░"
    elif value < 153:
        return "▒▒"
    elif value < 204:
        return "▓▓"
    elif value < 256:
        return "██"


def display_images(left_image, right_image):
    for i in range(28):
        for j in range(28):
            print(value_to_text(left_image[i][j]), end="")

        print("   ", end="")
        for j in range(28):
            try:
                print(value_to_text(right_image[i][j]), end="")
            except IndexError:
                print("", end="")
                break
        print("")

    print("\n" + "Before".center(28 * 2) + "   " + "After".center(28 * 2))


# update the images based on the entered number
num = int(input("Enter the number of the image you want to display: "))

try:
    left_image = images[int(num)]
except IndexError:
    left_image = np.ones((28, 28)) * 255


print(f"Computing for image n° {int(num)}...\n")

right_image = run_part1(left_image)[1]

display_images(left_image, right_image)
