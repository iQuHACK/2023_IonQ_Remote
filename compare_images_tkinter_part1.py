import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from pprint import pprint

from part1 import load_images, run_part1

images = load_images("data/images.npy")


def update_images(num):
    global left_image, right_image, left_photo, right_photo, left_label, right_label, status_label

    try:
        left_image = images[int(num)]
    except IndexError:
        left_image = np.ones((28, 28)) * 255

    # update the images based on the entered number
    print(f"Computing for image n° {int(num)}...")
    right_image = run_part1(left_image)[1]
    print("Right : ")
    pprint(left_image)
    print("Left : ")
    pprint(right_image)
    status_label.configure(text=f"Image n° {int(num)}")

    # right_image = np.random.randint(0, 255, (28, 28))

    left_photo = ImageTk.PhotoImage(
        Image.fromarray(left_image).resize((280, 280), Image.BOX)
    )
    right_photo = ImageTk.PhotoImage(
        Image.fromarray(right_image).resize((280, 280), Image.BOX)
    )
    left_label.configure(image=left_photo)
    right_label.configure(image=right_photo)


# Create the main window
root = tk.Tk()
root.title("Image Viewer")

# Create the left image
left_image = images[0]
left_photo = ImageTk.PhotoImage(
    Image.fromarray(left_image).resize((280, 280), Image.BOX)
)

# Create the right image
right_image = np.random.randint(0, 255, (28, 28))
right_photo = ImageTk.PhotoImage(
    Image.fromarray(right_image).resize((280, 280), Image.BOX)
)

# Create the left image label and add it to the main window
left_label = tk.Label(root, image=left_photo)
left_label.grid(row=0, column=0)

# Create the right image label and add it to the main window
right_label = tk.Label(root, image=right_photo)
right_label.grid(row=0, column=1)

# Status label
status_label = tk.Label(root, text="Image n° 0")
status_label.grid(row=1, column=0, columnspan=2)

# Create a label for the number prompt
prompt_label = tk.Label(root, text="Enter a number:")
prompt_label.grid(row=2, column=0, columnspan=2)

# Create an entry widget for the number prompt
number_entry = tk.Entry(root)
number_entry.grid(row=3, column=0, columnspan=2)

# Create a button to update the images
update_button = tk.Button(
    root, text="Update Images", command=lambda: update_images(number_entry.get())
)
update_button.grid(row=34, column=0, columnspan=2)

root.bind("<Return>", lambda e: update_images(number_entry.get()))

root.bind("<Escape>", lambda e: root.destroy())

# Run the main loop
root.mainloop()
