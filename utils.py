from PIL import Image
import numpy as np


def preprocess_state(state):
    image = Image.fromarray(state)
    image = image.resize((88, 80))
    image = image.convert("L")
    #     image.show()
    image = np.array(image)

    return image
