import numpy as np


class Color():
    # Colors
    RED = "red"
    BLUE = "blue"
    YELLOW = "yellow"
    BLACK = "black"

    # Red
    RED_LOWER_1 = np.array([0, 45, 15], dtype="uint8")
    RED_UPPER_1 = np.array([15, 245, 230], dtype="uint8")
    RED_LOWER_2 = np.array([170, 45, 15], dtype="uint8")
    RED_UPPER_2 = np.array([255, 245, 230], dtype="uint8")

    # Black
    LOWER_BLACK = np.array([0, 0, 0], dtype="uint8")
    UPPER_BLACK = np.array([255, 255, 100], dtype="uint8")

    def get(color):
        return {
            Color.RED: [[Color.RED_LOWER_1, Color.RED_UPPER_1],
                        [Color.RED_LOWER_2, Color.RED_UPPER_2]],
            Color.BLACK: [[Color.LOWER_BLACK, Color.UPPER_BLACK]]
        }.get(color, [])
