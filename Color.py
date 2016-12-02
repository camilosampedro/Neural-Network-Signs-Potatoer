# -----------------------------------------------------------------------------
# ------ NEURAL NETWORK SIGNS POTATOER ----------------------------------------
# ------ Basic OpenCV Neural Network that recognizes transit signals ----------
# ------ Por: Camilo A. Sampedro camilo.sampedro@udea.edu.co ------------------
# ------      Estudiante ingeniería de sistemas, Universidad de Antioquia -----
# ------      CC 1037640884 ---------------------------------------------------
# ------ Por: C. Vanessa Pérez cvanessa.perez@udea.edu.co ---------------------
# ------      Estudiante ingeniería de sistemas, Universidad de Antioquia -----
# ------      CC 1128440531 ---------------------------------------------------
# ------ Curso Básico de Procesamiento de Imágenes y Visión Artificial --------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# - Needed libraries ----------------------------------------------------------
# -----------------------------------------------------------------------------
import numpy as np


# -----------------------------------------------------------------------------
# - Color class ---------------------------------------------------------------
# -   Stores some useful color values (Color boundaries) ----------------------
# -----------------------------------------------------------------------------
class Color():
    # Colors
    RED = "red"
    BLUE = "blue"
    YELLOW = "yellow"
    BLACK = "black"

    # Red boundaries (Red has two, it splits on Hue value)
    RED_LOWER_1 = np.array([0, 45, 15], dtype="uint8")
    RED_UPPER_1 = np.array([15, 245, 230], dtype="uint8")
    RED_LOWER_2 = np.array([170, 45, 15], dtype="uint8")
    RED_UPPER_2 = np.array([255, 245, 230], dtype="uint8")

    # Black
    LOWER_BLACK = np.array([0, 0, 0], dtype="uint8")
    UPPER_BLACK = np.array([255, 150, 80], dtype="uint8")

    # Get a color boundaries based on its String
    def get(color):
        return {
            Color.RED: [[Color.RED_LOWER_1, Color.RED_UPPER_1],
                        [Color.RED_LOWER_2, Color.RED_UPPER_2]],
            Color.BLACK: [[Color.LOWER_BLACK, Color.UPPER_BLACK]]
        }.get(color, [])
