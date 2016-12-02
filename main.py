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
# ------ V1 Septiembre de 2016-------------------------------------------------
# ------ Nota: Algunos comentarios se dejaron en inglés para mantener la ------
# ------   aplicación legible para ambos idiomas (Inglés y español).     ------
# ------ Note: Some comments were left on English to keep the            ------
# ------   application readable for both English and Spanish.            ------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# - 1. Needed libraries -------------------------------------------------------
# -----------------------------------------------------------------------------
import sys
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog
from DatabaseReader import DatabaseReader
from ImageRecognizer import ImageRecognizer
import numpy as np


# -----------------------------------------------------------------------------
# - 2. Global variables -------------------------------------------------------
# -----------------------------------------------------------------------------
global_alpha = -3
image_folder = "./images/"
h = .02


# -----------------------------------------------------------------------------
# - 3. Global methods ---------------------------------------------------------
# -----------------------------------------------------------------------------
def main(args):
    if len(args) != 1:
        if args[1] == 'train':
            print("Training database")
            database_reader = DatabaseReader('database.csv')
            image_tag_database = database_reader.read_database()
            image_count = len(image_tag_database)
            print("Database has %d rows" % image_count)
            idx = 0
            x = []
            y = []
            for image_row in image_tag_database:
                idx += 1
                if idx != 1:
                    image_path = image_row[0]
                    img_rec = ImageRecognizer(image_path)
                    characteristics = img_rec.extract_characteristics()
                    x.append(characteristics)
                    print("%d%%: %s - %s" % (idx * 100 / image_count,
                                             image_path, str(characteristics)))
                    y.append(image_row[1])
                    print("Output: %s" % image_row[1])
            alphas = np.logspace(-10, 3, 10)
            classifiers = []
            for i in alphas:
                classifiers.append(MLPClassifier(alpha=i, random_state=1,
                                                 hidden_layer_sizes=(100,)))
            names = []
            for i in alphas:
                names.append('alpha ' + str(i))

            x = StandardScaler().fit_transform(x)
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=.4)
            x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
            y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5

            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            # iterate over classifiers
            for name, clf in zip(names, classifiers):
                # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
                clf.fit(x_train, y_train)
                score = clf.score(x_test, y_test)
                print("%s: %f" % (name, score))
                # Plot the decision boundary. For that, we will assign a color
                # to each
                # point in the mesh [x_min, x_max]x[y_min, y_max].
                # if hasattr(clf, "decision_function"):
                #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                # else:
                #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,
                # 1]
        elif args[1] == 'test':
            print("Testing")
        else:
            print("Showing to the teacher :P")
    # files = ask_for_files()
    # database_reader = DatabaseReader('database.csv')
    # image_tag_database = database_reader.read_database()
    # if len(files):
    #     print("Checking for files: %s" % (str(files)))
    # else:
    #     print("Program canceled by the user")
    #     sys.exit()
    # print(len(args))
    # if len(args) != 1:
    #     if args[0] == 'test':
    #         alphas = np.logspace(-5, 3, 5)
    #         classifiers = []
    #         for i in alphas:
    #             classifiers.append(create_neural_network(alpha=i))
    #     else:
    #         alpha = float(args[0])
    #         classifier = create_neural_network(alpha=alpha)
    # else:
    #     alpha = global_alpha
    #     classifier = create_neural_network(alpha=alpha)


# Ask user for files
def ask_for_files():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        filetypes=(
            ("Images", "*.ppm;*.jpg"),
            ("All files", "*.*")),
        initialdir=image_folder,
        multiple=True,
        title="Choose images to analyze (Multiple)")


# Read a image
#  Params:
#   * image_path - String with the path to the image in the file system
def read_image(image_path):
    # Use CV2 canonical image reading
    image_read = cv2.imread(image_path)
    return image_read


def create_neural_network(alpha):
    return MLPClassifier(alpha=alpha, random_state=1)


main(sys.argv)
