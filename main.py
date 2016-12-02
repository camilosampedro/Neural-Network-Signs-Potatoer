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
# ------   aplicación pública en Github.                                 ------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# - 1. Needed libraries -------------------------------------------------------
# -----------------------------------------------------------------------------
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    # If there is an argument
    if len(args) != 1:
        if args[1] == 'train':
            print("Training database")
            # Read the database file (Created tags manually)
            database_reader = DatabaseReader('database.csv')
            # Read the database and generate a list
            image_tag_database = database_reader.read_database()
            # Count database rows (For doing percentage)
            image_count = len(image_tag_database)
            print("Database has %d rows" % image_count)
            # Create an index
            idx = 0
            # A list for features
            x = []
            # A list for output (Or tag)
            y = []
            # Iterate over the images
            for image_row in image_tag_database:
                # Grow index
                idx += 1
                # If it is the first row, it will be title, so ignore it
                if idx != 1:
                    # Extract the path of the image
                    image_path = image_row[0]
                    # Create a recognizer (See ImageRecognizer class)
                    img_rec = ImageRecognizer(image_path)
                    # Extract characteristics (See ImageRecognizer class)
                    characteristics = img_rec.extract_characteristics()
                    # Append those characteristics to the features list
                    x.append(characteristics)
                    print("%d%%: %s - %s" % (idx * 100 / image_count,
                                             image_path, str(characteristics)))
                    # Append the output to the output list. A translation of
                    # this class can be found on signals.csv
                    y.append(image_row[1])
                    print("Output: %s" % image_row[1])
            # Alphas (Trying several for knowing which one is better)
            alphas = np.logspace(-10, 3, 10)
            # Create a classifier list
            classifiers = []
            # And initialize a classifier for each alpha
            for i in alphas:
                classifiers.append(MLPClassifier(alpha=i, random_state=1,
                                                 hidden_layer_sizes=(100,)))
            # Create a classifier name list
            names = []
            # And initialize them with the corresponding alpha
            for i in alphas:
                names.append('alpha ' + str(i))

            # Create train-test subsets
            x = StandardScaler().fit_transform(x)
            # Get those train test subsets
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=.4)
            x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
            y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
            # Get the coordinates of the rows (For not losing them later)
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            # Iterate over classifiers (With their names)
            for name, clf in zip(names, classifiers):
                # Train with train subset
                clf.fit(x_train, y_train)
                # Get the score with the test subset
                score = clf.score(x_test, y_test)
                print("%s: %f" % (name, score))
        else:
            print("Showing to the teacher :P")


main(sys.argv)
