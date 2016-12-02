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
import csv  # CSV files reader and writter


# -----------------------------------------------------------------------------
# - DatabaseReader class ------------------------------------------------------
# -   Reads the csv files and convert them to list that can be iterated in ----
# -   Python                                                               ----
# -----------------------------------------------------------------------------
class DatabaseReader(object):

    # Constructor
    #   database_path: String with the path to the database
    def __init__(self, database_path):
        # Read csv file in bytes (Not in binary)
        self.csv_file = open(database_path, 'rt')
        # Save anyway the database path
        self.database_path = database_path

    # Reads the database and convert it to a list
    def read_database(self):
        # Create a file-reader, using comma delimiter
        self.database = csv.reader(self.csv_file, delimiter=',')
        # Iterates over the file using a map and then saving it to a list
        self.database = list(map(lambda x: x, self.database))
        # Return the list
        return self.database

    # Apply a given function to the file
    def apply_function(self, function):
        return map(function, self.database_reader)

    # Get the database size (Number of rows)
    def size(self):
        # Initialize counter
        count = 0
        # For every row
        for image in self.database_reader:
            # Count one
            count += 1
        # Return that count
        return count
