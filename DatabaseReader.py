import csv


class DatabaseReader(object):

    def __init__(self, database_path):
        self.csv_file = open(database_path, 'rt')
        self.database_path = database_path

    def read_database(self):
        self.database_reader = csv.reader(self.csv_file, delimiter=',')
        return self.database_reader

    def apply_function(self, function):
        mapped_database = map(function, self.database_reader)
        return mapped_database
