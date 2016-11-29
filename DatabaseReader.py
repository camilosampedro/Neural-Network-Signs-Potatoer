import csv


class DatabaseReader(object):

    def __init__(self, database_path):
        self.csv_file = open(database_path, 'rt')
        self.database_path = database_path

    def read_database(self):
        self.database = csv.reader(self.csv_file, delimiter=',')
        self.database = list(map(lambda x: x, self.database))
        return self.database

    def apply_function(self, function):
        return map(function, self.database_reader)

    def size(self):
        count = 0
        for image in self.database_reader:
            count += 1
        return count
