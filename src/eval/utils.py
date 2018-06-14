import csv
from collections import defaultdict

def read_in_cats():
    categories = defaultdict()
    reader = csv.reader(open("./cats.txt"))
    for row in reader:
        categories[row[0].strip()] = row[1:]
    return categories