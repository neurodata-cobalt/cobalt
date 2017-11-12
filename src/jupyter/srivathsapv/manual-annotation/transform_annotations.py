import csv
import sys
import math

source_path = sys.argv[1]
target_path = sys.argv[2]

with open(source_path) as csv_file:
    reader = csv.reader(csv_file)
    annotations = [[int(row[1]), int(math.ceil(float(row[2]))), int(math.ceil(float(row[3])))] for row in list(reader)[1:]]

with open(target_path, 'wb') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(annotations)
