# Updated value for velocity
import math
import argparse
v = 1.11  # velocity in m/s
h =0.5

g =9.8
R = 0.548348807
# Calculate range for the new velocity
def range(row):
    print(row[5])
    v = float(row[5])
    h = float(row[7]) - 0.7
    g = 9.8
    R = (v**2 / (2 * g)) + (v / g) * math.sqrt((v**2 / 2) + 2 * g * h)
    print(R)
    row.append(R)
    with open('benchmark_5.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow(row)

def velocity():
    v_2 = (-4 * R * g - 8 * g * h + math.sqrt((4 * R * g + 8 * g * h)**2 + 16 * R**2 * g**2)) / (2)
    v = math.sqrt(v_2)
    print(v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['range','velocity'])
    args = parser.parse_args()
    import csv
    with open('benchmark_5.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)
            if args.mode == "range":    
                range(row)
            else:
                velocity()
    