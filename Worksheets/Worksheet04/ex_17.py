from math import sqrt

x1 = int(input("Insert x1 value: "))
x2 = int(input("Insert x2 value: "))
y1 = int(input("Insert y1 value: "))
y2 = int(input("Insert y2 value: "))

print("Distância Euclidiana entre (" + str(x1) + ', ' + str(y1) + ") e (" + str(x2) + ', ' + str(y2) + ") é " +
      str(sqrt((x1 - x2)**2 + (y1 - y2)**2)))
