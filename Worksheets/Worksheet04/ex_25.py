x = int(input("Insert x value: "))
y = int(input("Insert y value: "))
z = int(input("Insert z value: "))

if x == y and y == z:
    print("Triângulo equilátero")
elif x == y and y != z or x == z and x != y or y == z and x != y:
    print("Triângulo isósceles")
else:
    print("Triângulo escaleno")
