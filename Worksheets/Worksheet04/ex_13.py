numbers = list(str(input("Insert list of numbers separated by a comma: ")).split(','))

number = str(input("Insert a number: "))

if number in numbers:
    print(str(number) + " is included in the list!")
else:
    print(str(number) + " is not included in the list!")
