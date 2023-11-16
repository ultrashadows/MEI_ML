values = list(str(input("Insert list of values separated by a comma: ")).split(','))
floor = int(input("Insert min value: "))

found = 0

for value in values:
    if int(value) > floor:
        continue
    else:
        found = 1
        break

print("All values are bigger than the floor value!" if found == 0 else "Found a value that doesn't meet the "
                                                                       "requirements!")
