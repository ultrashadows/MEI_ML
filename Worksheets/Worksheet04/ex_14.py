instruction = str(input("Insert stoppage instruction: "))
values = list(str(input("Insert list of values separated by a comma: ")).split(','))

print("Values inserted: ")
for value in values:
    print(value)
    if value != instruction:
        continue
    else:
        print("Stoppage instruction found!")
        break
