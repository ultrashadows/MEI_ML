value = str(input("Insert a string to evaluate: "))
char = str(input("Insert character to find: "))
count = 0

chars = [*value]

for position in chars:
    if position == char:
        count += 1

print("Number of " + char + " in string: " + str(count))
