import string

numbers = [*"0123456789"]
letters = list(string.ascii_letters)

numbers_count = 0
letters_count = 0
value = [*str(input("Insert string to be evaluated: "))]

for char in value:
    if char in numbers:
        numbers_count += 1
    elif char in letters:
        letters_count += 1

print("Numbers: " + str(numbers_count) + " | Letters: " + str(letters_count))
