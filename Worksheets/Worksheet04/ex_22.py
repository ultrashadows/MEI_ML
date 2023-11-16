numbers = list(str(input("Insert list of numbers separated by a comma: ")).split(','))
count_par = 0
count_impar = 0

for number in numbers:
    if int(number) % 2 == 0:
        count_par += 1
    else:
        count_impar += 1

print("Even numbers: " + str(count_par) + ", Odd numbers: " + str(count_impar))
