low = int(input("Insert the low end of the sum: "))
high = int(input("Insert the high end of the sum: "))
value = int(input("Insert the value to return if the sum is within the parameters: "))

val1 = int(input("Input number 1: "))
val2 = int(input("Input number 2: "))

val_sum = val1 + val2

if low <= val_sum <= high:
    print("Parameters hit! Sum has been set to " + str(value))
else:
    print("Sum: " + str(val_sum))
