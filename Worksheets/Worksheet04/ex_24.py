human_age = int(input("Insert the dog's age in human years: "))

dog_age = 0

if 0 <= human_age <= 2:
    dog_age = human_age * 10.5
elif human_age > 2:
    dog_age = 21 + ((human_age - 2) * 4)
else:
    dog_age = -1

print("Dog age: " + str(dog_age) if dog_age != -1 else "That's not a possible age!")
