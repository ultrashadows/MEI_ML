values = []

for x in range(1500, 2701):
    if x % 7 == 0 and x % 5 == 0:
        values.append(x)

print(values)
