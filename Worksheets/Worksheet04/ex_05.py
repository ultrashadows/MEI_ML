data = str(input("Insert your data separated by commas: "))

data_list = list(data.split(','))
data_tuple = tuple(data.split(','))

print("Data List: " + str(data_list))
print("Data Tuple: " + str(data_tuple))
