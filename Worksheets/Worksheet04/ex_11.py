input_list = list(str(input("Insert a list of numbers separated by a comma: ")).split(','))

print("4th element: " + input_list[3] if len(input_list) > 3 else "List doesn't have a 4th element")
