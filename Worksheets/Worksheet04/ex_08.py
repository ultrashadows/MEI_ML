from datetime import date

start_date = list(str(input("Start Date (yyyy,mm,dd): ")).split(','))
end_date = list(str(input("End Date (yyyy,mm,dd): ")).split(','))

start_date = date(int(start_date[0]), int(start_date[1]), int(start_date[2]))
end_date = date(int(end_date[0]), int(end_date[1]), int(end_date[2]))

print("Days between dates: " + str(end_date - start_date))
