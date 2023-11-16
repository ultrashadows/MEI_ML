vowels = ['a', 'e', 'i', 'o', 'u']

character = str(input("Insert a character: "))
if character in vowels:
    print(character + " is a vowel!")
else:
    print(character + " is not a vowel!")
