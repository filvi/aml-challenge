import requests

def greetings():
    print("\n\n")

    ascii_art = requests.get("http://artii.herokuapp.com/make?text=Roosters&font=slant")

    print(ascii_art.text)

    print("%-20s %-20s %-20s %-20s" % ("Andrea Lomurno", "Jacopo Mocellin", "Tamara Quaranta", "Filippo Vicari"))

    print("\n\n")