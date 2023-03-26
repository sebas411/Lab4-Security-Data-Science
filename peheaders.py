import pefile
import sys
import os

filename = ""
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    print("Se necesitan argumentos")
    quit()

print("Se ingreso archivo: %s"%filename)

pe = pefile.PE(filename)

first_write = not os.path.exists("dataset.csv")

ds = open("dataset.csv", "a")

if first_write:
    ds.write("file,api\n")
print_comma = False
ds.write(filename.split('/')[-1])
ds.write(',"')
for entry in pe.DIRECTORY_ENTRY_IMPORT:
    for function in entry.imports:
        if print_comma:
            ds.write(',')
        else:
            print_comma = True
        ds.write(function.name.decode('utf-8'))
ds.write('"\n')
ds.close()
