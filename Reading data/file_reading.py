import re

file = open('tester.txt', 'r')

file.close()

starts_with_T = 0

with open('tester.txt') as f:
    for line in f:
        if 'about' in line:
            starts_with_T += 1

print(starts_with_T)