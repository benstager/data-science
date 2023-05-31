from collections import Counter

domains = []

with open('emails.txt') as f:
    for line in f:
        domains.append(line.split('@')[1])

print(Counter(domains))