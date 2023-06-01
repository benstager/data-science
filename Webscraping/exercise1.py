import sys, re

regex = sys.argv[1]

# Counting lines that match the regex sys.argv[1]
for line in sys.stdin:
    if re.search(regex,line):
        sys.stdout.write(line)

count = 0
for line in sys.stdin:
    count += 1

# Or?
# count = len(sys.stdin)

