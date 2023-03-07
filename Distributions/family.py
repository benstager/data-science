import enum, random

class Kid(enum.Enum):
    boy = 0
    girl = 1

def random_kid():
    return random.choice([Kid.boy, Kid.girl])

both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)

for _ in range(10000):
    younger  = random_kid()
    older = random_kid()

    if older == Kid.girl:
        older_girl += 1
    if older == Kid.girl and younger == Kid.girl:
        both_girls += 1
    if older == Kid.girl or younger == Kid.girl:
        either_girl += 1

print('P(B|G) = ', both_girls/older_girl)
print('P(B|E) = ', both_girls/either_girl)

# Demonstration of P(E|F) = P(E,F)/P(F)
