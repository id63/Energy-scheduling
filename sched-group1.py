import random

def cost(S, N, X):
    # Return the cost of solution S with respect to N and X
    assert len(S) == len(N) 
    total = 0
    j = 0
    for i in range(len(S)):
        if S[i] == 1:
            total += N[i] * X[j]
            j += 1
    return total

# Main Program: Ask the user to specify the input file, then read it into memory.
infile = input("Enter problem file name >> ")
with open(infile, 'r') as f:
    N = eval(f.readline())
    X = eval(f.readline())

# Do some basic checks to make sure the input is valid
assert type(N) == list and type(X) == list and len(N) >= len(X), "Error. Invalid input"
n = len(N)
x = len(X)

# Make a random solution
S = []
for i in range(x):
    S.append(1)
for i in range(n-x):
    S.append(0)
random.shuffle(S)

# Write some information to the screen
print("File    =", infile)
print("n       =", n)
print("x       =", x)
print("N       =", N)
print("X       =", X)
print("S       =", S)
print("cost(S) =", cost(S, N, X))


