import random
import matplotlib.pyplot as plt

class Appliance():
    """
    This class simulates a household appliance. 
    """
    def __init__(self, name, phases):
        self.name = name
        self.phases = phases
        self.scheduleLength = len(phases)

    def __repr__(self):
        return f"This appliance is a {self.name} and  has a schedule length of {self.scheduleLength}\nThe schedule is "
class Timings():
    def __init__(self, ):
        pass


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

def open_file(infile):

    with open(infile, 'r') as f:
        N = eval(f.readline())
        X = eval(f.readline())
    # Do some basic checks to make sure the input is valid
    assert type(N) == list and type(X) == list and len(N) >= len(X), "Error. Invalid input"
    n = len(N)
    x = len(X)
    return N,n,X,x
# Write some information to the screen
#print("File    =", infile)
#print("n       =", n)
#print("x       =", x)
#print("N       =", N)
#print("X       =", X)
#print("S       =", S)
#print("cost(S) =", cost(S, N, X))
ListOfCosts = []
Cheapest = 100000000000000000000
BestSchedule = {}
numberofruns = 100000
k=1
for i in range(numberofruns):
    print("we are ", round(i / numberofruns * 100, 2 ), "percent complete")
    S = []
    for i in range(x):
        S.append(1)
    for i in range(n-x):
        S.append(0)
    random.shuffle(S)
    ListOfCosts.append(cost(S, N, X))
    if ListOfCosts[-1] < Cheapest: #checks if this is the new cheapest
        Cheapest = ListOfCosts[-1] #if it is, replaces the old cheapest number with this
        BestSchedule = [S]  #saves the solution to the BestSchedule one
    if (ListOfCosts[-1]) == Cheapest: # if this is as cheap as another soln...
        BestSchedule.append(S)  # adds the new soln to the BestSchedule part

ListOfCosts = sorted(ListOfCosts)
NoDuplicatesListOfCosts = sorted(list(set(ListOfCosts)))
FrequencyOfCosts = []
for i in NoDuplicatesListOfCosts:
    FrequencyOfCosts.append(ListOfCosts.count(i))
print(min(NoDuplicatesListOfCosts))
print(Cheapest)
print(BestSchedule) # checking it worked in registering the cheapest
plt.xlabel("Cost")
plt.ylabel("Frequency")
plt.bar(NoDuplicatesListOfCosts, FrequencyOfCosts, color = 'white')
plt.plot(NoDuplicatesListOfCosts, FrequencyOfCosts, color = 'grey')
plt.show()
