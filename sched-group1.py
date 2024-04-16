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

def open_file(infile):
    
    with open(infile, 'r') as f:
        N = eval(f.readline())
        X = eval(f.readline())
    # Do some basic checks to make sure the input is valid
    assert type(N) == list and type(X) == list and len(N) >= len(X), "Error. Invalid input"
    n = len(N)
    x = len(X)
    return N,n,X,x



myWasher = Appliance("Washing machine", [4,6,0,1])
print(myWasher.hi)
