import random
import matplotlib.pyplot as plt
import copy
import time

class Appliance():
    """
    This class simulates a household appliance, with a name and phases of a cycle.
    """
    def __init__(self, name, phases):
        self.name = name
        self.phases = phases
        self.ScheduleLength = len(phases)

    def __repr__(self):
        return f"This appliance is a {self.name} and  has a schedule length of {self.ScheduleLength}\n The schedule is {self.phases}"

class Timings():
    """
    This class holds the information about what the price of energy is during periods of time.
    """
    def __init__(self, costPerPeriod):
        self.costPerPeriod = costPerPeriod # This attribute holds the price of electricity at each period of the day 
        self.length = len(costPerPeriod)

class Solution():
    """
    This class acts as a blueprint for a solution to our problem it includes:
        > The schedule of the phases of the solution
        > The times when the appliance is on/off
        > The cost of the solution
        > The object involved in the solution
        > Multiple ways to generate a solution
    """
    def __init__(self, appliance, timings):
        self.timings = timings
        self.length = timings.length
        self.appliance = appliance
        self.onOff = [1 for i in self.appliance.phases] + [0 for i in range(self.timings.length - self.appliance.ScheduleLength)]
        random.shuffle(self.onOff)
        self.onOffToSolutionSchedule()
        self.cost = "Unknown"
        self.calculateCost()
        

    def onOffToSolutionSchedule(self):
        """
        This method uses the onOff Arry to generate a new solution s
        """
        onOff = self.onOff
        self.solutionSchedule = []
        appliancePhaseIndex = 0
        for i in onOff:
            if i == 1:
                self.solutionSchedule.append(self.appliance.phases[appliancePhaseIndex])
                appliancePhaseIndex += 1
            else:
                self.solutionSchedule.append(0)

    def calculateCost(self):
        """
        This methoid sets the cost of the solution object.
        """
        total = 0
        for i in range(self.length):
            total += self.solutionSchedule[i] * self.timings.costPerPeriod[i]
        self.cost = total

    def graph(self):
        """
        The following function graphs out the solution nicely, with the times for the phases, the number of untis needed per phase and the cost per unit in each period.
        """
        xAxis =[i for i in range(self.length)]
        fig, ax = plt.subplots(figsize = (10, 7))
        plt.title("Energy Costs and when to use an Appliance for the lowest cost")
        ax_2 = ax.twinx()
        ax.bar(xAxis, self.solutionSchedule, color = "red", width = 0.5, label = 'Energy Required')
        ax_2.plot(xAxis, self.timings.costPerPeriod, ".-b", linewidth = 3, markersize = 15, label = 'Energy Cost Over Time')

        ax.set_xlabel("Time Period")
        ax.set_ylabel("Units Required")
        ax_2.set_ylabel("Cost Per Unit")
        ax.legend(bbox_to_anchor = (0.8, 1.12))
        ax_2.legend(bbox_to_anchor = (0.5, 1.12))
        plt.show()

    def __repr__(self):
        return f"This is a solution for the appliance {self.appliance.name}, with timings for use {self.solutionSchedule} which has a cost of {self.cost}"

def open_file(file):
    """
    This function reads a specific problem and returns the approiate Appliance and Timings objects
    """
    with open(file, 'r') as f:
        timingArray = eval(f.readline())
        applianceName = f.readline()
        applianceArray = eval(f.readline())
    return Appliance(applianceName, applianceArray), Timings(timingArray)

def task1(appliance, timing, numberOfRuns):
    """
    This function takes an appliance, a timing (energy units cost per periods) and a number of runs, and gets a random solution with the appliance and times a given number of times, and returns the list of costs, the list of all generated solutions with the lowest cost, and the best cost.
    """
    ListOfCosts = []
    Cheapest = 10000000000
    for i in range(numberOfRuns):
        print("we are ", round(i / numberOfRuns * 100, 2 ), "percent complete")
        tempSolution = Solution(appliance, timing)
        ListOfCosts.append(tempSolution.cost)
        if ListOfCosts[-1] < Cheapest: #checks if this is the new cheapest
            BestSchedules = []
            Cheapest = ListOfCosts[-1] #if it is, replaces the old cheapest number with this
            BestSchedules.append(tempSolution)  #saves the solution to the BestSchedules one
        if (ListOfCosts[-1]) == Cheapest: # if this is as cheap as another soln...
            BestSchedules.append(tempSolution)  # adds the new soln to the BestSchedule part
    solutions = BestSchedules
    best_cost = min(ListOfCosts)    #i unsorted the list of costs so we can use the random selections in a graph later on
    return ListOfCosts, solutions, best_cost

def graph_task_1(ListOfCosts):
    """
    Takes the data found in task1 and gives a graph showing the distribution of the random solutions' cost. 
    """
    NoDuplicatesListOfCosts = sorted(list(set(ListOfCosts)))
    FrequencyOfCosts = []
    for i in NoDuplicatesListOfCosts:
        FrequencyOfCosts.append(ListOfCosts.count(i))
    plt.xlabel("Cost")
    plt.ylabel("Frequency")
    plt.bar(NoDuplicatesListOfCosts, FrequencyOfCosts, color = 'white')
    plt.plot(NoDuplicatesListOfCosts, FrequencyOfCosts, color = 'grey')
    plt.show()

def testForImprovements(solution):
    """
    This function swaps a one and a zero and sees if it gives a better cost then returns the best change found
    """
    improvedSolutions = [solution] #So testForImprovements doesnt return nothing
    givenSchedule = solution.solutionSchedule
    for i in range(solution.length - 1):
        if (givenSchedule[i] == 0 and givenSchedule[i+1] != 0) or (givenSchedule[i] != 0 and givenSchedule[i+1] == 0):  #Checking if 2 adjectent indexs has one as 0 and the other as a number not 0
            tempSolution = copy.deepcopy(solution)
            tempSolution.solutionSchedule[i],  tempSolution.solutionSchedule[i + 1] = tempSolution.solutionSchedule[i + 1], tempSolution.solutionSchedule[i] #Swaping both the indexs
            tempSolution.calculateCost()
            if tempSolution.cost < solution.cost:
                improvedSolutions.append(tempSolution)
            else:
                del tempSolution

    bestSoultion = findBestSolution(improvedSolutions)

    return bestSoultion 

def testForImprovements2(solution):
    """
    Prioritising the highest energy cost phase, this function attempts to swap 2 timings of the appliance to give a cheaper result.
    """
    costOfElectricity = solution.timings.costPerPeriod
    appliancePhases = solution.solutionSchedule
    #Finding gaps in appliancePhases
    findingGapsResult = findGaps(appliancePhases, costOfElectricity)
    #Find the best swap
    bestSwap = findBestCostFromSplitArray(findingGapsResult)
    #Copying the solution and making the swap
    newSolution = copy.deepcopy(solution)
    newSolution.solutionSchedule[bestSwap["indexSwap"][0]] , newSolution.solutionSchedule[bestSwap["indexSwap"][1]] = newSolution.solutionSchedule[bestSwap["indexSwap"][1]], newSolution.solutionSchedule[bestSwap["indexSwap"][0]]
    newSolution.calculateCost()
    return newSolution


def testForImprovements3(solution):
    """
    Using a while loop instead of if and else statements, swaps a timing for a phase of the appliance's schedule that makes the solution cheaper, prioritising the highest energy cost phases first.
    """
    improved_solutions = [solution] #this is so it doesnt have an empty list if there are no improvements
    best_cost = solution.cost
    current_solution = copy.deepcopy(solution)
    temp_solution = copy.deepcopy(solution)
    phases = sorted(solution.appliance.phases, reverse=True)    #reversing the phases means that its going down from the most energy expensive phase to the cheapest so it prioritises the best improvements first

    for i in phases:    
        required_index = temp_solution.solutionSchedule.index(i)    #finding where the chosen phase is in the solution Schedule
        temp_solution.solutionSchedule[required_index] = 0  #instead of deleting the value we set it to zero so we can still count where the next index we want is from this list
        k = 1   #this is the distance away from the chosen phase we are going to so by checking the one next to it then the one 2 from it, etc we get to check all possible places for it to go. 
        if required_index + 2 < current_solution.length:    #so that it doesnt try to check outside the list
            while current_solution.solutionSchedule[required_index + k] == 0:
                current_solution.solutionSchedule[required_index + k], current_solution.solutionSchedule[required_index] = current_solution.solutionSchedule[required_index], current_solution.solutionSchedule[required_index + k]
                current_solution.calculateCost()
                if current_solution.cost <= best_cost:
                    improved_solutions.append(current_solution)
                    best_cost = current_solution.cost
                current_solution = copy.deepcopy(solution)  #resetting the solution back to original so we can run it again checking each different branch for the cheapest solution
                k += 1
                if (required_index + k + 1) >= current_solution.length: #fixes it trying to find things outside of the list when we change k
                    break
        k = 1
        if required_index - 1 >= 0: #so that it doesnt try to check outside the list
            while current_solution.solutionSchedule[required_index - k] == 0:
                current_solution.solutionSchedule[required_index - k], current_solution.solutionSchedule[required_index] = current_solution.solutionSchedule[required_index], current_solution.solutionSchedule[required_index - k]
                current_solution.calculateCost()
                if current_solution.cost <= best_cost:
                    improved_solutions.append(current_solution)
                    best_cost = current_solution.cost
                current_solution = copy.deepcopy(solution)  #resetting the solution back to original so we can run it again checking each different branch for the cheapest solution
                k += 1
                if (required_index - k) < 0:    #fixes it trying to find things outside of the list when we change k
                    break
    
    bestSolutionFound = findBestSolution(improved_solutions)
    return bestSolutionFound

def veryBasicSearch(solution):
    """
    This is an increbilily basic local search algorithm for finding a better solution. It does this by swapping a 1 and 0 then recaluating the cost and excepts it if its better.
    """
    onIndexList = [i for i in range(solution.length) if solution.onOff[i] == 1] #Getting a list of all the indexs where the solution is on
    offIndexList = [i for i in range(solution.length) if solution.onOff[i] == 0] #Getting a list of all the indexs where the solution is off
    onIndexChoice = random.choice(onIndexList)
    offIndexChoice = random.choice(offIndexList)
    tempSolution = copy.deepcopy(solution)
    tempSolution.onOff[onIndexChoice], tempSolution.onOff[offIndexChoice] = tempSolution.onOff[offIndexChoice], tempSolution.onOff[onIndexChoice] #Swapping the indexs
    tempSolution.onOffToSolutionSchedule()
    tempSolution.calculateCost()
    if tempSolution.cost < solution.cost:
        return tempSolution
    else:
        return solution

def findGaps(appliancePhases, electricityPrices):
    """
    This function is how we find gaps and return them as a 2d array 
    """
    nonZeroCount = 0 #Used for tracking how many non 0 entries we pass when we see 2 we push into the dictionary
    result = []
    # Temp dictionary represents 
    tempDictionary = {
        "appliancePhases": [],
        "electricityPrices": [],
        "startIndex": 0,
        "endIndex": 0
    }
    for num, i in enumerate(appliancePhases):
        if num == len(appliancePhases) - 1: #If we are at the end of the list push the temp dictionary to the result
            tempDictionary["appliancePhases"].append(i)
            tempDictionary["electricityPrices"].append(electricityPrices[num])
            tempDictionary["endIndex"] = num
            result.append(tempDictionary)
        elif i != 0 and nonZeroCount == 1: #If we had seen 2 non zero vaules we push the temp dictionary to the result
            tempDictionary["appliancePhases"].append(i)
            tempDictionary["electricityPrices"].append(electricityPrices[num])
            tempDictionary["endIndex"] = num
            result.append(tempDictionary) #Push the dictionary to the result
            #Now we reset the tempDictionary
            tempDictionary = {
                "appliancePhases": [i],
                "electricityPrices": [electricityPrices[num]],
                "startIndex": num,
                "endIndex": 0
            }
        elif i !=0 or (num == 0 and i ==0): #If we see a non 0 or a 0 at the start of the array add the starting index
            tempDictionary["appliancePhases"].append(i)
            tempDictionary["electricityPrices"].append(electricityPrices[num])
            tempDictionary["startIndex"] = num
            nonZeroCount += 1
        else: #else add data to the dictionary
            tempDictionary["appliancePhases"].append(i)
            tempDictionary["electricityPrices"].append(electricityPrices[num])
    return result

def findBestCostFromSplitArray(splitArray):
    """
    This function now finds the best improvement after the arrays have been turned into dictionary from the find costs
    """
    bestChange = {
        "indexSwap": [0, 0],
        "changeInCost": 0
    }
    for i in splitArray:
            possibleSwaps = [i["electricityPrices"][f] for f,x in enumerate(i["appliancePhases"]) if x == 0] #Gets the prices of all the current avalible swaps
            leftSideCurrentCost = i["appliancePhases"][0] * i["electricityPrices"][0]
            rightSideCurrentCost = i["appliancePhases"][-1] * i["electricityPrices"][-1]
            if possibleSwaps != []:
                for num,x in enumerate(possibleSwaps):
                    possilbeCostForPartLeftSide = i["appliancePhases"][0] * x
                    possilbeCostForPartRightSide = i["appliancePhases"][-1] * x
                    if possilbeCostForPartLeftSide < leftSideCurrentCost and (leftSideCurrentCost - possilbeCostForPartLeftSide) > bestChange["changeInCost"]: #Checks the left side to see if there are any better costs
                        bestChange["changeInCost"] = (leftSideCurrentCost - possilbeCostForPartLeftSide)
                        bestChange["indexSwap"] = [i["startIndex"], i["startIndex"] + num + 1]
                    elif possilbeCostForPartRightSide < rightSideCurrentCost and (rightSideCurrentCost - possilbeCostForPartRightSide) > bestChange["changeInCost"]: #Checks the right side for any better costs
                        bestChange["changeInCost"] = (rightSideCurrentCost - possilbeCostForPartRightSide)
                        bestChange["indexSwap"] = [i["endIndex"], i["startIndex"] + num + 1]
    return bestChange

def findBestSolution(solutions):
    """
    Given a list of solutions, the function checks the costs of them all, and outputs the one with the lowest cost, and if there is more than one with that cost, outputs the first.
    """
    bestCost = 9999999999999
    bestSolution = solutions[0]
    for i in solutions:
        if i.cost < bestCost:
            bestSolution = i
            bestCost = i.cost 
    return bestSolution

def findBestSolutionsList(solutions):
    """
    Given a list of solutions, the function checks the costs of them all, and outputs all the ones with the lowest cost.
    """
    bestCost = 9999999999999
    bestSolutions = []
    for i in solutions:
        if i.cost < bestCost:
            bestSolutions.append(i)
            bestCost = i.cost
    return bestSolutions

def graph_2_different_solutions(solution1, solution2):  
    """
    Takes the 2 given solutions, and plots them together on 2 graphs, with the times for the phases, the number of untis needed per phase and the cost per unit in each period.
    """
    xAxis =[i for i in range(solution1.length)]
    
    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize = (15, 8.1))
    plt.subplots_adjust(left=0.04, right=0.965, bottom=0.06, top=0.9, wspace=0.2, hspace=0.2)

    ax_1b = ax_1.twinx()
    ax_1.bar(xAxis, solution1.solutionSchedule, color = "red", width = 0.5, label = 'Energy Required')
    ax_1b.plot(xAxis, solution1.timings.costPerPeriod, ".-b", linewidth = 3, markersize = 15, label = 'Energy Cost Over Time')
    ax_1.set_title('Energy Costs and when to use an Appliance for the lowest cost')
    ax_1.set_xlabel("Time Period")
    ax_1.set_ylabel("Units Required")
    ax_1b.set_ylabel("Cost Per Unit")

    ax_2b = ax_2.twinx()
    ax_2.bar(xAxis, solution2.solutionSchedule, color = "red", width = 0.5, label = 'Energy Required')
    ax_2b.plot(xAxis, solution2.timings.costPerPeriod, ".-b", linewidth = 3, markersize = 15, label = 'Energy Cost Over Time')
    ax_2.set_title('Energy Costs and when to use an Appliance for the lowest cost')
    ax_2.legend(bbox_to_anchor = (0.06, 1.07))
    ax_2b.legend(bbox_to_anchor = (0.1, 1.12))
    ax_2.set_xlabel("Time Period")
    ax_2.set_ylabel("Units Required")
    ax_2b.set_ylabel("Cost Per Unit")

    plt.show()

def graph_iterations_of_small_improvements(solution, iterations, searchFunction):
    """
    Runs the testForImprovements function a given number of times, each time taking the new best solution, and then graphing the change in the cost against the iterations.
    """
    current_best_solution = solution
    costs = [solution.cost]
    startTime = time.time()
    for i in range(iterations + 1):
        improved_solution = searchFunction(current_best_solution)
        costs.append(improved_solution.cost)
        current_best_solution = improved_solution
    endTime = time.time()
    print(f"It took {endTime - startTime} seconds to complete {iterations} of this function")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost of Schedule")
    plt.plot(costs)
    plt.show()
    return current_best_solution

def graph_iterations_against_random_selection(filename = "p1.txt", iterations = 1000):
    """
    Runs the function task1 to a given number of iterations, giving the improvements in costs over time, and then graphs it against the number of iterations.
    """
    Appliance1, Timings1 = open_file(filename)
    costs, schedules, best_cost = task1(Appliance1, Timings1, iterations)
    cheapest_cost = costs[0]
    list_of_cheapest_costs = [] #this is a list that i can put the lowest costs in as they appear over iterations
    for i in costs:
        if i <= cheapest_cost:
            cheapest_cost = i
            list_of_cheapest_costs.append(i)
        else:
            list_of_cheapest_costs.append(cheapest_cost)

    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost of Schedule")
    plt.plot(list_of_cheapest_costs)
    plt.show()

#------------------------------------------------
# End of code for functions and classes


testAppliance, testTimings = open_file("p3.txt")
#costs, schedules, best_cost = task1(testAppliance, testTimings, 100000)

testSolution = Solution(testAppliance, testTimings)

#print(best_cost)
#graph_task_1(costs)
#print(schedules[0])
#schedules[0].graph()


#The things above this in comments are what you want to print off to complete task 1.

# The following is mostly random code to check if a thing is working, can be deleted if want, needs to be deleted before hand in
#-------------------------------------------------------------------------------------------------------------------------------

#bestRandomSolution = findBestSolution(schedules)
#completely_random_solution = Solution(testAppliance, testTimings)
testForImprovementsTwoBestSolution = graph_iterations_of_small_improvements(testSolution, 1000, testForImprovements2)
veryBasicSearchBestSolution = graph_iterations_of_small_improvements(testSolution, 1000, veryBasicSearch)
print(f"Testforimprovements2()'s best cost is {testForImprovementsTwoBestSolution.cost} and veryBasicSearch()'s best cost is {veryBasicSearchBestSolution.cost}")
graph_2_different_solutions(testForImprovementsTwoBestSolution, veryBasicSearchBestSolution)

#bestFoundSolutionMethod1 = graph_iterations_of_small_improvements(bestRandomSolution, 1000)
#random_through_small_changes_graph = graph_iterations_of_small_improvements(completely_random_solution, 1000)
#graph_2_different_solutions(completely_random_solution, random_through_small_changes_graph)

#graph_2_different_solutions(bestRandomSolution, bestFoundSolutionMethod1)

#print(bestRandomSolution.cost, bestFoundSolutionMethod1.cost)


#graph_iterations_against_random_selection("p2.txt", 100000)

#solution1 = Solution(testAppliance, testTimings)
#print(solution1)

#solutions_from_solution1, best_cost_from_solution1 = testForImprovements3(solution1)
#print(solutions_from_solution1)
#print(best_cost_from_solution1)
#best_solutions = findBestSolutions(solutions_from_solution1)
#print(best_solutions)


#print(solution1)
#graph_2_different_solutions(solution1, best_solutions)
