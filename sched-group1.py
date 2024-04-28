import random
import matplotlib.pyplot as plt
import copy
import time
import math

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
    def __init__(self, appliance, timings, shuffle = True):
        self.timings = timings
        self.length = timings.length
        self.appliance = appliance
        self.onOff = [1 for i in self.appliance.phases] + [0 for i in range(self.timings.length - self.appliance.ScheduleLength)]
        if shuffle == True:
            self.getShuffledOnOff()
        else:
            self.getBlockedOnOff()
        self.onOffToSolutionSchedule()
        self.cost = "Unknown"
        self.calculateCost()

    def getShuffledOnOff(self):
        """
        This method shuffles the onOff phases randomly.
        """
        random.shuffle(self.onOff)
        self.onOffToSolutionSchedule()
        self.calculateCost()


    def getBlockedOnOff(self):
        """
        Using a block of on/off times, like if we cannot pause the appliance once it's on, this method finds the best place to place the 
        block to get the lowest cost.
        """
        listOfCosts = []
        listOfBlocks = []
        for i in range(self.timings.length + 1):
            self.onOffToSolutionSchedule()
            self.calculateCost()
            listOfCosts.append(self.cost)
            currentOnOff = copy.deepcopy(self.onOff)
            listOfBlocks.append(currentOnOff)
            self.onOff.insert(0, self.onOff[-1])
            del self.onOff[-1]
        index = listOfCosts.index(min(listOfCosts))
        self.onOff = listOfBlocks[index]


    def onOffToSolutionSchedule(self):
        """
        This method uses the onOff Arry to generate a new solution s
        """
        self.solutionSchedule = []
        appliancePhaseIndex = 0
        for i in self.onOff:
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

def task1(appliance, timing, iterations):
    """
    This function takes an appliance, a timing (energy units cost per periods) and a number of runs, and gets a random solution with the appliance and times a given number of times, and returns the list of costs, the list of all generated solutions with the lowest cost, and the best cost.
    """
    ListOfCosts = []
    Cheapest = 10000000000
    for i in range(iterations + 1):
        #print("we are ", round(i / numberOfRuns * 100, 2 ), "percent complete")
        tempSolution = Solution(appliance, timing)
        ListOfCosts.append(tempSolution.cost)
        if ListOfCosts[-1] < Cheapest: #checks if this is the new cheapest
            BestSchedules = []
            Cheapest = ListOfCosts[-1] #if it is, replaces the old cheapest number with this
            BestSchedules.append(tempSolution)  #saves the solution to the BestSchedules one
        if (ListOfCosts[-1]) == Cheapest: # if this is as cheap as another soln...
            BestSchedules.append(tempSolution)  # adds the new soln to the BestSchedule part
    best_solution = findBestSolution(BestSchedules)
    best_cost = min(ListOfCosts)    #i unsorted the list of costs so we can use the random selections in a graph later on
    return ListOfCosts, best_solution

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
    plt.title("Distribution of costs of random solutions")
    plt.bar(NoDuplicatesListOfCosts, FrequencyOfCosts, color = '#E85285')
    plt.plot(NoDuplicatesListOfCosts, FrequencyOfCosts, color = '#6A1B9A')
    plt.show()

def testForImprovements1(solution):
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

def bestNeighbourSearch(solution, iterations):
    """
    Prioritising the highest energy cost phase, this function attempts to swap 2 timings of the appliance to give a cheaper result.
    """
    
    listOfCosts = [solution.cost]
    currentSolution = solution
    for i in range(iterations):
        #Finding gaps
        costOfElectricity = currentSolution.timings.costPerPeriod
        appliancePhases = currentSolution.solutionSchedule
        #Finding gaps in appliancePhases
        findingGapsResult = findGaps(appliancePhases, costOfElectricity)
        #Find the best swap
        bestSwap = findBestCostFromSplitArray(findingGapsResult)
        #Copying the solution and making the swap
        newSolution = copy.deepcopy(currentSolution)
        newSolution.solutionSchedule[bestSwap["indexSwap"][0]] , newSolution.solutionSchedule[bestSwap["indexSwap"][1]] = newSolution.solutionSchedule[bestSwap["indexSwap"][1]], newSolution.solutionSchedule[bestSwap["indexSwap"][0]]
        newSolution.calculateCost()
        currentSolution = newSolution
        listOfCosts.append(currentSolution.cost)
    return currentSolution, listOfCosts

    

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

def findNeighbour(solution):
    """
    This function takes a solution and returns a random neighbouring solution
    """
    onIndexList = [i for i in range(solution.length) if solution.onOff[i] == 1] #Getting a list of all the indexs where the solution is on
    offIndexList = [i for i in range(solution.length) if solution.onOff[i] == 0] #Getting a list of all the indexs where the solution is off
    onIndexChoice = random.choice(onIndexList)
    offIndexChoice = random.choice(offIndexList)
    tempSolution = copy.deepcopy(solution)
    tempSolution.onOff[onIndexChoice], tempSolution.onOff[offIndexChoice] = tempSolution.onOff[offIndexChoice], tempSolution.onOff[onIndexChoice] #Swapping the indexs
    tempSolution.onOffToSolutionSchedule()
    tempSolution.calculateCost()
    return tempSolution

def hillClimbSearch(solution, iterations):
    """
    This is an increbilily basic local search algorithm for finding a better solution. It does this by swapping a 1 and 0 then recaluating the cost and excepts it if its better.
    """
    currentSolution = copy.deepcopy(solution)
    listOfCosts = [solution.cost]
    for i in range(iterations):
        neighbourSoultion = findNeighbour(currentSolution)
        if neighbourSoultion.cost <= currentSolution.cost:
            currentSolution = neighbourSoultion
        listOfCosts.append(currentSolution.cost)       
    return currentSolution, listOfCosts


def simulatedAnnealingSearch(solution, iterations):
    """
    A local search function using a simulated annealing techinque 
    """
    #Setting initial temperature conditions
    temperature = 1
    minTemperature = 0.0001
    alpha = 0.9
    bestSolution = solution
    currentSolution = solution
    listOfCosts = [solution.cost]
    
    while temperature > minTemperature:
        for i in range(iterations):
            if currentSolution.cost < bestSolution.cost:
                bestSolution = currentSolution
            newSolution = findNeighbour(currentSolution)
            acceptanceProbability = math.exp((currentSolution.cost - newSolution.cost)/ temperature) #Calcuating the new acceptance probability for the new solution
            if acceptanceProbability > random.uniform(0,1):
                currentSolution = newSolution
        
        temperature *= alpha
        listOfCosts.append(currentSolution.cost)
    
    return bestSolution, listOfCosts
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
    bestSolution, costs = searchFunction(solution, iterations)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost of Schedule")
    plt.title(f"Iterations against small improvements")
    plt.plot(costs)
    plt.show()
    return bestSolution

def graph_iterations_against_random_selection(solution, iterations = 1000):
    """
    Runs the function task1 to a given number of iterations, giving the improvements in costs over time, and then graphs it against the number of iterations.
    """
    costs, best_schedules = task1(solution.appliance, solution.timings, iterations)
    cheapest_cost = costs[0]
    list_of_cheapest_costs = []
    for i in costs:
        if i <= cheapest_cost:
            cheapest_cost = i
            list_of_cheapest_costs.append(i)
        else:
            list_of_cheapest_costs.append(cheapest_cost)

    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost of Schedule")
    plt.title("Iterations against costs from random selection solution")
    plt.plot(list_of_cheapest_costs)
    plt.show()

def graph_iterations_of_small_improvements_and_random_selection(solution, iterations, searchFunction):
    """
    This combines the 2 functions which plot their graphs of cost improvements against iteration to show a more direct comparison.
    """
    bestSearchSolution, search_costs = searchFunction(solution, iterations)
    random_improving_costs, random_best_schedules = task1(solution.appliance, solution.timings, iterations)
    cheapest_cost = random_improving_costs[0]
    print(random_improving_costs[0])
    print(search_costs[0])
    list_of_cheapest_random_costs = [random_improving_costs[0]]
    for i in random_improving_costs:
        if i <= cheapest_cost:
            cheapest_cost = i
            list_of_cheapest_random_costs.append(i)
        else:
            list_of_cheapest_random_costs.append(cheapest_cost)

    plt.subplots(figsize = (10, 7))
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost of Schedule")
    plt.title(f"Improvements in cost over iterations between {searchFunction.__name__} and Random Selection")
    plt.plot(search_costs, label = "Costs From Search Function")
    plt.plot(list_of_cheapest_random_costs, label = "Costs From Random Generation")
    plt.legend()
    plt.show()

def graphTwoSoultionFinders(solution, iterations, searchFunction1, searchFunction2):
    """
    This combines the 2 functions which plot their graphs of cost improvements against iteration to show a more direct comparison.
    """
    bestSearchSolution, search_costs = searchFunction1(solution, iterations)
    bestSearchSolution2, search_costs2 = searchFunction2(solution, iterations)

    print(f"{searchFunction1.__name__} best solutions cost is {bestSearchSolution.cost} and {searchFunction2.__name__} best solution's cost is {bestSearchSolution2.cost}")
    plt.subplots(figsize = (10, 7))
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost of Schedule")
    plt.title(f"Improvements in cost over iterations between {searchFunction1.__name__} and {searchFunction2.__name__}")
    plt.plot(search_costs, label = f"Costs From {searchFunction1.__name__}")
    plt.plot(search_costs2, label = f"Costs From {searchFunction2.__name__}")
    plt.legend()
    plt.show()


#--------------------------------------------------------------------------------------------------------------------------------------------
def print_task1(filename = "p2.txt", iterations = 100_000):
    """
    This is a function with all our print statements for completing task 1 in.
    """
    givenAppliance, givenTimings = open_file(filename)
    givenSolution = Solution(givenAppliance, givenTimings, shuffle = True)
    costs, best_solution = task1(givenAppliance, givenTimings, iterations)

    print(best_solution)
    graph_task_1(costs)
    best_solution.graph()

def print_task2(filename = "p2.txt", iterations = 1_000, searchFunction = hillClimbSearch):
    """
    This is a function with all our print statements for completing task 2 in.
    """
    givenAppliance, givenTimings = open_file(filename)
    givenSolution = Solution(givenAppliance, givenTimings, shuffle = True)
    startTime1 = time.time()
    basic_search_solution, list_of_costs = searchFunction(givenSolution, iterations)
    endTime1 = time.time()
    startTime2 = time.time()
    listOfCosts, best_solution_task1 = task1(givenAppliance, givenTimings, iterations)
    endTime2 = time.time()
    print(f"It took {endTime2 - startTime2} seconds to generate {iterations}  random solutions and the best solution found was {best_solution_task1}")
    print(f"It took {endTime2 - startTime2} seconds to complete {iterations} iterations of this {searchFunction.__name__} and the best solution found was {basic_search_solution}")
    
    graph_2_different_solutions(best_solution_task1, basic_search_solution)     #this plots a graph with the best solution found from randomly generating solutions on the left and using the basic search algorithm on the right.
    
    graph_iterations_of_small_improvements_and_random_selection(givenSolution, iterations, searchFunction)

#--------------------------------------------------------------------------------------------------------------------------------------------

#print_task1(filename="p3.txt", iterations=100000)      #change the filename between "p1.txt", "p2.txt" and "p3.txt" for each problem, and if needed you can change the iterations as well.

#--------------------------------------------------------------------------------------------------------------------------------------------

#print_task2(filename="p2.txt", iterations=1_000, searchFunction=hillClimbSearch)      #change the filename between "p1.txt", "p2.txt" and "p3.txt" for each problem, and change the searchFunction between testForImprovements1, testForImprovements2, testForImprovements3 and veryBasicSearch and if needed you can change the iterations as well.

#--------------------------------------------------------------------------------------------------------------------------------------------

#testAppliance, testTimings = open_file("p3.txt")
#testSolution = Solution(testAppliance, testTimings)
#graphTwoSoultionFinders(testSolution, 100, simulatedAnnealingSearch, hillClimbSearch)


def time_of_function(filename = "p2.txt", iterations = 10_000, searchFunction1 = hillClimbSearch, number_of_mean_repeats = 20):
    """
    This function will calculate the average time a function takes to complete in the document p3. The code is largely copied from the code for task 2, but tweaked to show the information we want instead.
    """
    for i in range(number_of_mean_repeats):
        print(f"we are {i} repeats in.")
        givenAppliance, givenTimings = open_file(filename)
        givenSolution = Solution(givenAppliance, givenTimings, shuffle = True)
        startTime1 = time.time()
        basic_search_solution, list_of_costs = searchFunction1(givenSolution, iterations)
        endTime1 = time.time()
        startTime2 = time.time()
        listOfCosts, best_solution_task1 = task1(givenAppliance, givenTimings, iterations)
        endTime2 = time.time()
        sum_time_to_complete =+ (endTime1 - startTime1)
        sum_time_to_complete2 =+ (endTime2 - startTime2)
    mean_time_to_complete = sum_time_to_complete/number_of_mean_repeats
    mean_time_to_complete2 = sum_time_to_complete2/number_of_mean_repeats

    short_iterations = math.floor(iterations / 10)

#this bit runs it to 1/10 of the repeats so we can compare the scalability of the function.
    for j in range(number_of_mean_repeats):
        print(f"we are {j} repeats into the shorter iterations.")
        givenAppliance, givenTimings = open_file(filename)
        givenSolution = Solution(givenAppliance, givenTimings, shuffle = True)
        startTime1b = time.time()
        basic_search_solution, list_of_costs = searchFunction1(givenSolution, short_iterations)
        endTime1b = time.time()
        startTime2b = time.time()
        listOfCosts, best_solution_task1 = task1(givenAppliance, givenTimings, short_iterations)
        endTime2b = time.time()
        sum_time_to_completeb =+ (endTime1b - startTime1b)
        sum_time_to_complete2b =+ (endTime2b - startTime2b)
    mean_time_to_completeb = sum_time_to_completeb/number_of_mean_repeats
    mean_time_to_complete2b = sum_time_to_complete2b/number_of_mean_repeats
    print(f"It took a mean time of {mean_time_to_complete} seconds to complete {searchFunction1} to {iterations} iterations with the mean of {number_of_mean_repeats} tries calculated. The first 10% of the iterations took {mean_time_to_completeb/mean_time_to_complete * 100}% of the time.")
    print(f"It took a mean time of {mean_time_to_complete2} seconds to complete task 1's random function to {iterations} iterations with the mean of {number_of_mean_repeats} tries calculated. The first 10% of the iterations took {mean_time_to_complete2b/mean_time_to_complete2 * 100}% of the time.")
    #print(f"It took a mean time of {mean_time_to_complete3} seconds to complete {searchFunction2} to {iterations} iterations with the mean of {number_of_mean_repeats} tries calculated")
    #graph_2_different_solutions(best_solution_task1, basic_search_solution)     #this plots a graph with the best solution found from randomly generating solutions on the left and using the basic search algorithm on the right.
    
    #graph_iterations_of_small_improvements_and_random_selection(givenSolution, iterations, searchFunction1)

time_of_function(filename = "p2.txt", iterations = 10_000, searchFunction1 = hillClimbSearch, number_of_mean_repeats = 50)
