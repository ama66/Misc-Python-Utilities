# Min Rewards (Q4 H)
# Imagine that you're a teacher who's just graded the final exam in a class. 
# You have a list of student scores on the final exam in a particular order (not necessarily sorted), 
# and you want to reward your students. You decide to do so fairly by giving them arbitrary rewards 
# following two rules: first, all students must receive at least one reward; second, any given student must 
# receive strictly more rewards than an adjacent student (a student immediately to the left or to the right)
# with a lower score and must receive strictly fewer rewards than an adjacent student with a higher score. 
# Assume that all students have different scores; in other words, the scores are all unique. Write a function
#  that takes in a list of scores and returns the minimum number of rewards that you must give out to 
#  students, all the while satisfying the two rules.
# Sample input: [8, 4, 2, 1, 3, 6, 7, 9, 5]
# Sample output: 25 ([4, 3, 2, 1, 2, 3, 4, 5, 1])

## Brute Force Sol
# O(n^2) time | O(n) space - where in is the length of the input array

def minRewards(scores):

    rewards = [1 for _ in scores]

    ## Scan from left to right starting at second element 
    
    for i in range(1, len(scores)):
        
        ## Keep Track of Previous element index 
        j = i - 1
        ## if scores are increasing give i reward(+1) relative to j 
        if scores[i] > scores[j]:

            rewards[i] = rewards[j] + 1

        else:
            ## need to go back all the way to local max and stop there 
            while j >= 0 and scores[j] > scores[j + 1]:

                rewards[j] = max(rewards[j], rewards[j + 1] + 1)

                j -= 1

    return sum(rewards)

minRewards([4, 2, 1, 3])  # answer =8 


minRewards([8, 4, 2, 1, 3, 6, 7, 9, 5])


# Solution 2 More optimized Copyright 
# O(n) time | O(n) space - where in is the length of the input array

## Find local minima and fan out in both directions left and right and add rewards 

def minRewards(scores):
    rewards = [1 for _ in scores]
    localMinIdxs = getLocalMinIdxs(scores)
    ## Iterate over all local minima and fan out/expand from there giving rewards all the way to local maxima 
    for localMinIdx in localMinIdxs:
        expandFromLocalMinIdx(localMinIdx, scores, rewards)
    return sum(rewards)

def getLocalMinIdxs(array):
    ## Edge Case
    if len(array) == 1:
        return [0]
    ## initialize local minima array 
    localMinIdxs = []
    
    for i in range(len(array)):
        ## Case of local minima at i=0
        if i == 0 and array[i] < array[i + 1]:
            localMinIdxs.append(i)
        ## Case of local minima at end of array 
        if i == len(array) - 1 and array[i] < array[i - 1]:
            localMinIdxs.append(i)
        ## More general case where it is not local minima just skip it 
        if i == 0 or i == len(array) - 1:
            continue
        ## for point in the middle check both sides to ensure i is local minima 
        if array[i] < array[i + 1] and array[i] < array[i - 1]:
            localMinIdxs.append(i)
    return localMinIdxs

def expandFromLocalMinIdx(localMinIdx, scores, rewards):
    ## this function is called in a loop , iterating over all local minima indices 
    
    ## first expand to the left ! 
    leftIdx = localMinIdx - 1
    while leftIdx >= 0 and scores[leftIdx] > scores[leftIdx + 1]:
        rewards[leftIdx] = max(rewards[leftIdx], rewards[leftIdx + 1] + 1)
        leftIdx -= 1
        
    ## then expand to the right 
    rightIdx = localMinIdx + 1
    while rightIdx < len(scores) and scores[rightIdx] > scores[rightIdx - 1]:
        ## Note that we need not check max(rewards[rightidx],rewards[rightidx-1]+1)
        ## because we only do that when we fan out to the left because we could be overwriting 
        ## values from previous right fan out from previous local minima 
        rewards[rightIdx] = rewards[rightIdx - 1] + 1
        rightIdx += 1
    
    
 minRewards([8, 4, 2, 1, 3, 6, 7, 9, 5])  
 
 
 ## an Even more optimized solution 
# O(n) time | O(n) space - where in is the length of the input array
def minRewards(scores):
    ## really do not need to find the minima just keep fanning to the right and ignore the minima itself!
    rewards = [1 for _ in scores]
    for i in range(1, len(scores)):
        ## This loop will continue skipping the indices of local minima 
        if scores[i] > scores[i - 1]:
            rewards[i] = rewards[i - 1] + 1
            
    ## now expand backwards and fix issues with rewards! 
    for i in reversed((range(len(scores) - 1))):
        if scores[i] > scores[i + 1]:
            rewards[i] = max(rewards[i], rewards[i + 1] + 1)
    return sum(rewards)
    
    
    minRewards([8, 4, 2, 1, 3, 6, 7, 9, 5])  
    
    
