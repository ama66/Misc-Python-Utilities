
# def getNeighbors(i, j, board):
    
#     kernel=[[-1,-1],[-1,0],[-1,1],
#             [0,-1],[0,1],
#             [1,-1],[1,0],[1,1]
#            ]
    
#     neighbors = []
    
#     for k in kernel:
#         index_i=i+k[0]
#         index_j=j+k[1]
#         if index_i >=0 and index_i <= len(board)-1 and index_j >=0 and index_j <= len(board[0])-1:
#             neighbors.append([index_i,index_j])
            
#     return neighbors


#!/usr/bin/env python
# coding: utf-8

# In[90]:


# Session 2
# Medium https://leetcode.com/problems/validate-binary-search-tree/
## Definition for a binary tree node.
## time complexity O(N) because you need to traverse every single node
## space complexity O(log(N)) for a balanced tree otherwise O(N) for a linkedlist like tree (one deep branch!)
## and this is the worst case scenario 

class TreeNode:
  def __init__(self, x):
    self.val = x
    self.left = None
    self.right = None

class Solution:
  def isValidBST(self, root):
    def helper(node, lower, upper):
    # Base Case if you are null (call from a leaf node!)
      if not node:
        return True
      # current node value 
      val = node.val
    ## need to make sure that I am between the lower and upper range passed 
    ## Note the equal sign is important for a [1,1] tree as right and left has to be strictly 
    # greater and smaller than current node val 
      if val <= lower or val >= upper:
        return False
    ## now recurse left and right and pass appropriate lower and upper limits 
    ## for the right tree my value is the lower limit
      if not helper(node.right, val, upper):
        return False
    ## for the left tree my value is the upper limit.
      if not helper(node.left, lower, val):
        return False
      ## if I get here after recursion and I do not exist then return True
      return True
    return helper(root, float('-inf'), float('inf'))

node = TreeNode(5)
node.left = TreeNode(4)
node.right = TreeNode(7)
print(Solution().isValidBST(node))
# True


# In[91]:


# Session 3
# Easy: Ransom Note Expect to solve quickly probably need to go through 2-3 like these in a 45 minute session 
#https://leetcode.com/problems/ransom-note/
## brute force would be O(n.m) for each letter scan the magazine and find letters replace them with sentinel
## space is O(1) for this solution
## best solution is hashmap preprocess the magazine  key=letter: value= frequency do this for the magazine.
# this is O(N) process , single pass algorithm.  space is O(N) for this solution. 
## make sure note is contained in the magazine. scan the note and subtract one each time you find a letter 
## until you go below 0...if that ever happens return False! so O(N+M) is time complexity. space is O(n)

import collections

class Solution:
  def canConstruct(self, ransomNote, magazine):
    mag_dict = collections.defaultdict(int)
    for char in magazine:
      mag_dict[char] += 1
    for char in ransomNote:
      mag_dict[char] -= 1  ## note that if char is not already in the dictionary it will be set to 0 
      if mag_dict[char] < 0:
        return False
    return True

print(Solution().canConstruct('aa', 'aab'))
# True


# In[92]:


## Session 4:) Add two numbers as a linked list
# Medium https://leetcode.com/problems/add-two-numbers/
class Node(object):
  def __init__(self, x):
    self.val = x
    self.next = None


class Solution:
  def addTwoNumbers(self, l1, l2):
   # return self.addTwoNumbersRecursive(l1, l2, 0)
    return self.addTwoNumbersIterative(l1, l2)

  def addTwoNumbersRecursive(self, l1, l2, c):
    val = l1.val + l2.val + c
    c = val // 10        ### 6+7 =13 so for 13 the carry is 1 which is 13//10 or floor(13/10)
    ## the outcome of the addition is 3 which is the remainder of dividing 13 by 10 so 13 mod 10 =3 
    ret = Node(val % 10)
    ## as I iterate backwards towards more significant numbers if I do not find a node i replace it by 0 
    ## node if I still have non-zero node in the other number 
    
    ## If I at least have one nonzero next node 
    if l1.next != None or l2.next != None:
      if not l1.next: ## if l1.next is empty/null set it to zero node 
        l1.next = Node(0)
      if not l2.next:
        l2.next = Node(0)
    ## if the above conditions are not satisified we just pass next element to the recursive call 
      ret.next = self.addTwoNumbersRecursive(l1.next, l2.next, c)
    ## If I exhausted all other elements and I still have a carry I create a new node with the carry and return it 
    elif c:
      ret.next = Node(c)
    
    return ret

  def addTwoNumbersIterative(self, l1, l2):
    ## initial states
    a = l1
    b = l2
    c = 0
    ret = current = None
    ## iterate over numbers while at least one is not None
    while a or b:
      print("adding a +  b ", a.val,"+", b.val)
      val = a.val + b.val + c
      c = val // 10
      ## initially current is None 
      if not current:
        ## ret always refers to the initial position of current
        ## for next iterations current will represent the sum of the current l1 and l2
        ret = current = Node(val % 10)
      else:
        current.next = Node(val % 10)
        ## Advance current
        current = current.next
        
      ## if at least one number is not empty
      if a.next or b.next:
        if not a.next:
          a.next = Node(0)
        if not b.next:
          b.next = Node(0)
      elif c:
        current.next = Node(c)
      ## Advance a and b and reiterate 
    
      a = a.next
      b = b.next
      print("hi",current.val, ret.val)
    return ret
###############################
### l1= 342 
l1 = Node(2)
l1.next = Node(4)
l1.next.next = Node(9)
## l2 = 465
## Answer = 465+342 = 807 so 7 > 0 > 8 should be returned 
l2 = Node(5)
l2.next = Node(6)
l2.next.next = Node(4)

answer = Solution().addTwoNumbers(l1, l2)
while answer:
  print(answer.val, end=' ')
  answer = answer.next
# 7 0 8


# In[93]:


## Session5:  three sum problem 
## https://leetcode.com/problems/3sum/
## O(N^2) time and O(1) space 
class Solution:
  def threeSum(self, nums):
    res = []
    nums.sort()
    for i in range(len(nums) - 2):
      ## avoid duplication in case two consecutive numbers are the same 
      if i > 0 and nums[i] == nums[i - 1]:
        continue
      # two sum
    ## Left pointer
      j = i + 1
    ## Right pointer
      k = len(nums) - 1
      while j < k:
        if nums[i] + nums[j] + nums[k] == 0:
          res.append([nums[i], nums[j], nums[k]])
          ## avoid duplication 
          while j < k and nums[j] == nums[j + 1]:
            j += 1
          ## avoid duplication 
          while j < k and nums[k] == nums[k - 1]:
            k -= 1
          ## move both pointers after appending 
          j += 1
          k -= 1
        elif nums[i] + nums[j] + nums[k] > 0:
          k -= 1
        else:
          j += 1
      # end two sum
    return res

print(Solution().threeSum([-1, 0, 1, 2, -1, -4]))
# [[-1, -1, 2], [-1, 0, 1]]


# In[94]:


############################################################
## Session 6 
# https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
## First and Last Indices of an Element in a Sorted Array
###########################################################
class Solution:
  def getRange(self, arr, target):
    ## LAst parameter (findfirst) indicates whether we want first or last occurence
    if arr==[]:
        return [-1,-1]
    first = self.binarySearchIterative(arr, 0, len(arr) - 1, target, True)
    last = self.binarySearchIterative(arr, 0, len(arr) - 1, target, False)
    return [first, last]

  def binarySearch(self, arr, low, high, target, findFirst):
    if high < low:
      return -1
    mid = low + (high - low) // 2
    if findFirst:
        ## if I find target at mid but target is greater than previous element It means I found
        ## the lower index/bound and I return it immediately otherwise I Continue! 
      if (mid == 0 or target > arr[mid - 1]) and arr[mid] == target:
        return mid
    
      if target > arr[mid]:
        return self.binarySearch(arr, mid + 1, high, target, findFirst)
      else:
        return self.binarySearch(arr, low, mid - 1, target, findFirst)
    
    else:
    ## Same deal if i am at the end then it is the highest index/bound or if i am at a number < next one
    ## like 9,10...as opposed to 9,9,10...
      if (mid == len(arr)-1 or target < arr[mid + 1]) and arr[mid] == target:
        return mid
    
      elif target < arr[mid]:
        return self.binarySearch(arr, low, mid - 1, target, findFirst)
      else:
        return self.binarySearch(arr, mid + 1, high, target, findFirst)

  def binarySearchIterative(self, arr, low, high, target, findFirst):
    while True:
      if high < low:
        return -1
      mid = low + (high - low) // 2
      if findFirst:
        if (mid == 0 or target > arr[mid - 1]) and arr[mid] == target:
          return mid
        if target > arr[mid]:
          low = mid + 1
        else:
          high = mid - 1
      else:
        if (mid == len(arr)-1 or target < arr[mid + 1]) and arr[mid] == target:
          return mid
        elif target < arr[mid]:
          high = mid - 1
        else:
          low = mid + 1

arr = [1, 3, 3, 5, 7, 8, 9, 9, 9, 15]
x = 9
print(Solution().getRange(arr, x))
# [6, 8]


# In[95]:


## Session 7 
## Permutations

class Solution:
  def permute(self, nums):
    res = []

    def permuteHelper(start):
        
      if start == len(nums) - 1:
        res.append(nums[:])
        
      for i in range(start, len(nums)):
        ## Make a choice! 
        nums[start], nums[i] = nums[i], nums[start]
        ## Explore Deeper
        permuteHelper(start + 1)
        ## Unexplore (restore to state before the choice)
        nums[start], nums[i] = nums[i], nums[start]
        
    permuteHelper(0)
    return res

print(Solution().permute([1, 2, 3]))
# [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 2, 1], [3, 1, 2]]


# In[96]:


def perm_eic(A):
# base case 
    if len(A)==0:
        return []
    comb=[]
    for i in range(len(A)):
        cur=A[i]
        perms=perm_eic(A[0:i]+A[i+1:])
        if len(perms)==0:
            return [[cur]]
        else:
            for p in perms:
                p.insert(0,cur)
                comb.append(p) 
    return comb


# In[97]:


A=[1,2,3]
perm_eic(A)


# In[98]:


## Session 8 
# Sorting a list with 3 unique numbers
def sortNums(nums):
  counts = {}
  for n in nums:
    counts[n] = counts.get(n, 0) + 1    ### get(n,0) if you cannot find key set value to 0 
    
  return ([1] * counts.get(1, 0) +
          [2] * counts.get(2, 0) +
          [3] * counts.get(3, 0))


# In[99]:


def sortNums2(nums):
  one_index = 0
  three_index = len(nums) - 1
  index = 0
  while index <= three_index:
    if nums[index] == 1:
      nums[index], nums[one_index] = nums[one_index], nums[index]
      one_index += 1
      index += 1
    elif nums[index] == 2:
      index += 1
    elif nums[index] == 3:
      nums[index], nums[three_index] = nums[three_index], nums[index]
      three_index -= 1
  return nums


# In[100]:


print(sortNums2([3, 3, 2, 1, 3, 2, 1]))
# [1, 1, 2, 2, 3, 3, 3]


# In[101]:


# session 9 https://leetcode.com/problems/queue-reconstruction-by-height/
# Queue Reconstruction By Height
class Solution:
  def reconstructQueue(self, people):
    people.sort(key=lambda x: (-x[0], x[1]))  # O(nlogn)
    res = []
    for p in people:  # O(N)
      res.insert(p[1], p)  # O(n)
    return res

  # Time Complexity O(N^2)
  # Space : O(N)

print(Solution().reconstructQueue([[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]))
# [[5, 0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]


# In[102]:


a=[[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]
a.sort(key=lambda x: (-x[0], x[1])) ### sort descending based on first elem, ascending based on second elem
a


# In[103]:


b=[8,4,3,0,1]
b.sort(key=lambda x: -x)
b


# In[104]:


c=list(filter(lambda x: True if x<5 else False , b))
c


# In[105]:


# xor of two similar numbers=0  for ex 4^4=0
print(4^4, 7^7)


# In[106]:


print(0^1,1^0)


# In[107]:


3^4^5==(3^4)^5


# In[108]:


3^4^5==3^(4^5)


# In[109]:


3^4^5==(4^3)^5


# In[110]:


3^4^5==(5^3)^4==5^3^4


# In[111]:


7^0 


# In[112]:


##Session 10  is associative (regroup as you wish) , commutative (order does not matter)
# https://leetcode.com/problems/single-number/
class Solution(object):
  def singleNumber(self, nums):
    occurrence = {}

    for n in nums:
      occurrence[n] = occurrence.get(n, 0) + 1

    for key, value in occurrence.items():
      if value == 1:
        return key

  def singleNumber2(self, nums):
    unique = 0
    for n in nums:
      unique ^= n
    return unique

print(Solution().singleNumber2([4, 3, 2, 4, 19, 3, 2]))
# 1


# In[113]:


0^0 , 0^1 , 0^7,3^3


# In[114]:


## iterate over a tuple containing iterable objects
#L=("aa","bf","vf")
L=([1,2],[5,6])
for i,j in zip(*L):
    print(i,j)


# In[115]:


pair=('xww', 'wxyz')
for i,j in zip(*pair):
    print(i,j)


# In[116]:


t = zip([3],[4],[3])
print("1:",*t)
list(t) # t is now empty


# In[117]:


## zipping Values in Python
z=zip([1,2,3],['a','b','c'],['#','*','$'])
for i in z:
    print(i)


# In[118]:


## zipping Values in Python
z=zip([1,2,3],['a','b','c'],['#','*','$'])
## Unzipping Values in Python (the inverse/opposite of the process described above)
a,b,c=zip(*z)
# for i,j,k in zip(*z):
#     print(i,j,k)
print(a,b,c)


# In[119]:


words=["xww","wxyz","wxyw","ywx","ywz"]
for pair in zip(words,words[1:]):
   # "pair is a tuple containing iterable strings you can unpack the strings to compare letter by letter"
    print("*"*20,pair,"*"*20)
    for letter1,letter2 in zip(*pair):
        print(letter1,letter2)


# In[120]:


# glue words together by ''.join(words) ='xwwwxyzwxywywxywz' then apply a set() function to get unique character set
# {'w', 'x', 'y', 'z'}


# In[121]:


def create_graph(words):
    letters=set(''.join(words))
    graph_x={letter: set() for letter in letters}
    for pair in zip(words,words[1:]):
        ## loop over each consecutive words  for example ("xww","wxyz")
        for before,after in zip(*pair):
            if before !=after:
                graph_x[after].add(before)
                break
                ## exit out of the for loop as soon as a letter from one word does not match the corresponding 
                ## letter in the next word. After is the dependency and before is the prereq
    return graph_x

def create_job_dep(words):
    
    g=create_graph(words)
    jobs=list(g.keys())
    deps=[]
    for key, val in g.items():
        gd=[[i,key] for i in val]
        deps.extend(gd)
     
    return jobs , deps


# In[122]:


words=["xww","wxyz","wxyw","ywx","ywz"]
#create_graph(words)
jobs,deps=create_job_dep(words)


# In[123]:


jobs


# In[124]:


deps


# In[125]:


# Topological Sort

# You are given a list of arbitrary jobs that need to be completed; these jobs are represented by integers. 
# You are also given a list of dependencies. A dependency is represented as a pair of jobs where the first 
# job is prerequisite of the second one. In other words, the second job depends on the first one; 
# it can only be completed once the first job is completed. Write a function that takes in a list of jobs and
# a list of dependencies and returns a list containing a valid order in which the given jobs can be completed. 
# If no such order exists, the function should return an empty list.

# Sample input: [1, 2, 3, 4], [[1, 2], [1, 3], [3, 2], [4, 2], [4, 3]]
# Sample output: [1, 4, 3, 2] or [4, 1, 3, 2]


# O(j + d) time | O(j + d) space

def topologicalSort(jobs, deps):
    jobGraph = createJobGraph(jobs, deps)
    return getOrderedJobs(jobGraph)

def createJobGraph(jobs, deps):
    ## Add all JobNodes This merely adds vertices to the graph 
    graph = JobGraph(jobs)
    ##  Add prereqs (edges of the graph!)
    for prereq, job in deps:
        graph.addPrereq(job, prereq)
    return graph

def getOrderedJobs(graph):
    orderedJobs = []
    nodes = graph.nodes
    ### nodes is a list of all JobNodes to be sorted
    while len(nodes):
        node = nodes.pop()
        containsCycle = depthFirstTraverse(node, orderedJobs)
        if containsCycle:
            return []
    return orderedJobs


# In[126]:


def depthFirstTraverse(node, orderedJobs):
    ## if node has been visited before return False! 
    if node.visited:
        return False
    ## if returning to a node that is in progress(visiting state) it means we found a cycle! 
    if node.visiting:
        return True
    
    ## otherwise start DFS 
    ## First Mark the Node as in progress/visiting
    node.visiting = True
    ## iterate over prereqs
    ## if we started by a node with no prereqs this fo loop will be skipped
    ## and this job will be added to the orderedJobs
    for prereqNode in node.prereqs:
        ## this recursive call will continue DFS until you reach a node with no prereqs then backtrack and continue DFS
        containsCycle = depthFirstTraverse(prereqNode, orderedJobs)
        if containsCycle:
            return True
    ### now that you visited all prereqs you can safely mark node as "visited" and annul its visiting state
    node.visited = True
    node.visiting = False
    orderedJobs.append(node.job)
    return False

class JobGraph:
    def __init__(self, jobs):
        ## will keep track of all nodes in a list of JobNodes (see class below)
        self.nodes = []
        ## In order to check membership we quickly build this hashtable where job number is key
        ## and value is the corresponding JobNode 
        
        self.graph = {}
        for job in jobs:
            self.addNode(job)
        ## this will create graph nodes and put them in the dictionary self.graph and the list self.nodes
        ## Note that as this point we did not create dependencies (edges of the graph!)

    def addPrereq(self, job, prereq):
        jobNode = self.getNode(job)
        prereqNode = self.getNode(prereq)
        jobNode.prereqs.append(prereqNode)

    def addNode(self, job):
        self.graph[job] = JobNode(job)
        self.nodes.append(self.graph[job])

    def getNode(self, job):
        if job not in self.graph:
            self.addNode(job)
        return self.graph[job]

class JobNode:
    def __init__(self, job):
        self.job = job
        self.prereqs = []
        self.visited = False
        self.visiting = False

## alien dictionary problem 

words=["xww","wxyz","wxyw","ywx","ywz"]
#create_graph(words)
jobs,deps=create_job_dep(words)

# jobs= [1, 2, 3, 4] 
# deps=[[1, 2], [1, 3], [3, 2], [4, 2], [4, 3]]
topologicalSort(jobs,deps)

## 1 preq >> dep 2  , 1 preq >> dep 3 , 3 preq >> dep 2 , 4 preq >> dep 2  , 4 preq >> dep 3


# In[127]:


## Session 53 FB 
# https://leetcode.com/problems/merge-intervals/
class Solution:
  def merge(self, intervals):
#     def takeFirst(elem):
#       return elem[0]
#     intervals.sort(key=takeFirst)
    intervals.sort(key=lambda x: x[0])
    res = []
    for interval in intervals:
      if not res or res[-1][1] < interval[0]:
        res.append(interval)
      else:
        res[-1][1] = max(res[-1][1], interval[1])
    return res


print(Solution().merge([[1, 5], [2, 8], [10, 12]]))


# In[128]:


## https://leetcode.com/problems/meeting-rooms-ii/
## Session 35 


# In[129]:


### Session 30 of 62
class Node(object):
  def __init__(self, val, next=None):
    self.val = val
    self.next = next
## define print for this class 
  def __str__(self):
    c = self
    answer = ''
    while c:
      answer += str(c.val) if c.val else ""
      c = c.next
    return answer

## brute force solutin 

def merge(lists):
## put everything in a long array
  arr = []
  for node in lists:
    while node:
      arr.append(node.val)
      node = node.next
   
  head = root = None
  for val in sorted(arr):
    if not root:
      head = root = Node(val)
    else:
      root.next = Node(val)
      root = root.next
  return head

## mergesort type  solution O(Nk)

def merge2(lists):
    
  head = current = Node(-1)

  while any(list is not None for list in lists):
        
    current_min, i = min((list.val, i) for i, list in enumerate(lists) if list is not None)
    ## get the minimum element in all nodes and advance pointer of the linkedlist that has the minimum 
    lists[i] = lists[i].next
    ## append to new linked list 
    current.next = Node(current_min)
    current = current.next

  return head.next


a = Node(1, Node(3, Node(5)))
b = Node(2, Node(4, Node(6)))

print(a)
# 135
print(b)
# 246
print(merge2([a, b]))
# 123456


# In[130]:


listl=[[-10,13],[2,-5],[-50,2]]
min((val, i) for i,val in enumerate(listl))


# In[131]:


min(((val, i) for i,val in enumerate(listl)))


# In[132]:


## PriorityQ or min heap solution. push everything in a min heap
## pop all items of the min heap into a list!
import heapq
def mergeKLists(lists):
    heap=[]
    for llist in lists:
        cur=llist
        while cur:
            heapq.heappush(heap,cur.val)
            cur=cur.next 
    if heap:
        Head= Node(heapq.heappop(heap))
        Nodee=Head
        while heap:
            Nodee.next=Node(heapq.heappop(heap))
            Nodee=Nodee.next
        return Head
    else:
        return Node(None).next
a = Node(1, Node(3, Node(5)))
b = Node(2, Node(4, Node(6)))

print(mergeKLists([a,b]))


# In[133]:


# https://leetcode.com/problems/intersection-of-two-arrays/
# session 46 
class Solution:
  def intersection(self, nums1, nums2):
    results = {}
    for num in nums1:
      if num in nums2 and num not in results:
        results[num] = 1
    return list(results.keys())

  def intersection2(self, nums1, nums2):
    set1 = set(nums1)
    set2 = set(nums2)
    return [x for x in set1 if x in set2]

  def intersection3(self, nums1, nums2):
    hash = {}
    duplicates = {}
    for i in nums1:
      hash[i] = 1
    for i in nums2:
      if i in hash:
        duplicates[i] = 1

    return tuple(duplicates.keys())
print(Solution().intersection3([4, 9, 5], [9, 4, 9, 8, 4]))
# (9, 4)


# In[134]:


## https://leetcode.com/problems/intersection-of-two-arrays-ii/submissions/
## return duplicates (as many as there is in both arrays)
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        hash1={}
        duplicates={}
        ## unique numbers in nums1 and their frequency
        for i in nums1:
            hash1[i]=hash1.get(i,0)+1 
  
        ## 
        for i in nums2:
            if i in hash1 and hash1[i]>0:
                duplicates[i]=duplicates.get(i,0)+1 
                hash1[i]-=1
      
        return self.find_intersec(duplicates)
    
    def find_intersec(self,dups):
        res=[]
        for k,v in dups.items():
            for i in range(v):
                res.append(k)
        return res
    
                
print(Solution().intersect([1,2,2], [9, 2, 2, 1]))
         


# In[135]:


## Session 50 
# Tree Serialization
# Coding Sessions

class Node:
  def __init__(self, val, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

  def __str__(self):
    result = ''
    result += str(self.val)
    if self.left:
      result += str(self.left)
    if self.right:
      result += str(self.right)
    return result


def serialize(node):
  if node == None:
    return '#'
  return str(node.val) + ' ' + serialize(node.left) + ' ' + serialize(node.right)


def deserialize(str):
    
  def deserialize_helper(values):
    value = next(values)
    print(value)
    if value == '#':
      return None
    node = Node(int(value))
    node.left = deserialize_helper(values)
    node.right = deserialize_helper(values)
    return node

  values = iter(str.split()) ## remove spaces and yield an iterable 
  return deserialize_helper(values)


#      1
#     / \
#    3   4
#   / \   \
#  2   5   7
tree = Node(1)
tree.left = Node(3)
tree.right = Node(4)
tree.left.left = Node(2)
tree.left.right = Node(5)
tree.right.right = Node(7)
string = serialize(tree)
print("serialized string: ",string)
print(deserialize(string))
# 132547


# In[136]:


stri="1 # 2"  
values = iter(stri.split())
print(next(values))
print(next(values))


# In[137]:


stri.split() ## remove spaces 


# In[138]:


## Session 62
def fib(n):
  a = 0
  b = 1
  if n == 0:
    return a
  if n == 1:
    return b

  for _ in range(2, n+1):
    value = a + b

    a = b
    b = value
  return value
fib(3)


# In[139]:


## Session 11
# Definition for singly-linked list.
class ListNode:
  def __init__(self, x):
    self.val = x
    self.next = None

  def __str__(self):
    result = str(self.val)
    if self.next:
      result += str(self.next)
    return result

class Solution:
    def reverseList(self, head):
        prev = None
        curr = head
        while (curr != None):
          temp = curr.next
          curr.next = prev
          prev = curr
          curr = temp
        return prev
    def reverseList_rec(self, head):
        ## base case either tail calling none or single item list 
        if head==None or head.next==None:
            return head 
        
        cur=head
        return self.reverseList_rec_h(cur)
## Recursive Version of Linked List Reversal 

    def reverseList_rec_h(self, node):
        if node.next==None:
            return node
        nextelem=node.next
        ## Very important to annul this link so that we remove cyclicity 
        node.next=None
        rev_list_rest=self.reverseList_rec_h(nextelem)
        nextelem.next=node
        
        return rev_list_rest

node = ListNode(1)
node.next = ListNode(2)
node.next.next = ListNode(3)

#print(Solution().reverseList(node))
print(Solution().reverseList_rec(node))
# 321


# In[140]:


#https://leetcode.com/problems/remove-nth-node-from-end-of-list/submissions/
# Session 22
class Node:
  def __init__(self, val, next):
    self.val = val
    self.next = next

  def __str__(self):
    n = self
    answer = ''
    while n:
      answer += str(n.val)
      n = n.next
    return answer

def remove_kth_from_linked_list(node, k):
  slow, fast = node, node
  for i in range(k):
    fast = fast.next
  if not fast:
    return node.next

  prev = None
  while fast:
    prev = slow
    fast = fast.next
    slow = slow.next
  prev.next = slow.next
  return node

head = Node(1, Node(2, Node(3, Node(4, Node(5, None)))))
print(head)
# 12345

head = remove_kth_from_linked_list(head, 1)
print(head)
# 1234


# In[141]:


head = Node(1, Node(2, None))
print(head)
# 12345

head = remove_kth_from_linked_list(head, 1)
print(head)


# In[142]:


a="wew"
"x"+a


# In[143]:


## https://leetcode.com/problems/add-binary/
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        i=len(a)-1
        j=len(b)-1
        sum=0
        s=""
        carry=0
        while i>=0 or j>=0 : 
            sum+=carry
            if i>=0:
                sum+=int(a[i])
                i-=1
            if j>=0:
                sum+=int(b[j])
                j-=1
            s=str(sum%2)+s
            carry=sum//2 
            sum=0
        
        return s if carry==0 else str(1)+s
                
Solution().addBinary("11","1")  ## '100'


# In[144]:


## https://leetcode.com/problems/add-strings/submissions/
## similar to addbinary only difference is the basis
class Solution(object):
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        i=len(num1)-1
        j=len(num2)-1
        sum=0
        s=""
        carry=0
        while i>=0 or j>=0 : 
            sum+=carry
            if i>=0:
                sum+=int(num1[i])
                i-=1
            if j>=0:
                sum+=int(num2[j])
                j-=1
            s=str(sum%10)+s
            carry=sum//10 
            sum=0
        
        return s if carry==0 else str(1)+s
    


# In[145]:


## https://leetcode.com/problems/product-of-array-except-self/
## Session 17
### these solutions 1,2,3  are not acceptable because the question says do not use division !


class Solution(object):
    def productExceptSelf1(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        s=1 
        prod=[1 for i in range(len(nums))]
        for i in nums[:-1]:
            s*=i
            
        prod[-1]=s
      
        for i in reversed(range(len(nums)-1)):
            prod[i]=int(prod[i+1]*nums[i+1]/nums[i])
        return prod 
    
    ## constant space version 
    def productExceptSelf2(self, nums):
        
        s=1 
        for i in nums[:-1]:
            s*=i
            
        lastprod=s
        lastnum=nums[-1]
        nums[-1]=lastprod 
        
        for i in reversed(range(len(nums)-1)):
            temp=nums[i]
            nums[i]=int(lastprod*lastnum/nums[i])
            lastprod=nums[i]
            lastnum=temp 
        return nums
    ## constant space version 
    
    def productExceptSelf3(self, nums):
        productall=1 
        for i in nums:
            productall*=i
        for i in range(len(nums)):
            nums[i]=int(productall/nums[i])
        return nums
    
    def productExceptSelf4(self, nums):
        leftprod=[1 for i in nums]
        rightprod=[1 for i in nums]
        s=1
        for i in range(len(nums)):
            leftprod[i]=s*nums[i]
            s=leftprod[i]
        s=1
        for i in reversed(range(len(nums))):
            rightprod[i]=s*nums[i]
            s=rightprod[i]
        for i in range(len(nums)):
            if i>=1 and i < len(nums)-1 : 
                nums[i]=leftprod[i-1]*rightprod[i+1]
            if i==0:
                nums[i]=rightprod[1]
            if i==len(nums)-1:
                nums[i]=leftprod[len(nums)-2]
        return nums
                
    
                
print(Solution().productExceptSelf1([2,3,4,5]))

print(Solution().productExceptSelf2([2,3,4,5]))
        
print(Solution().productExceptSelf3([2,3,4,5]))
Solution().productExceptSelf4([2,3,4,5])


# In[146]:


## Session 17
## note that you can use the output array without additional space! so O(1) additional space. Right side
## prod is not totally needed. you could start from the end of the array and get the relevant Right value
## and slide it backwards ! 

class Solution:
  def productExceptSelf(self, nums):
    res = [1] * len(nums)
    for i in range(1, len(nums)):
      res[i] = res[i-1] * nums[i-1]
    
    right = 1
    
    for i in range(len(nums) - 2, -1, -1):
      right *= nums[i+1]
      res[i] *= right
    return res

print(Solution().productExceptSelf([2,3,4,5]))


# In[147]:


### Session 24 
## Find the Kth Largest Element in a List

import heapq
import random


def findKthLargest(nums, k):
  return sorted(nums)[len(nums) - k]


def findKthLargest2(nums, k):
  return heapq.nlargest(k, nums)[-1]


def findKthLargest3(nums, k):
    
  def select(list, l, r, index):
    if l == r:
      return list[l]

    pivot_index = random.randint(l, r)
    # move pivot to the beginning of list
    list[l], list[pivot_index] = list[pivot_index], list[l]
    
    # partition
    i = l
    for j in range(l + 1, r + 1):
      if list[j] < list[l]:
        i += 1
        list[i], list[j] = list[j], list[i]
    # move pivot to the correct location
    list[i], list[l] = list[l], list[i]
    
    # recursively partition one side
    if index == i:
      return list[i]
    elif index < i:
      return select(list, l, i - 1, index)
    else:
      return select(list, i + 1, r, index)

  return select(nums, 0, len(nums) - 1, len(nums) - k)


print(findKthLargest3([3,2,1,5,6,4], 2))
# 5


# In[148]:


# https://leetcode.com/problems/k-closest-points-to-origin/submissions/
# Session 60 of 62

import heapq

class Solution(object):
    def kClosest(self, points, K):
        """
        :type points: List[List[int]]
        :type K: int
        :rtype: List[List[int]]
        """
        ## store data in a min heap
        data=[(self.calcdist(x),x)  for x in points]
        #
        heapq.heapify(data)
        
        res=[]
        
        for i in range(K):
            res.append(heapq.heappop(data)[1])

        return res
        
    def calcdist(self,point):
        return point[1]*point[1]+point[0]*point[0]
    


# In[149]:


## Next Permutation
# Good Explanation here 
# https://github.com/bephrem1/backtobackswe/blob/3b21637af6f9be6d1d32dcef94c8a3c04d74cefa/Arrays%2C%20Primitives%2C%20Strings/NextPermutation/NextPermutation.java
## FB Problem https://leetcode.com/problems/next-permutation/

class Solution(object):
    
    def swap(self,i,j,nums):
        nums[i],nums[j]=nums[j],nums[i]
        
    def reverse(self,nums):
        left=0
        right=len(nums)-1
        while left < right:
            self.swap(left,right,nums)
            left+=1
            right-=1
        return nums 
    
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        ## First/larget point  point x in decreasing sequence from the end [.......x,x-o,x-o-g,...etc]
        N=len(nums)
        ## initialize at the next to last element and walk backwards to x
        
        i=N-2
        
        while i>=0 and nums[i]>=nums[i+1]:
            i-=1
        
        ## Now point at i is where the next sequence/permutation will change. Now we need to pick from x-o,x-o-g...etc nums[-1]] 
        ## the next element greater than nums[i]
        
        ## if i =0 the whole array will be reversed basically! 
    
        if i<0:
            nums=self.reverse(nums)
            print(nums)
            return nums 
        else:
            j=N-1
            while j>i and nums[j] <= nums[i]:
                j-=1
            self.swap(i,j,nums)
            ## now reverse the array ahead of i 
            nums[i+1:]=self.reverse(nums[i+1:])
            
            return nums 

Solution().nextPermutation([1,2,3])    


# In[150]:


## Merge two sorted arrays in place
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        ## start at the end of the first array nums1 because it has empty places at the end
        ## start comparing large elements to large elements from the end of nums1 and num2 and work your way back to i=i and j=0 
        ## Edge Case
        if n==0:
            return nums1 
        else: 
            k= m + n -1 
            i=m-1
            j=n-1  ## start here at nums2 and walk backwards/leftwards
            
            while j >=0 : 
                if i>=0 and nums1[i] > nums2[j]:
                    nums1[k]=nums1[i]
                    i-=1
                else:
                    nums1[k]=nums2[j]
                    j-=1
                k-=1
            return nums1 
        
      
Solution().merge([1,2,3,0,0,0],3,[2,5,6],3) ## Expected [1, 2, 2, 3, 5, 6]


# In[151]:


## FB https://leetcode.com/problems/monotonic-array/
class Solution(object):
    
    def isMonotonic(self, A):    
        IsIncreasing=True 
        IsDecreasing=True 
        ## Walk through the array and try to invalidate it 
        for i in range(len(A)-1):
            if A[i]< A[i+1]:
                IsDecreasing=False
            if A[i] > A[i+1]:
                IsIncreasing=False
      
        return IsIncreasing or IsDecreasing

Solution().isMonotonic([1,2,2,3])


# In[152]:


class Solution(object):
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        ## BaseCase
        if len(s)==1:
            return True 
        i=0
        j=len(s)-1
        while i<j:
            if s[i]!=s[j]:
                return self.IsPalindrome(s[i+1:j+1]) or self.IsPalindrome(s[i:j])
            else:
                i+=1
                j-=1
                
        return True 
        
        return False 
    
    def IsPalindrome(self, st):
        i=0
        j=len(st)-1
       
        while i<j:
            if st[i]!=st[j]:
                return False
            else:
                i+=1
                j-=1
        
        return True 
Solution().validPalindrome("deeee")


# In[153]:


Solution().validPalindrome("atbbga")


# In[154]:


# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        low=0
        high=n
        middle=low 
        
        while low <= high:
            middle= (low + high) // 2
            ## if solution is found return it otherwise do binary search
            if isBadVersion(middle) and not isBadVersion(middle-1):
                return middle 
            elif isBadVersion(middle):
                high=middle-1 
            else:
                low=middle+1 
                
        return middle 
## this is only for testing case where input is 5,4 meaning 4,5 are bad versions so 4 is the first one 
## in the [1,2,3,4,5] versions list 
def isBadVersion(n):
    if n<4:
        return False
    else:
        return True 
    
Solution().firstBadVersion(5) ## expected 4 


# In[155]:


## FB https://leetcode.com/problems/valid-palindrome/submissions/
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if s=="":
            return True 
        ## Exclude Non-AlphaNumeric characters
        s= ''.join(i for i in s if i.isalnum()).lower()
        
        i=0
        j=len(s)-1
       
        while i<j:
            
            if s[i]!=s[j]:
                return False
            else:
                i+=1
                j-=1
        
        return True 

Solution().isPalindrome("deer")


# In[156]:


# FB Question https://leetcode.com/problems/diameter-of-binary-tree/
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        if root==None:
            return 0 
        
        leftheight=self.heightOfBinaryTree(root.left) 
        rightheight=self.heightOfBinaryTree(root.right)
        
        leftdiameter=self.diameterOfBinaryTree(root.left) 
        rightdiameter=self.diameterOfBinaryTree(root.right)
        
        ## Definition is about number of edges so no need to add 1 to include_root_height 
        
        include_root_height=   leftheight + rightheight
        
        not_include_root_height= max(leftdiameter,rightdiameter)
        
        return max( include_root_height, not_include_root_height)
        
    
    def heightOfBinaryTree(self,root):
        if root==None:
            return 0 
        
        leftheight=self.heightOfBinaryTree(root.left) 
        rightheight=self.heightOfBinaryTree(root.right)
        
        if leftheight > rightheight:
            h=1+leftheight
        else:
            h=1+rightheight
        return h 
    
    
    


# In[157]:


# FB Q ## https://leetcode.com/problems/closest-binary-search-tree-value/submissions/
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def closestValue_iter(self, root, target):
        """
        :type root: TreeNode
        :type target: float
        :rtype: int
        """
## iterative Solution O(1) space 
        closest = root.val
    
        while root:
            closest = min(root.val, closest, key = lambda x: abs(target - x))
            root = root.left if target < root.val else root.right
        return closest
    
    ## Recursive O(Log N) space and time 
    def closestValue_recur(self, root, target):
    ## initialize and do Binary Search // Recursive Solution 
        closest=root.val
        return self.binarysearchclosest(root,closest,target)
    
    def binarysearchclosest(self,node,closest,target):
        if not node:
            return closest 
               
        if abs(node.val-closest) < abs(closest-target):
            closest=node.val
            
        if node.val > target:
            return self.binarysearchclosest(node.left,closest,target)
        else:
            return self.binarysearchclosest(node.right,closest,target)
            
#input: [4,2,5,1,3]
#3.714286
## output 4


# In[158]:


## Goat Latin FB https://leetcode.com/problems/goat-latin/submissions/ 
class Solution(object):
    def toGoatLatin(self, S):
        """
        :type S: str
        :rtype: str
        """
        def convert(word):
            if word[0] not in 'aeiouAEIOU':
                word = word[1:] + word[:1]
            return word + 'ma'

        return " ".join(convert(word) + 'a' * i for i, word in enumerate(S.split(), 1))
    
Solution().toGoatLatin("I speak Goat Latin") ## Expected "Imaa peaksmaaa oatGmaaaa atinLmaaaaa"


# In[159]:


## Note second argument of enumerate =1 means start index at 1 rather than 0 
S="I love u"
[(i,word) for i, word in enumerate(S.split(), 1)]  


# In[160]:


## verify an alien dictionary FB https://leetcode.com/problems/verifying-an-alien-dictionary/

# for pair in zip(words,words[1:]):
#     ## loop over each consecutive words  for example ("xww","wxyz")
#     for before,after in zip(*pair):
#         if before !=after:
#             graph_x[after].add(before)
#             break
# Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
# Output: true
# Explanation: As 'h' comes before 'l' in this language, then the sequence is sorted.


# In[162]:


class Solution(object):
    def isAlienSorted(self, words, order):
        order_index = {c: i for i, c in enumerate(order)}

        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i+1]

            # Find the first difference word1[k] != word2[k].
            for k in range(min(len(word1), len(word2))):
                # If they compare badly, it's not sorted.
                if word1[k] != word2[k]:
                    if order_index[word1[k]] > order_index[word2[k]]:
                        return False
                    break
            else:
                # If we didn't find a first difference, the
                # words are like ("app", "apple").
                if len(word1) > len(word2):
                    return False

        return True
Solution().isAlienSorted(["kuvp","q"],"ngxlkthsjuoqcpavbfdermiywz")


# In[164]:


## my solution works for all edge cases 
## FB https://leetcode.com/problems/verifying-an-alien-dictionary/submissions/

from collections import defaultdict
class Solution(object):
    def isAlienSorted(self, words, order):
        """
        :type words: List[str]
        :type order: str
        :rtype: bool
        """
        worddic=defaultdict()
        
        for i,char in enumerate(order):
            worddic[char]=i
        ## keep track of index so you can compare words if they agreed on every letter
        ## in case one got chopped 
        for ind, pair in enumerate(zip(words,words[1:])):
            wordA=pair[0]
            wordB=pair[1]
            disagreed=False 
            ## if they ever disagree we reassign disagreed to True 
            for letterA,letterB in zip(*pair):
                if letterA!=letterB:
                    disagreed=True 
                    if worddic[letterA] > worddic[letterB]:
                        return False 
                        break 
                    else: 
                        break    
                        
            if disagreed==False and len(words[ind]) > len(words[ind+1]):
                return False
                
        return True 
                
Solution().isAlienSorted(["kuvp","q"],"ngxlkthsjuoqcpavbfdermiywz")


# In[ ]:


## Top K Frequent Element  (like for the example the top 3 most frequent elements of an array)
class Solution:
  def topKFrequent(self, nums, k):
    count = collections.defaultdict(int)
    for n in nums:
      count[n] += 1
    
    ## this can be accomplished in one liner in python
    ## count=collections.Counter(nums)
    heap = []
    
    for key, v in count.items():
      heapq.heappush(heap, (v, key))
      if len(heap) > k:
        heapq.heappop(heap)
    res = []
    while len(heap) > 0:
      res.append(heapq.heappop(heap)[1])
    return res
## maintaining size k heap can be accomplished also in one liner in more pythonic way
## heapq.nlargets(k,count.keys, ley=count.get)
## Time Complexity:
## O(n) for building the hashmap
## O(n logk) for pushing into the heap the elements of the hashmap note that in worst case
## hashmap size is n if all elements of the array are unique
## space complexity is O(n) for the hashmap and O(k) for the heap so overall
## Time is O(n log k ) and space is O(n)


# In[181]:


## Unique Paths using Recursion + Memoization 
# https://leetcode.com/problems/unique-paths/submissions/
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        memo={}
        i=0
        j=0
        coords=(i,j)
        return self.unique_path_helper(coords,m,n,memo)
    
    def unique_path_helper(self,grid,m,n,memo):
       
        ic,jc=grid
        
        ## BaseCases 
        ##(1) out of bounds stop path
        if ic > m or jc > n:
            return 0
        ## Reached target 
        if ic==m-1 and jc==n-1:
            return 1 
        ## check if key is in memoization dictionary and return it if found (Significant speedup)
        if (ic,jc) in memo.keys():
            return memo[(ic,jc)]
        ## from each point explore two depths (to the right [0,1] and down [1,0])
        ## if you hit the target through any or all of the two paths return 1 and add it to local sum
        ## which represents number of paths going through ic,jc that hit the target. ic,jc propagate this
        ## knowledge back up the recursion tree all the way to the origin (0,0)
        ## solution without memoization leads to time limit exceeded on leetcode 
        localsum=0
        for incr in [[1,0],[0,1]]:
            new_i=ic+incr[0]
            new_j=jc+incr[1]
            localsum+=self.unique_path_helper((new_i,new_j),m,n,memo)
            
        memo[(ic,jc)]=localsum 
        return localsum
    
Solution().uniquePaths(4,6) ## Expected 56


# In[180]:


# Bottom up Approach using Dynamic Programming. Fill out first row with ones as there is only 1 way to get
## to these points by moving to the right. Fill out first column by ones as only way to get there is by going down. 
## at any other point i,j you can come there either from above (i-1,j) or left (i,j-1)
#https://leetcode.com/problems/unique-paths/submissions/
class Solution:
  def uniquePathsDP(self, m, n):
    matrix = [[0 for i in range(n)] for j in range(m)]
    for i in range(m):
      matrix[i][0] = 1
    for j in range(n):
      matrix[0][j] = 1
    for i in range(1, m):
      for j in range(1, n):
        matrix[i][j] = matrix[i][j-1] + matrix[i-1][j]
    return matrix[m-1][n-1]
Solution().uniquePathsDP(4,6)


# In[183]:


## Session 12 Maximum In A Stack
## https://leetcode.com/articles/max-stack/#
class MaxStack(object):
  def __init__(self):
    self.stack = []
    self.maxes = []

  def push(self, val):
    self.stack.append(val)
    if self.maxes and self.maxes[-1] > val:
      self.maxes.append(self.maxes[-1])
    else:
      self.maxes.append(val)

  def pop(self):
    if self.maxes:
      self.maxes.pop()
    return self.stack.pop()

  def max(self):
    return self.maxes[-1]

s = MaxStack()
s.push(1)
s.push(2)
s.push(3)
s.push(2)
print('max', s.max())
print(s.pop())
print('max', s.max())
print(s.pop())
print('max', s.max())
print(s.pop())
print('max', s.max())
print(s.pop())


# In[184]:


## similarly min heap 
#https://leetcode.com/problems/min-stack/submissions/

class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack=[]
        self.minima=[]
        

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.stack.append(x)
        if self.minima and x > self.minima[-1]:
            self.minima.append(self.minima[-1])
        else:
            self.minima.append(x)
            
    def pop(self):
        """
        :rtype: None
        """
        if self.minima:
            self.minima.pop()
            
        return self.stack.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.minima[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


# In[187]:


#### FB https://leetcode.com/problems/course-schedule/
## Detecting cycle in a DG
class Solution:
  def canFinish(self, numCourses, prerequisites):
    graph = collections.defaultdict(list)
    for edge in prerequisites:
      graph[edge[0]].append(edge[1])

    visited = set()

    # True if there is a cycle, False if not
    def visit(vertex):
      visited.add(vertex)
      for neighbour in graph[vertex]:
        if neighbour in visited or visit(neighbour):
          return True
      visited.remove(vertex)
      return False

    for i in range(numCourses):
      if visit(i):
        return False
    return True


# In[8]:


def findPythagoreanTriplets(nums):
  for a in nums:
    for b in nums:
      for c in nums:
        if a*a + b*b == c*c:
          return True
  return False

def findPythagoreanTriplets2(nums):
  squares = set([n*n for n in nums])

  for a in nums:
    for b in nums:
      if a * a + b * b in squares:
        return True , (a,b,a*a+b*b)
  return False

print(findPythagoreanTriplets2([3, 5, 12, 5, 13]))
# True


# In[14]:


### Push Dominos 
# https://leetcode.com/problems/push-dominoes/
class Solution:
  def pushDominoes(self, dominoes):
    N = len(dominoes)
    force = [0] * N

    # Populate Rs
    f = 0
    for i in range(N):
      if dominoes[i] == 'R':
        f = N
      elif dominoes[i] == 'L':
        f = 0
      else:
        f = max(f-1, 0)
      force[i] += f
    print(force)
    # Populate Ls 
    for i in range(N-1, -1, -1):
      if dominoes[i] == 'L':
        f = N
      elif dominoes[i] == 'R':
        f = 0
      else:
        f = max(f-1, 0)
      force[i] -= f
    print(force)
    print(dominoes)
    for i in range(N):
      if force[i] == 0:
        force[i] = '.'
      elif force[i] > 0:
        force[i] = 'R'
      else:
        force[i] = 'L'
    return "".jo
in(force)

Solution().pushDominoes(".L.R...LR..L..")


# In[19]:


## https://leetcode.com/problems/basic-calculator/description/  Simple Calculator Hard Category 
class Solution(object):
  def __eval_helper(self, expression, index):
    op = '+'
    result = 0
    while index < len(expression):
      char = expression[index]
      if char in ('+', '-'):
        op = char
      else:
        ## make sure you have a value initialized 
        value = 0
        if char.isdigit():
          value = int(char)
        elif char == '(':
          (value, index) = self.__eval_helper(expression, index + 1)
        if op == '+':
          result += value
        if op == '-':
          result -= value
      index += 1
    
    return (result, index)

  def eval(self, expression):
    return self.__eval_helper(expression, 0)[0]

print(Solution().eval('(1 + (2 + (3 + (4 + 5))))'))
# 15


# In[18]:


## https://leetcode.com/problems/evaluate-reverse-polish-notation/submissions/
## Evaluate postfix expression 
def integer_divide_towards_zero(a, b):
    return -(abs(a) // abs(b)) if a*b < 0 else a // b
    
class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        nums=[]  ## stack for storing number stream
        for t in tokens:
            if t in ["*","/","+","-"]:
                n2=int(nums.pop())
                n1=int(nums.pop())
                if t=="+":
                    nums.append(n1+n2)
                if t=="-":
                    nums.append(n1-n2)
                if t=="*":
                    nums.append(n1*n2)
                if t=="/":
                    nums.append(integer_divide_towards_zero(n1,n2))
            else:
                nums.append(t)
                
    
        return nums.pop()
                
Solution().evalRPN(["2","1","+","3","*"])


# In[23]:


class Solution(object):
    def calculate(self, expression):
        """
        :type s: str
        :rtype: int
        """
        index=0 
        op="+"
        result=0 
        val=0
        
        while index < len(expression):
          char = expression[index]
          if char in ('+', '-'):
            op = char
            index+=1
          else:
            if  expression[index].isdigit():
                val,index=self.read_number(expression,index)
                if op=="+":
                    result+=val
                if op == '-':
                    result -= val
                    
            else: ## for spaces just increase the index 
                index+=1
                
        return result
    
    def read_number(self,exp,ide):
        val=""
        while ide < len(exp) and exp[ide].isdigit():
            val+= exp[ide]
            ide+=1

        return int(val),ide
            
Solution().calculate("10 + 24")     
            


# In[24]:


## Longest Sequence with Two Unique Numbers

def findSequence(seq):
  if len(seq) < 2:
    return len(seq)

  a, b = seq[0], seq[1]

  last_num = b
  last_num_count = 1 if (a == b) else 0
  length = 1

  max_length = 1

  for n in seq[1:]:
    
    if n in (a, b):
      length += 1
      if b == n:
        last_num_count += 1
      else:
        last_num = a
        last_num_count = 1
        
    else:
      a = last_num
      b = n
      last_num = b
    
      length = last_num_count + 1
        
      last_num_count = 1
    
    max_length = max(length, max_length)
    
    
  return max_length


print(findSequence([1, 3, 5, 3, 1, 3, 1, 5]))
# 4
print(findSequence([1, 1, 3, 3, 4, 4]))
# 4


# In[27]:


## https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/
class Solution(object):
    def lengthOfLongestSubstringKDistinct(self, st, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        ## invariant: a window starting s and ending e with at most k distinct characters before 
        ## the start of a new iteration. info is saved in a hashset that gets modified 
        maxlen=0 ##keep track of max window length
        window_dic={} ## this hashset keeps track of frequency of elements within the window 
        ## if dictionary size goes beyond k we move start pointer and decrease the count
        ## of the element at previous s location. if count=0 we remove it from set an update
        ## max length , otherwise we continue to traverse the array by increasing e
        
        s=e=0
        ## Traverse The String update dictionary , s, e make sure len(window_dic)<=k
        while e < len(st):
            c=st[e]
            if c in window_dic:
                window_dic[c]+=1
            else:
                window_dic[c]=1
                
            ## move c to the right and remove elements by moving c to the right 
            while len(window_dic) > k : 
                c=st[s]
                window_dic[c]-=1
                if window_dic[c]==0:
                    del window_dic[c]
                s+=1
            ## once you remove extra element check window size  and update maxlen 
            maxlen=max(maxlen,e-s+1)
            e+=1
            
            
        return maxlen 
    
Solution().lengthOfLongestSubstringKDistinct("eceba",2)


# In[ ]:


# ## word ladder 127 https://leetcode.com/problems/word-ladder/
# # level 0 modification (original word)
# ## leve 1 modifications (original word with one position changed)
# ## level 2 modifications (orginal word with two positions changed)
# ##...etc until level n where all n positions of a word are completely changed! 
# ## Each level comprises a lot of possible words however not all of them are contained within the
# ## given word list..so need to construct this implied graph on the fly
# ## the solution requires creation of a BFS on a graph using queue and look for final word
# ## it is possible that you will find it at some level of modification which would be closest
# ## to original word..this is why BFS is needed here 
# initial_state="Initial"
# waiting_state="Waiting"
# visited_state="Visited"
# from string import ascii_lowercase

# def replace_char(node,position,ch):
#     new = list(node.word)
#     new[position] = ch
#     node.word=''.join(new)
#     print(node.state)
#     return node.word
       
# class wordnode:
#     def __init__(self,word,state):
#         self.word=word
#         self.state=state 
        
# class Solution(object):
    
#     def ladderLength(self, beginWord, endWord, wordList):
#         """
#         :type beginWord: str
#         :type endWord: str
#         :type wordList: List[str]
#         :rtype: int
#         """
#         wordlst=collections.defaultdict()
        
#         for i in wordList:
#              wordlst[i]=wordnode(i,initial_state)
                
#         if endWord not in wordlst:
#             return 0 
        
#         ## Breadth First Search on an implied word graph 
#         level = 0
#         word_len = len(beginWord) 
#         qu=[]
#         qu.append(wordnode(beginWord,waiting_state))
        
#         while qu:
#             level+=1
#             # level_items=len(qu)
#             # for i in range(level_items):
#             #     print(level_items)
#             print("queue",[i.word for i in qu])
            
#             Node=qu.pop(0)
#             Node.state=visited_state 
            
#             ## First Generate Children of the word (those with higher level of change compared to original word )
#             New_Node=wordnode(Node.word,initial_state)
            
#             print("popped",New_Node.word)
            
#             for pos in range(word_len):
#                 orig_char =  New_Node.word[pos]
#                 for char in ascii_lowercase:  ## a-z
#                     New_Node.word=replace_char(New_Node,pos,char)
#                     if  New_Node.word == endWord:
#                         return level + 1
#                     if  New_Node.word in wordlst.keys():  ## it has to be a word of interest (in the given list)
#                         print(New_Node.word,New_Node.state,wordlst[New_Node.word].state, "checked in list")
#                         if wordlst[New_Node.word].state==initial_state:
#                             qu.append(New_Node)
#                             wordlst[New_Node.word].state=waiting_state
#                             print(New_Node.word, "pushed")
#                             print("qq",[i.word for i in qu])

#                 New_Node.word =replace_char(New_Node,pos,orig_char) 
                
#         print("quitting",[i.word for i in qu])
#         return 0 
    


# In[58]:


# from string import ascii_lowercase
# import collections
# def replace_char(word,position,ch):
#     new = list(word)
#     new[position] = ch
#     word=''.join(new)
#     return word
        
# class Solution(object):
    
#     def ladderLength(self, beginWord, endWord, wordList):
#         """
#         :type beginWord: str
#         :type endWord: str
#         :type wordList: List[str]
#         :rtype: int
#         """
#         wordlst=collections.defaultdict()
        
#         for i in wordList:
#              wordlst[i]=1
                
#         if endWord not in wordlst:
#             return 0 

#         ## Breadth First Search on an implied word graph 
#         level = 0
#         word_len = len(beginWord) 
#         qu=[]
#         qu.append(beginWord)
        
#         while qu:
            
#             Node=qu.pop(0)
        
#             ## First Generate Children of the word (those with higher level of change compared to original word 
#             for pos in range(word_len):
#                 orig_char =  Node[pos]
#                 for char in ascii_lowercase:  
#                     word_cand=replace_char(Node,pos,char)
#                     if  word_cand== endWord:
#                         return level +1
#                     if  word_cand in wordlst.keys():  ## it has to be a word of interest (in the given list)
#                             qu.append(word_cand)
#                             level+=1
#                           #  print("word cand", word_cand,level)
#                             del wordlst[word_cand]

#                 Node =replace_char(Node,pos,orig_char) 
                
#         return 0 
                            
                         
# Solution().ladderLength("a","c",["a","b","d","c"])  


# In[60]:


Solution().ladderLength("hot","dog",["hot","dog","dot"])  


# In[64]:


from collections import defaultdict
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """

        if endWord not in wordList or not endWord or not beginWord or not wordList:
            return 0

        # Since all words are of same length.
        L = len(beginWord)

        # Dictionary to hold combination of words that can be formed,
        # from any given word. By changing one letter at a time.
        all_combo_dict = defaultdict(list)
        for word in wordList:
            for i in range(L):
                # Key is the generic word
                # Value is a list of words which have the same intermediate generic word.
                all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)
                

        print(all_combo_dict)
        # Queue for BFS
        queue = collections.deque([(beginWord, 1)])
        # Visited to make sure we don't repeat processing same word.
        visited = {beginWord: True}
        while queue:
            current_word, level = queue.popleft()      
            for i in range(L):
                # Intermediate words for current word
                intermediate_word = current_word[:i] + "*" + current_word[i+1:]

                # Next states are all the words which share the same intermediate state.
                for word in all_combo_dict[intermediate_word]:
                    # If at any point if we find what we are looking for
                    # i.e. the end word - we can return with the answer.
                    if word == endWord:
                        return level + 1
                    # Otherwise, add it to the BFS Queue. Also mark it visited
                    if word not in visited:
                        visited[word] = True
                        queue.append((word, level + 1))
                all_combo_dict[intermediate_word] = []
        return 0


# In[65]:


Solution().ladderLength("hot","dog",["hot","dog","dot"])  


# In[66]:


Solution().ladderLength("a","d",["a","v","d"])  


# In[67]:


Solution().ladderLength("a","d",["a","b","c","d"])  


# In[68]:


Solution().ladderLength("hot","dog",["hot","dog","dot"])  


# In[88]:


from string import ascii_lowercase
import collections
def replace_char(word,position,ch):
    new = list(word)
    new[position] = ch
    word=''.join(new)
    return word
        
class Solution(object):
    
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        wordlst=collections.defaultdict()
        
        for i in wordList:
             wordlst[i]=1
                
        if endWord not in wordlst:
            return 0 

        ## Breadth First Search on an implied word graph 
        level = 1
        word_len = len(beginWord) 
        qu=[]
        qu.append(beginWord)
        
        
        while qu:
            ## iterate over level 
            print(qu)
            for i in range(len(qu)):
                word=qu.pop(0)
                ## First Generate Children of the word (those with higher level of change compared to original word 
                for pos in range(word_len):
                    orig_char =  word[pos]
                    for char in ascii_lowercase:  
                        word_cand=replace_char(word,pos,char)
                        if  word_cand== endWord:
                            return level+1
                        
                        if  word_cand in wordlst.keys() and word_cand!=word:  ## it has to be a word of interest (in the given list)
#                                 print("adding", word_cand)
                                qu.append(word_cand)
                                del wordlst[word_cand]
#                                 print(wordlst)

                    word =replace_char(word,pos,orig_char) 
            level+=1
            
        return 0 
                            
#Solution().ladderLength("hot","dog",["hot","dog","dot"])                          
#Solution().ladderLength("a","d",["a","b","c","d"])  
Solution().ladderLength("hit","cog",["hot","dot","dog","lot","log","cog"])


# In[84]:



Solution().ladderLength("a","d",["a","v","d"]) 


# In[104]:


from collections import defaultdict
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """

        if endWord not in wordList or not endWord or not beginWord or not wordList:
            return 0

        # Since all words are of same length.
        L = len(beginWord)
        
        all_combo_dict = defaultdict(list)
        ## preprocess wordlist to put all words in the samel level together
        ## i.e. all words that are one edit away from each other (since they come from  
        ## same patterm). Some words could fit in different patterns 
        
        for word in wordList:
            for i in range(L):
                all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)
            
        #print(all_combo_dict)
        
        # Queue for BFS
        queue = [(beginWord, 1)]
        visited = {beginWord: True}
        while queue:
            current_word, level = queue.pop(0)      
            for i in range(L):
                # Intermediate words for current word
                intermediate_word = current_word[:i] + "*" + current_word[i+1:]

                # loop over all words at the same level
                for word in all_combo_dict[intermediate_word]:
                    # If at any point if we find what we are looking for
                    # i.e. the end word - we can return with the answer.
                    if word == endWord:
                        return level + 1
                    # Otherwise, add it to the BFS Queue. Also mark it visited
                    if word not in visited:
                        visited[word] = True
                        queue.append((word, level + 1))
                        
                del all_combo_dict[intermediate_word]  # = []
        return 0

Solution().ladderLength("hit","cog",["hot","dot","dog","lot","log","cog"])


# In[112]:


## word ladder II  https://leetcode.com/problems/word-ladder-ii/

class Solution(object):
    def findLadders(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        all_adj_combine = collections.defaultdict(list)
        
        wordLen = len(beginWord)
        
        for w in wordList:
            for i in range(wordLen):
                all_adj_combine[w[:i] + "*" + w[i + 1:]].append(w)
                
        level = {}
        for w in wordList:
            level[w] = float("inf")
            
        level[beginWord] = 0
            
        ###BFS
        ans_list = []
        
        ans = float("inf")
        queue = [(beginWord, [beginWord])]
        
        while queue:
            word, path_list = queue.pop(0)
            
            if len(path_list) >= ans:
               # print(path_list)
                 #  print(path_list)
               # print(ans_list) if you get a longer path then exit with the shorter ones
            ## already in the path_list
                return ans_list
           
            
            for i in range(wordLen):
                adj_words = all_adj_combine[word[:i] + "*" + word[i + 1:]]
                
                for adj_w in adj_words:
                    
                    if adj_w == endWord:
                        ans_list.append(path_list + [adj_w])
                        ans = len(path_list + [adj_w])
                        
                    elif level[adj_w] > level[word]:
                        queue.append((adj_w, path_list + [adj_w]))
                        level[adj_w] = level[word] + 1
                        
        return ans_list
    
    


# In[113]:


Solution().findLadders("hit","cog",["hot","dot","dog","lot","log","cog"])


# In[118]:


###  Friends Of Appropriate Ages
## https://leetcode.com/problems/friends-of-appropriate-ages/

class Solution(object):
    def numFriendRequests(self, ages):
        """
        :type ages: List[int]
        :rtype: int
        """
        ## max age is 120 
        count = [0] * 121
        for age in ages:
            count[age] += 1  
        #store frequence of age=index in count[index] array
       # print(count)
        ans = 0
        for ageA, countA in enumerate(count):
            for ageB, countB in enumerate(count):
                if ageA * 0.5 + 7 >= ageB: continue
                if ageA < ageB: continue
                if ageA < 100 < ageB: continue
                ans += countA * countB
                if ageA == ageB: ans -= countA

        return ans
    
Solution().numFriendRequests([16,16])


# In[123]:


## https://leetcode.com/problems/subarray-sum-equals-k/
## Subarray Sum Equals K

class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        
        SumHash={0:1}
        sumse=0
        result=0 
        
        for i in range(len(nums)):
            sumse+=nums[i]
            if sumse-k in SumHash:
                result+=SumHash[sumse-k]
                
            SumHash[sumse]=SumHash.get(sumse,0)+1
    
        return result
    
Solution().subarraySum([0,0,0,0,0,0,0,0,0,0],0)


# In[122]:


Solution().subarraySum([1,1,1],2)


# In[126]:


#### https://leetcode.com/problems/simplify-path/ 
class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        s=[] ## stack to store path component directories 
        p=path.split("/")
        ## Now iterate over split directories and check 
        ## what to do..sometimes you need to go back one level
        ## if you see ..
        for direc in p:
            if s and direc=="..":
                s.pop()
            elif(direc not in [".","",".."]):
                s.append(direc)
                
        return self.construct_path(s)
    
    def construct_path(self,st):
        return "/" +"/".join(st)

Solution().simplifyPath("/a/../../b/../c//.//")


# In[129]:


## string anagrams 
#        Example :    traps  2          sprat    strap
#                     opt               2          top       pot
#                     star              1          rats
# Input:
# s: "cbaebabacd" p: "abc"
# Output:
# [0, 6]

# Explanation:
# The substring with start index = 0 is "cba", which is an anagram of "abc".
# The substring with start index = 6 is "bac", which is an anagram of "abc".
from collections import Counter
class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
#         comp = Counter(s[:len(p)])
#         print(comp)
#         target = Counter(p)
#         ans = []
#         if comp==target:
#             ans.append(0)
        
#         for i in range(1, len(s)-len(p)+1):
#             # pop last word from Counter
#             comp[s[i-1]] -= 1
#             if comp[s[i-1]]==0:
#                 comp.pop(s[i-1])
#             # add next word to Counter as like sliding window
#             if s[i+len(p)-1] not in comp:
#                 comp[s[i+len(p)-1]] = 1
#             else:
#                 comp[s[i+len(p)-1]] += 1
#             # check windows(Counter)
#             if comp==target:
#                 ans.append(i)
#         return ans
## Match two dictionaries of character frequency and slide the window along 

        myDictP=collections.Counter(p)
        ## first dic myDictP is fixed
        myDictS=collections.Counter(s[:len(p)])
        ## this dict slides along the array
        
        output=[]
        
        ## initialize where we are in array s[:] (start)
        i=0
        j=len(p)   #( j is index right after end of potential pattern)
        
        while j<=len(s):
            ## if there is a match store index i 
            if myDictS==myDictP:
                print("match")
                output.append(i)
            ## as we slide the window past index i we decrement
            ## the counter of character at s[i]
            myDictS[s[i]]-=1
    
            ## if after decrementing counter of s[i] it becomes 0 we remove it from the dic
            print(myDictS , i)
            if myDictS[s[i]]<=0:
                myDictS.pop(s[i])
                print(myDictS , i)
                #del myDictS[s[i]] this is less efficient than pop
            # we increase the counter of s[j] because the end of the window is now at j  
            if j<len(s):    
                 myDictS[s[j]]+=1
            ## slide i, j forward 
            j+=1
            i+=1
            
        return output    
Solution().findAnagrams("cbaebabacd","abc")


# In[146]:


## 211. Add and Search Word - Data structure design
# Design a data structure that supports the following two operations:

# void addWord(word)
# bool search(word)

# search(word) can search a literal word or a regular expression string containing only letters a-z or .. A . means it can represent any one letter.

# Example:

# addWord("bad")
# addWord("dad")
# addWord("mad")
# search("pad") -> false
# search("bad") -> true
# search(".ad") -> true
# search("b..") -> true
## https://leetcode.com/problems/add-and-search-word-data-structure-design/
##d={"a":1,"b":2}  d.values() iterator dict_values([1, 2]) 
class TrieNode:
    def __init__(self):
        self.is_end = False
        self.children = {}

class WordDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
        

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: None
        """
        node = self.root
        
        for c in word:
            if not c in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        
        node.is_end = True

        

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        def contains(node, word):
            for i in range(len(word)):
                c = word[i]
                ## if node has children we are good but need to check rest 
                ## of the word thru recursive call 
                if c == '.' and node.children:
                    for child in node.children.values():
                        if contains(child, word[i+1:]):
                            return True
                    return False
                else: ## for any non "." character we are good with simple comparison
                    if not c in node.children:
                        return False
                    else:
                        node = node.children[c]

            return node.is_end

        return contains(self.root, word)
        


# Your WordDictionary object will be instantiated and called as such:
obj = WordDictionary()
for word in ["WordDictionary","addWord","addWord","addWord","search","search","search","search"]:
    obj.addWord(word)
res=[]
for word in [[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."],["addWord"]]:
    if word==[]:
        res.append(obj.search(""))
    else: 
        res.append(obj.search(word[0]))
res


# In[147]:


## https://leetcode.com/problems/check-completeness-of-a-binary-tree/
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isCompleteTree(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        ## DO level order traversal if you see null and one element after the null return False
        ## because it means we have leaf not as far to the left as possible like here
        ##  [1,2,3,4,5,null,7]
        sawnull=False
        Qu=[]
        Qu.append(root)
        while Qu:
            current=Qu.pop(0)
            if current==None:
                sawnull=True
            else:
                if sawnull:
                    return False
                else:
                    Qu.append(current.left)
                    Qu.append(current.right)
                    
        return True 


# In[148]:


## Second solution (node, number of nodes sofar ) if at the end (x,n) where n==number of nodes then its complete      
#         nodes = [(root, 1)]
#         i = 0
#         while i < len(nodes):
#             node, v = nodes[i]
#             i += 1
#             if node:
#                 nodes.append((node.left, 2*v))
#                 nodes.append((node.right, 2*v+1))

#         return  nodes[-1][1] == len(nodes)


# In[ ]:







