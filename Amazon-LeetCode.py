## https://leetcode.com/problems/first-unique-character-in-a-string/submissions/
class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s=="":
            return -1 
        
        HashMap={}
        for i in range(len(s)):
            if s[i] in HashMap:
                HashMap[s[i]]=float("inf")
            else:
                HashMap[s[i]]=i
        out=min(HashMap.items(), key=lambda x: x[1])[1]
        return  out if out!=float("inf") else -1 
    
Solution().firstUniqChar("leetcode")   # Expected 0   

#https://leetcode.com/problems/single-number/
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        solution using ^ operator 
        """ 
        re=0
        for i in nums:
            re^=i 
        return re
Solution().singleNumber([2,2,1]) ## Expected 1 

class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        HashMap={}
        for i in range(len(nums)):
            if nums[i] in HashMap:
                del HashMap[nums[i]]
            else:
                HashMap[nums[i]]=True

        return [i for i in HashMap.keys()][0]

Solution().singleNumber([2,2,1]) ## Expected 1  on leetcode HashMap.keys()[0] works fine! 


### 
class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        ## using BackTracking and a helper function 
        configs=[]
        board=[["."]*n]
        con=self.SolveNq_Helper(n,board,row=0)
        print("Returning", con)
        configs.append( con)
        return configs 
        
    def SolveNq_Helper(self, n,board=[[]],row=0):
        ## Baseline I made all possible choices
        if  row==n+1:
            return board
        if row > 0 :
            print("Calling with row =", row,board[row-1])
        for i in range(n):
            ## Choose
            board[row][i]="Q"
            ## Explore Deeper
            if self.is_valid_move(board , row , i ): 
                board.append(["."]*n)
                self.SolveNq_Helper(n,board,row+1)
            else:
            ## unchoose if it does not work 
                board[row][i]="."
                
    def is_valid_move(self, board,row,col):
        ncol=len(board[0])
        for i in range(row):
            for c in range(ncol):
                if board[i][c]=="Q":
                    diffcol=abs(col-c)
                    diffdiag=abs(row-i)
                    if diffcol==0 or diffcol==diffdiag:
                        return False
        return True 
  ## https://leetcode.com/problems/power-of-two/submissions/
class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        i=1
        while i < n:
            i*=2
            
        return i==n
Solution().isPowerOfTwo(16) ## Expected True 


## https://leetcode.com/problems/robot-return-to-origin/submissions/

class Solution(object):
    def judgeCircle(self, moves):
        """
        :type moves: str
        :rtype: bool
        """
        HashMap={"UD":0 , "LR":0}
        
        for i in moves:
            if i=="U":
                HashMap["UD"]+=1
            if i=="D":
                HashMap["UD"]-=1
            if i=="L":
                HashMap["LR"]+=1
            if i=="R":
                HashMap["LR"]-=1

        return HashMap["UD"]==0 and HashMap["LR"]==0
            
Solution().judgeCircle("LL") # Expected False

## Missing Number 
## https://leetcode.com/problems/missing-number/
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
#         sume=sum(nums)
#         n=len(nums)+1
#         return int(n*(n-1)/2) - sume
        HashMap={}
        for i in nums:
            if i not in HashMap:
                HashMap[i]=True 
        for i in range(len(nums)+1):
            if i not in HashMap:
                return i 
Solution().missingNumber([0,1,3]) ## Expected 2

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# https://leetcode.com/problems/path-sum/submissions/
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if root is None:
            return False
        elif root.left==None and root.right==None and sum-root.val==0:
            return True
        else:
            return self.hasPathSum(root.left,sum-root.val) or self.hasPathSum(root.right,sum-root.val)
 ### Test Expected True 
# [5,4,8,11,null,13,4,7,2,null,null,null,1]
# 22
## https://leetcode.com/problems/remove-element/submissions/
from  collections import Counter
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
            count_val = len(nums) - Counter(nums)[val] 
            ## overwrite nums
            nums[:] = [x for x in nums if x != val]
            return count_val
        
        
## https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/submissions/
  # Recursive
 # if p.val <= root.val <= q.val or q.val <= root.val <= p.val:
        #     return root
        # elif p.val < root.val and q.val < root.val:
        #     return self.lowestCommonAncestor(root.left, p,q)
        # else:
        #     return self.lowestCommonAncestor(root.right,p,q)
  ## Iterative:
        while root:
            if p.val <= root.val <= q.val or q.val <= root.val <= p.val:
                return root
            elif p.val < root.val and q.val < root.val:
                root = root.left
            else:
                root = root.right
     
   # https://leetcode.com/problems/jewels-and-stones/submissions/
class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        count = 0
        jewels = {}
        for j in J:
            jewels[j] = True
        for s in S:
            count += 1 if jewels.get(s) else 0
        return count
    
#https://leetcode.com/problems/flood-fill/

class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        
#  input       
#         [[1,1,1],
#          [1,1,0],
#          [1,0,1]]
        
#         sr = 1, sc = 1, newColor = 2
# Output: [[2,2,2],
#          [2,2,0],
#          [2,0,1]]
# using DFS method
        if not image:
            return []
        
        m = len(image)
        n = len(image[0])
        value = image[sr][sc]

        stack = []
        seen = set()
        stack.append((sr, sc))
        seen.add((sr, sc))
        
        while stack:
            row, col = stack.pop()
            image[row][col] = newColor
            for i, j in [(row - 1, col),(row + 1, col), (row, col - 1),(row, col + 1)]:
                if 0 <= i < m and 0<= j < n and image[i][j] == value:
                    if (i, j) in seen:
                        continue
                    stack.append((i, j))
                    seen.add((i, j))
        return image
## BFS 
 row, col, prevColor = len(image), len(image[0]), image[sr][sc]
        if prevColor == newColor: return image
        
        q = [(sr, sc)]
        while q:
            cr, cc = q.pop(0)
            if cr < row and cr >= 0 and cc < col and cc >= 0 and image[cr][cc] == prevColor:
                image[cr][cc] = newColor
                q.append((cr+1,cc))
                q.append((cr-1,cc))
                q.append((cr,cc+1))
                q.append((cr,cc-1))

        return image

# https://leetcode.com/problems/most-common-word/submissions/ 

class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        
        bannedSet = set(banned)
        
#         for i in "!?',;.":
#             paragraph = paragraph.replace(i,' ')
        
#         words = paragraph.lower().split()
## can also use regex
        words= re.findall(r'\w+', paragraph.lower())
        wordfreq = {}
        maxCount = 0
        result = ""
        
        for word in words:
            if word not in bannedSet:
                wordfreq[word] = wordfreq.get(word, 0) + 1
                
                if wordfreq[word] > maxCount:
                    result = word
                    maxCount = wordfreq[word]
                    
        
        return result
    
    
### https://leetcode.com/problems/k-closest-points-to-origin/
## PQ 
from heapq import heappush , heappop  

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        ## Throw all points into max heap
        h=[]
        for i in points:
            ## push a tuple (distance,coords(x,y))
            heappush(h, (i[0]**2+i[1]**2, i))
            
        L=[]
        for i in range(K):
            L.append(heappop(h)[1])
        return L
    
# https://leetcode.com/problems/group-anagrams/submissions/
## Group Anagrams
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = collections.defaultdict(list)
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord('a')] += 1
            ans[tuple(count)].append(s)
       
        return ans.values()
    
  #########Algop******************

import collections

def hashkey(str):
  return "".join(sorted(str))

def hashkey2(str):
  arr = [0] * 26
  for c in str:
    arr[ord(c) - ord('a')] = 1
  return tuple(arr)

def groupAnagramWords(strs):
  groups = collections.defaultdict(list)
  for s in strs:
    groups[hashkey2(s)].append(s)

  return tuple(groups.values())

print(groupAnagramWords(['abc', 'bcd', 'cba', 'cbd', 'efg']))
# (['abc', 'cba'], ['bcd', 'cbd'], ['efg'])

 #########Algop******************
  ##  https://leetcode.com/problems/unique-paths/
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        if m <= 0 or n <= 0:
            return
        dp = [[0]*n]*m
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]

#

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        ## Recursive solution
        if root == None:
            return []
        self.paths = []
        self.dfs(root,[root.val],root.val,sum)
        return self.paths
    
    def dfs(self,node,partial_path,path_sum,sum):
        if node == None:
            return
        else:
            if node.left == None and node.right == None and path_sum  == sum:
                    self.paths.append(partial_path)
            else:
                if node.left:
                    self.dfs(node.left,partial_path+[node.left.val],path_sum+node.left.val,sum)
                if node.right:
                    self.dfs(node.right,partial_path+[node.right.val],path_sum + node.right.val,sum)
    
    
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        ## Iterative solution using Stack
                    
        if not root:
            return []
        ## stack contain tuples of (node,cur_sum=sum-node.val, [partial path])
        
        stack = [(root,sum-root.val,[root.val])]
        result = []
        
        while stack:
            
            node, curr_sum,partial_path = stack.pop()
            
            if not node.left and not node.right and curr_sum == 0:
                result.append(partial_path)
                
            for neighbor in (node.right,node.left):
                if neighbor:
                    stack.append((neighbor,curr_sum-neighbor.val,partial_list+[neighbor.val]))
					
        return result
   
	# https://leetcode.com/problems/symmetric-tree/submissions/
    
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:

        def helper(p, q):
            if p and q: 
                return p.val == q.val and helper(p.left, q.right) and helper(p.right, q.left)
            else: 
                return p is q
            
        return helper(root, root)

  
  #  https://leetcode.com/problems/sort-array-by-parity/submissions/
class Solution:
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        if A is None:
            return []
        
        even_index = 0 
        
        for i,num in enumerate(A): 
            if num % 2 == 0:
                A[i],A[even_index]=A[even_index],A[i]
                even_index += 1
            
        return A

## https://leetcode.com/problems/combination-sum/
import copy
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def _combsum(candidates,target,curchosen,cursum,curindex):
            ##BaseCase
            
            if cursum==target:
                return[curchosen] 
            
            if cursum>target: 
                return[]
            
            #iteration
            output=[]
            for i in range(curindex,len(candidates)):
                curchosen_temp=copy.deepcopy(curchosen)
                ## pick item i 
                curchosen_temp.append(candidates[i])
                ## see if all current chosen item add up to target
                combinations=_combsum(candidates,target,curchosen_temp,cursum+candidates[i],i)
                output+=combinations 
            return output 
        
        return  _combsum(candidates,target,[],0,0)
       
	## Another solution
	        res = []
        
        def backtrack(nums=[], index=0):
            s = sum(nums)
            if s > target:
                return
            if s == target:
                res.append(nums)
                return
            
            for i in range(index, len(candidates)):
                backtrack(nums + [candidates[i]], i)
        
        backtrack()
        return res


##############################

class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        answer = []
        sublist = []
        candidates.sort() # makes skipipng duplicate subsets easier since we can ignore adjacent numbers that are the same
        
        # process subsets for each number in array (initially 1 original recursive call for each num),
        # then loop from that current num to end of array seeking to satisfy conditions for target
        # O(N) recursive calls for each numbers recursive stack)
        # e.g. [1, 1, 2, , 5, 6, 7, 10] as for loop runs we will have O(n), O(n-1), O(n-2)...O(n - n)
        def findCombinations(answer, candidates, target, sublist, index):
            if target < 0:
                return
            if target == 0:
                answer.append(sublist)
                return
            
            # process possible subsets for each number in candidates, starting at index of current num we are processing
            for i in range(index, len(candidates)):
                
                # only do work if either first iteration, or dealing with a different number than the last one
                if i == index or candidates[i] != candidates[i-1]:
                    
                    # simulate choosing current number for possible subset
                    sublist.append(candidates[i])
                    
                    # this continues adding the next numbers until one of the target conditions above are met
                    findCombinations(answer, candidates, target - candidates[i], list(sublist), i + 1)
                    
                    # simulate not choosing current number for part of our subset
                    # remove current number we are at, and move on to next number in the loop
                    sublist.remove(candidates[i])
                
        findCombinations(answer, candidates, target, sublist, 0)
        return answer
        
# https://leetcode.com/problems/partition-labels/	
class Solution:
    def partitionLabels(self, S: str) -> List[int]:
        lastIndex = {c:i for i, c in enumerate(S)}
        print(lastIndex)
        partition_lengths= []
        ## i Marks the beginning index of the first substring then second substring then third...etc
        i=0 
        while i < len(S):
            end = lastIndex[S[i]]
            if end==i:
                partition_lengths.append(1)
                i+=1
            else:
                j=i+1
                while j < end:
                     end =max(end,lastIndex[S[j]])
                     j+=1

                partition_lengths.append(j-i+1)
                ## Now move i to the beginning of the new substring
                i=j+1

        return partition_lengths
    
### https://leetcode.com/problems/longest-common-prefix/submissions/
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if strs == []:
            return ""
        
        common = ""
        ## first find smallest le
        smallest = min(strs, key=lambda x: len(x))
        ## Iterate over the characters of the smallest length string
        for i in range(len(smallest)):
            ## iterate over all strings. Note that a list of strings is equivalent to a 2D array
            for j in range(1,len(strs)):
                if strs[0][i] != strs[j][i]:
                    return common
                
            common += strs[0][i]
        return common


https://leetcode.com/problems/subsets/

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []
        ## iterate over different length subsets
        ## starting with [] the null subset
        for sub_len in range(len(nums)+1):
            self.generate_subset_len(result, nums, [], sub_len)
        
        return result
    
    def generate_subset_len(self, result, nums, Accumulator, sub_len):
        if len(Accumulator) == sub_len:
            result.append(Accumulator.copy())
            return
        
        for i, num in enumerate(nums):
            ##Make a Choice
            Accumulator.append(num)
            del nums[i]
            ## Explore Deeper
            self.generate_subset_len(result, nums[i:], Accumulator, sub_len)
            ##Unchoose
            nums.insert(i, num)
            Accumulator.pop()
        return


## https://leetcode.com/problems/merge-two-sorted-lists/

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        list_merged = current = ListNode(0)
		
        while l1 or l2:
            if l1 and l2:
                if l1.val <= l2.val:
                    current.next = ListNode(l1.val)
                    current = current.next
                    l1 = l1.next
                else:
                    current.next = ListNode(l2.val)
                    current = current.next
                    l2 = l2.next
            else:
                if not l1:
                    current.next = ListNode(l2.val)
                    current = current.next
                    l2 = l2.next
                else:
                    current.next = ListNode(l1.val)
                    current = current.next
                    l1 = l1.next
            
        return list_merged.next


## https://leetcode.com/problems/coin-change/

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float("inf") for i in range(amount+1)]
        dp[0] = 0
        
        for coin in coins:
            for current_amount in range(1, amount+1):
                if coin <= current_amount:
                    dp[current_amount] = min(dp[current_amount], 1+dp[current_amount- coin])
                    
        return dp[amount] if dp[amount] != float("inf") else -1

# https://leetcode.com/problems/binary-tree-right-side-view/

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if root==None:
            return []
        
        qu=[]
        ## keep track of level information when you append to the queue 
        l=0
        qu.append((root,l))
        level_nodes={}
        
        while qu:
            node,level=qu.pop(0)
           
            if level in level_nodes:
                level_nodes[level].append(node.val)
            else:
                level_nodes[level]=[node.val]
                
            if node.left:
                qu.append((node.left,level+1))
            if node.right:
                qu.append((node.right,level+1))
       
        l=sorted([i for i in level_nodes.keys()])
        res=[]
        for key in l:
            res.append(level_nodes[key][-1])
            
        return res
    
  ## Remove vowels from a string  
## 
def rem_vowel(string): 
    vowels = ('a', 'e', 'i', 'o', 'u')  
    for x in string.lower(): 
        if x in vowels: 
            string = string.replace(x, "") 
              
    # Print string without vowels 
    print(string) 
  
# Driver program 
string = "GeeksforGeeks - A Computer Science Portal for Geeks"
rem_vowel(string) 
###############

## https://leetcode.com/problems/minimum-path-sum/
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
	m=len(grid)
        n=len(grid[0])
        dp=[[0 for i in range(n)]for j in range(m)]
        dp[0][0] = grid[0][0]
        ## initialize first row (could only come from the left i-1)
        for i in range(1, n):
            dp[0][i] = grid[0][i]+dp[0][i-1]
        ## initialize first colum. can only come from above i-1 
        for i in range(1, m):
            dp[i][0] = grid[i][0]+dp[i-1][0]
        ## for any other point need to pick the min of two options
        ## coming from above or from left 
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
                
        return dp[m-1][n-1]


##  https://leetcode.com/problems/binary-tree-paths/

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        full_paths=[]
        if root is None:
            return full_paths
        ###########################
        def dfs(root,path_accumulator,full_paths):
            path_accumulator+=str(root.val)
            ##BaseCase
            if root.left is None and root.right is None:
                full_paths.append(path_accumulator)
                return
            if root.left:
                dfs(root.left,path_accumulator+"->",full_paths)
            if root.right:
                dfs(root.right,path_accumulator+"->",full_paths)

        ########################
        path_accumulator=""
        dfs(root,path_accumulator,full_paths)
        
        return full_paths
    
       
    
    
