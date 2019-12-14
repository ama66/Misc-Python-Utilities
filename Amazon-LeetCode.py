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
    
    



