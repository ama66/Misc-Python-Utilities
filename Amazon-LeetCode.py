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



            
            
        
        
