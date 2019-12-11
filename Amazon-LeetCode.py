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
        """
        re=0
        for i in nums:
            re^=i 
        return re
Solution().singleNumber([2,2,1]) ## Expected 1 

