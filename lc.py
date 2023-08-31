class Solution:
    def canPartition(self, nums):
        total = sum(nums)
        if total  % 2 != 0 or len(nums) < 2: return False
        amount = total // 2
        dp = [False] * (amount+1)
        dp[0] = True
        for n in nums:
            for i in range(amount, n-1, -1):
                dp[i] = dp[i] or dp[i-n]
                # print('n: {}, i: {}, dp: {}'.format(n, i, dp))
        return dp[-1]



class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False]*n for _ in range(n)]
        max_len = 0
        res = ''
        for i in range(n):
            for l in range(1, n+1):
            
        # for l in range(1, n+1):
        #     for i in range(n):
                j = i+l-1
                if j >= n: break
                if l == 1: 
                    dp[i][j] = True
                elif l == 2:
                    dp[i][j] = s[i] == s[j]
                elif l > 2:
                    dp[i][j] = dp[i+1][j-1] and s[i] == s[j]
                if  dp[i][j] and max_len < l:
                    max_len = l
                    res = s[i:j+1]
        return res
    
class Solution:
    def threeSum(self, nums):
        n = len(nums)
        nums.sort()
        if not nums or n < 3:
            return []
        res = []
        for i in range(n):
            if nums[i] > 0:
                return res
            if i > 0 and nums[i] == nums[i-1]:
                continue
            
            left = i + 1
            right = n - 1
            
            while left < right:
                if (nums[i] + nums[left] + nums[right]) == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif (nums[i] + nums[left] + nums[right]) > 0:
                    right -= 1
                else:
                    left += 1
        return res
    
if __name__ == "__main__":
    # data = [5,1,11,5]
    # a = Solution()
    # a.canPartition(data)

    data = "babad"
    a = Solution()
    a.longestPalindrome(data)