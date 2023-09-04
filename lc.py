

    
# 冒泡排序
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n-1-i):
            if arr[j+1] < arr[j]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)
    print('left_half: {}, right_half: {}'.format(left_half, right_half))
    return merge(left_half, right_half)

def merge(left, right):
    result = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    middle = [i for i in arr if i == arr[mid]]
    left = [i for i in arr if i < arr[mid]]
    right = [i for i in arr if i > arr[mid]]
    return quick_sort(left) + middle + quick_sort(right)

import heapq
def findKthLargest(nums, k):
    heap = []
    
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num)
            print(heap)
        elif num > heap[0]:
            print('pre: k: {}, num: {}, heap: {}'.format(k, num, heap))
            heapq.heappop(heap)
            heapq.heappush(heap, num)
            print('aft: k: {}, num: {}, heap: {}'.format(k, num, heap))

    return heap[0]

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
    data = "babad"
    a = Solution()
    a.longestPalindrome(data)
    nums2 = [3, 2, 3, 1, 2, 4, 5, 5, 6]
    k2 = 4
    result2 = findKthLargest(nums2, k2)
    print(result2)  # 输出：4
    print(list('lishaowei'))