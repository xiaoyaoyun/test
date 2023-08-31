# Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] 
# such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

# Notice that the solution set must not contain duplicate triplets.


# Example 1:

# Input: nums = [-1,0,1,2,-1,-4]
# Output: [[-1,-1,2],[-1,0,1]]
# Explanation: 
# nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
# nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
# nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
# The distinct triplets are [-1,0,1] and [-1,-1,2].
# Notice that the order of the output and the order of the triplets does not matter.
# Example 2:

# Input: nums = [0,1,1]
# Output: []
# Explanation: The only possible triplet does not sum up to 0.
# Example 3:

# Input: nums = [0,0,0]
# Output: [[0,0,0]]
# Explanation: The only possible triplet sums up to 0.
def find_sumequal_zero(arr_data):
    if not arr_data or len(arr_data) < 3: return []
    arr_data.sort()
    n = len(arr_data)
    res = []
    for i in range(n-1):
        if i > 0 and arr_data[i] == arr_data[i-1]:
            continue
        left = i + 1
        right = n-1
        while left < right:
            if arr_data[i] + arr_data[left] + arr_data[right] == 0: 
                res.append([arr_data[i], arr_data[left], arr_data[right]])
                while left < right and arr_data[left] == arr_data[left + 1]:
                    left += 1
                while left < right and arr_data[right] == arr_data[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif arr_data[i] + arr_data[left] + arr_data[right] > 0:
                right -= 1
            else:
                left += 1
    return res

# You are climbing a staircase. It takes n steps to reach the top.

# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
# Example 1:
# Input: n = 2
# Output: 2
# Explanation: There are two ways to climb to the top.
# 1. 1 step + 1 step
# 2. 2 steps
# Example 2:

# Input: n = 3
# Output: 3
# Explanation: There are three ways to climb to the top.
# 1. 1 step + 1 step + 1 step
# 2. 1 step + 2 steps
# 3. 2 steps + 1 step
def calc_steps(n):
    results = [0 for i in range(n)]
    results[0] = 1
    results[1] = 2
    for i in range(2, n):
        results[i] = results[i-1] + results[i-2]
    return results[-1]



if __name__ == "__main__":
    test = [0,1,1]
    result = find_sumequal_zero(test)
    print(result)