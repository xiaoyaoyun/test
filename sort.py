
    
# 冒泡排序
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n-1-i):
            if arr[j+1] < arr[j]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


arr = [64, 34, 25, 12, 22, 11, 90]
# arr = [64, 34, 25]
bubble_sort(arr)
print("排序后的数组:", arr)

# 归并排序
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

arr = [12, 11, 13, 5, 6, 7]
sorted_arr = merge_sort(arr)
print("排序后的数组:", sorted_arr)


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

arr = [12, 11, 13, 5, 6, 7]
sorted_arr = merge_sort(arr)
print("排序后的数组:", sorted_arr)

# 快速排序
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    middle = [i for i in arr if i == arr[mid]]
    left = [i for i in arr if i < arr[mid]]
    right = [i for i in arr if i > arr[mid]]
    return quick_sort(left) + middle + quick_sort(right)

arr = [12, 11, 13, 5, 6, 7]
sorted_arr = quick_sort(arr)
print("排序后的数组:", sorted_arr)

# 堆排序
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

# nums1 = [3, 2, 1, 5, 6, 4]
# k1 = 2
# result1 = findKthLargest(nums1, k1)
# print(result1)  # 输出：5

nums2 = [3, 2, 3, 1, 2, 4, 5, 5, 6]
k2 = 4
result2 = findKthLargest(nums2, k2)
print(result2)  # 输出：4
print(list('lishaowei'))