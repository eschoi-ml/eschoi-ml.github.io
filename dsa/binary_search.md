[<-PREV](dsa.md)

# Ultimate Binary Search Template
This is an ultimate binary search template inspired by and improved from [this.](https://towardsdatascience.com/powerful-ultimate-binary-search-template-and-many-leetcode-problems-1f850ef95651)

1. Initialize l inclusively and r exclusively: [l, r)
2. Decide whether your searching value is **minimal** or **maximal** and design the condition accordingly.
3. Return l or l-1
- return l when searching **minimal** value that satisfies the condition
- return l-1 when searching **maximal** value that does NOT satisfy the condition


```python
l, r = 0, len(array)        # l inclusive, r exclusive: the answer lies in [l, r)
while l < r:
    mid = (l + r)//2
    if condition(mid):      # condition function such that l is the minimal k that satisfies the condition
        r = mid             # condition function such that l-1 is the maximal k that doesn't satisfy the condition
    else:
        l = mid + 1
return l or l-1             # return l or l-1
```

## Basic application 
### [278. First Bad Version](https://leetcode.com/problems/first-bad-version/)



```python
def firstBadVersion(self, n):
    l, r = 1, n+1
    while l < r:
        mid = (l + r)//2
        if isBadVersion(mid):
            r = mid
        else:
            l = mid + 1
    return l
```

### [35. Search Insert Position](https://leetcode.com/problems/search-insert-position/)


```python
def searchInsert(self, nums: List[int], target: int) -> int:
    
    l, r = 0, len(nums)
    while l < r:
        mid = (l + r)//2
        if nums[mid] >= target:
            r = mid
        else:
            l = mid + 1
    return l
```

### [69. Sqrt(x)](https://leetcode.com/problems/sqrtx/)


```python
def mySqrt(self, x: int) -> int:
    """
    mid^2 <= x < (mid+1)^2

    """
    if x == 0: return 0
    
    l, r = 1, x+1
    while l < r:
        mid = (l + r)//2
        if x < (mid+1)**2:
            r = mid
        else:
            l = mid + 1

    return l

    l, r = 1, x+1
    while l < r:
        mid = (l + r)//2
        if mid * mid > x:
            r = mid
        else:
            l = mid + 1

    return l-1
```

### [981. Time Based Key-Value Store](https://leetcode.com/problems/time-based-key-value-store/)


```python
def get(self, key: str, timestamp: int) -> str:
    
    if key not in self.h: return ""
    
    l, r = 0, len(self.h[key])
    while l < r:
        mid = (l + r)//2
        if self.h[key][mid][0] > timestamp:
            r = mid
        else:
            l = mid + 1

    if l-1 >= 0:
        return self.h[key][l-1][1]
    return ""
```

### [704. Binary Search](https://leetcode.com/problems/binary-search/)


```python
def search(self, nums: List[int], target: int) -> int:
    
    if not nums or nums[0] > target or nums[-1] < target: return -1

    l, r = 0, len(nums)
    while l < r:
        mid = (l + r)//2
        if nums[mid]>=target:
            r = mid
        else:
            l = mid + 1
    return l if nums[l] == target else -1
```

### [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)


```python
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    
    m, n = len(matrix), len(matrix[0])
    
    # row
    l, r = 0, m
    while l < r:
        mid = (l + r)//2
        if matrix[mid][0] > target:
            r = mid
        else:
            l = mid + 1
    row = l-1
    
    # col
    l, r = 0, n
    while l < r:
        mid = (l+r)//2
        if matrix[row][mid] > target:
            r = mid
        else:
            l = mid + 1
    col = l-1
    return matrix[row][col]==target
```

## Advanced application
- Type 1: Create a condition using a **feasible** function
- Type 2: Create a **enough** function for counting
- Type 3: Create a **cumsum** array to make it sorted

### [1011. Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)


```python
def shipWithinDays(self, weights: List[int], D: int) -> int:
    
    def feasible(capacity):
        
        cnt = 1
        total = 0
        for weight in weights:
            total += weight
            if total > capacity:
                total = weight
                cnt += 1
                if cnt > D:
                    return False
        return True
        

    l, r = max(weights), sum(weights)
    while l < r:
        mid = (l + r)//2
        if feasible(mid):
            r = mid
        else:
            l = mid + 1
    return l
```

### [410. Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)


```python
def splitArray(self, nums: List[int], m: int) -> int:
    
    def feasible(s):
        
        cnt = 1
        total = 0
        for num in nums:
            total += num
            if total > s:
                total = num
                cnt += 1
                if cnt > m:
                    return False
        return True
        
    
    l, r = max(nums), sum(nums)+1
    while l < r:
        mid = (l + r)//2
        if feasible(mid):
            r = mid
        else:
            l = mid + 1
    return l
```

### [875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)


```python
def minEatingSpeed(self, piles: List[int], h: int) -> int:
        
    def feasible(speed):
        
        hr = 0
        for pile in piles:
            hr += math.ceil(pile/speed)
        return hr <= h
            
    if len(piles) > h:
        return max(piles[:h])
    
    if len(piles) == h:
        return max(piles)
    
    # len(piles) < h
    
    l, r = 1, max(piles)
    while l < r:
        mid = (l + r)//2
        if feasible(mid):
            r = mid
        else:
            l = mid + 1
    return l
```

### [668. Kth Smallest Number in Multiplication Table](https://leetcode.com/problems/kth-smallest-number-in-multiplication-table/)


```python
def findKthNumber(self, m: int, n: int, k: int) -> int:
    """
    row\col  1   2   3   4   5
        1   1*1 1*2 1*3 1*4 1*5
        2   2*1 2*2 2*3 2*4 2*5
        3   3*1 3*2 3*3 3*4 3*5
    
    """

    def enough(mid):
        
        cnt = 0
        row = 1
        while row < m+1:
            curr_cnt = min(mid//row, n) # min(5, 3) min(2, 3) min(1, 3)
            if curr_cnt == 0:
                break
            cnt += curr_cnt    
            row += 1
        
        return cnt >= k
    
    l, r = 1, m * n
    while l < r:
        mid = (l + r)//2
        if enough(mid):
            r = mid
        else:
            l = mid + 1
    return l
```

### [719. Find K-th Smallest Pair Distance](https://leetcode.com/problems/find-k-th-smallest-pair-distance/)


```python
def smallestDistancePair(self, nums: List[int], k: int) -> int:
    """
    0, 1, 2, 3, 4,  5
    1, 1, 2, 4, 7, 10 m = 3
    i=0, j=4 cnt+= j-i-1(3)
    i=1, j=4 cnt+= 2
    i=2, j=4 cnt+= 1
    i=3, j=5 cnt+= 1
    i=4, j=6 cnt+= 1
    i=5, j=6 cnt+= 0
    i=
    
    """
    
    def enough(m):
        
        cnt = 0
        i, j = 0, 0
        while i < n:
            while j < n and nums[j] - nums[i] <= m:
                j += 1
            cnt += j-i-1
            i += 1           
        return cnt >= k

    nums.sort() # nlogn
    n = len(nums)
    
    l, r = 0, nums[-1] - nums[0]
    while l < r:
        mid = (l + r)//2
        if enough(mid):
            r = mid
        else:
            l = mid + 1
    return l
```

### [378. Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)



```python
def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
    """
    k = 8
    
    1  5  9
    10 11 13
    12 13 15
    

    """
    def count(mid):
        
        cnt = 0
        
        row, col = n - 1, 0
        smaller, larger = matrix[0][0], matrix[n-1][n-1]
        
        while row >= 0 and col < n:
            if matrix[row][col] > mid:
                larger = min(matrix[row][col], larger)
                row -= 1
            else:
                smaller = max(matrix[row][col], smaller)
                cnt += row + 1
                col += 1
                
        return cnt, smaller, larger
    
    def enough(m):
        
        cnt = 0
        i, j = n-1, 0
        while i >= 0 and j < n:
            if matrix[i][j] > m:
                i -= 1
            else:
                j += 1
                cnt += i+1
            
        return cnt >= k
    
    n = len(matrix)
    l, r = matrix[0][0], matrix[n-1][n-1]
    while l < r:
        
        mid = (l + r)//2
        
        if enough(mid):
            r = mid
        else:
            l = mid + 1
    
    return l
```

### [528. Random Pick with Weight](https://leetcode.com/problems/random-pick-with-weight/)


```python
class Solution:
    """
    array = [1, 3, 6]
    dist = [1/10, 4/10, 10/10] = [0.1, 0.4, 1.0]
    choose 0<= target< 1 random.random()
    if target<0.1: return 0
    elif target<0.4: return 1
    elif target<1: return 2
    """

    def __init__(self, w: List[int]):
        
        self.dist = [w[0]]
        for i in range(1, len(w)):
            self.dist[i] = self.dist[i-1] + w[i]
        
    def pickIndex(self) -> int:
        
        target = self.dist[-1] * random.random()
        l, r = 0, len(self.dist)
        while l < r:
            mid = (l + r)//2
            if target < self.dist[mid]:
                r = mid
            else: # self.dist[mid] >= target
                l = mid + 1
        return l
```

### [900. RLE Iterator](https://leetcode.com/problems/rle-iterator/)


```python
class RLEIterator:
    """
    [3, 8, 0, 9, 2, 5]
    seq = [8, 8, 8, 5, 5]
    next(2) = 8
    next(1) = 8
    next(1) = 5
    next(2) = -1
    
    self.count = [3, 3, 5]
    self.nums = [8, 9, 5]
    self.pointer = 0
    
    next(1)
    self.pointer = self.pointer(0) + 1 = 1
    
    binary search where self.pointer is located in self.count array
    l, r = 0, len(self.count)

    """

    def __init__(self, A: List[int]):
        
        n = len(A)//2
        self.count = [A[0]]
        self.nums = [A[1]]
        for i in range(1, n):
            self.count.append(self.count[i-1] + A[2*i])
            self.nums.append(A[2*i+1])  
        self.pointer = 0
        
    def next(self, n: int) -> int:
        
        self.pointer += n
        if self.pointer > self.count[-1]:
            return -1
        
        l, r = 0, len(self.count)
        while l < r:
            mid = (l + r)//2
            if self.pointer <= self.count[mid]:
                r = mid
            else:
                l = mid + 1
        return self.nums[l]
```
[<-PREV](dsa.md)
