[<-PREV](dsa.md)

# Monotonic Stack
- Monotonic **increasing** vs. **decreasing** stack
- Save **value** vs. **index** in the stack


Problem list: 
- [Minimum Cost Tree From Leaf Values](https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/)
- [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)
- [Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)


## [Minimum Cost Tree From Leaf Values](https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/)


```python
def mctFromLeafValues(self, arr: List[int]) -> int:
    """
    dp[i][j]: the minimum sum of non-leaf nodes from arr[i] to arr[j] 
    
    for k in range(i, j+1):
        dp[i][j] = min(dp[i][k] + max(arr[i:k+1]) * max(arr[k+1:j+1]) + dp[k+1][j])
    
    l = 1, ..., n-1
    i = 0, ..., n-l-1
    j = i + l = l, ..., n-1
    k: left(i~k) and right(k+1~j) subtrees dividor, k = i, ..., j-1 
    
    
        6   2   4
    i\j 0   1   2 
    6 0 0      ans
    2 1 -   0   
    4 2 -   -   0
    
    """
    # Solution1. dynamic programming
    # O(n^4) & O(n^2)
    
    n = len(arr)
    dp = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 0
        
    for l in range(1, n): # O(n)
        for i in range(n - l): # *O(n)
            j = i + l 
            for k in range(i, j):
                rootVal = max(arr[i:k+1]) * max(arr[k+1:j+1])   # *O(n*n)
                dp[i][j] = min(dp[i][j], dp[i][k] + rootVal + dp[k+1][j])

    return dp[0][-1]
    
    """
    arr = [6, 2, 4]
    min = 2: 2*4 = 8
    arr = [6, 4]
    min = 4: 6*4 = 24
    arr = [6]
        24 
        (6) 8
        (2) (4)
    """
    # Solution 2. greedy algorithm 
    # O(n^2), O(1)
    res = 0
    while len(arr) > 1: # O(n)
        idx = arr.index(min(arr)) # O(n)
        if idx == 0:
            res += arr[idx] * arr[idx + 1]            
        elif idx == len(arr) - 1:
            res += arr[idx-1] * arr[idx]
        else:
            res += arr[idx] * min(arr[idx-1], arr[idx + 1])
        
        arr.pop(idx)
        
    return res

    """
    arr = [6, 2, 4]
    num = 6
    stack = [inf, 6]
    num = 2
    stack = [inf, 6, 2]
    num = 4
        stack = [inf, 6]
        curr = 2
        res += 2 * min(4, 6) += 8
    stack = [inf, 6, 4]    
    
    res += 4 * 6 += 24
    
    """
    # Solution3. monotonic decresing stack
    # O(n), O(1)
    stack = []
    res = 0
    for num in arr:
        while stack and stack[-1] <= num:
            curr = stack.pop()
            if stack:
                res += curr * min(stack[-1], num)
            else:
                res += curr * num
        stack.append(num)
        
    while len(stack) >= 2:
        res += stack.pop() * stack[-1]
    
    return res
```

## [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)


```python
def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
    
    # Solution1. Brute force
    # O(m * n) & O(m)
    
    h = {}
    for num in nums1:
        h[num] = -1
        
    m = len(nums2)
    for i, num in enumerate(nums2):
        if num in h:
            j = i+1
            while j < m and num >= nums2[j]:
                j += 1
            if j < m:
                h[num] = nums2[j]
    
    return h.values()
    
    """
    h = {4:-1, 1:3, 2:-1}
    nums2 = [1, 3, 4, 2]
    num = 1
    stack = [1]
    num = 3
    stack = [3]
    num = 4
    stack = [4]
    num = 2
    stack = [4, 2]
    
    """
    # Solution2. monotonic decreasing stack
    # O(m + n) & O(m + n)
    
    h = {}
    for num in nums1:
        h[num] = -1
    
    stack = []
    for num in nums2:
        while stack and stack[-1] < num:
            curr = stack.pop()
            if curr in h:
                h[curr] = num
        stack.append(num)

    return h.values()
```

## [Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)


```python
def nextGreaterElements(self, nums: List[int]) -> List[int]:
    """
            0  1. 2.  3. 4
    nums = [4, 5, 10, 3, 10]
                0  1.  2. 3. 4
                3. 4.  0. 1. 2
    nums2 = [3, 10, 4, 5, 10]
    res =   [-1,-1,-1,-1,-1]

    """
    # O(n) & O(n)

    maxIdx = nums.index(max(nums))
    n = len(nums)
    res = [-1] * n
    stack = []

    for i in range(n):
        
        idx = (maxIdx + 1 + i) % n
        num = nums[idx]
        
        while stack and nums[stack[-1]] < num:
            curr = stack.pop()
            res[curr] = num
        stack.append(idx)

    return res   
```

[<-PREV](dsa.md)
