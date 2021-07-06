[<-PREV](dsa.md)

# Dynamic Programming
# Pattern 3. Merging intervals
Given a set of numbers find an optimal solution for a problem considering the current number and the best you can get from the left and right sides.

    l = 1/2, ..., n-1 (distance or difference)
    i = 0, ..., n-1-l (start)
    j = i + l = l, ..., n-1 (end)
    k = i/i+1, ..., j-1/j (possibility between start and end)

- Unique Binary Search Trees
- Minimum Cost Tree From Leaf Values
- Minimum Score Triangulation of Polygon
- Burst Balloons
- Remove Boxes
- Guess Number Higher or Lower II
- Minimum Cost to Merge Stones

## [Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)
*Given an integer n, return the number of structurally unique BST's (binary search trees)*


```python
def numTrees(self, n: int) -> int:
    """
    dp[i]: the number of structurally unique BST's
    
    1, 2, 3, ..., j-1, j, j+1, ..., i
    |_______________|      |________|
        j - 1                i - j
        dp[j-1]         *   dp[i-j]   
    """
    
    dp = [0] * (n + 1)
    dp[1] = dp[0] = 1
    
    for i in range(2, n + 1):
        for j in range(1, i + 1):
            dp[i] += dp[j-1] * dp[i-j]
    
    return dp[-1]
```

## [Minimum Cost Tree From Leaf Values](https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/)
*Among all possible binary trees considered, return the smallest possible sum of the values of each non-leaf node.*


```python
def mctFromLeafValues(self, arr: List[int]) -> int:
    """
    dp[i][j]: the minimum sum of non-leaf nodes from arr[i] to arr[j] 
    
    l = 1 ~ n-1
    i = 0 ~ n-l-1
    j = i + l = l ~ n-1
    i~k/ k+1:j
    k = i~j-1
    
    for k in range(i, j+1):
        dp[i][j] = min(dp[i][k] + max(arr[i:k+1]) * max(arr[k+1:j+1]) + dp[k+1][j])
        6   2   4
    i\j 0   1   2 
    6 0 0      ans
    2 1 -   0   
    4 2 -   -   0
    
    """
    n = len(arr)
    dp = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 0
        
    for l in range(1, n): 
        for i in range(n - l): 
            j = i + l 
            for k in range(i, j):
                rootVal = max(arr[i:k+1]) * max(arr[k+1:j+1])
                dp[i][j] = min(dp[i][j], dp[i][k] + rootVal + dp[k+1][j])

    return dp[0][-1]
```

## [Minimum Score Triangulation of Polygon](https://leetcode.com/problems/minimum-score-triangulation-of-polygon/)
*Return the smallest possible total score that you can achieve with some triangulation of the polygon.*


```python
def minScoreTriangulation(self, values: List[int]) -> int:
    """
    dp[i][j]: the smallest possilbe total score from i to j
           
       i\j 0  1  2  3
        0  _  0    ans 
        1  _  _  0
        2  _  _  _  0
        3  _  _  _  _ 
    
    - l: the distance from a start point to an end point of a triangle bottom, l = 2, ..., n-1
    range = i, i+1, ..., j-1, j
    - i: start point, i = 0, ..., n-l-1
    - j: end point, j = i + l = l, ..., n-1
    - k: triangle third point, k = i+1, ..., j-1 
    i~k//k~j
    
    dp[i][j] = min(dp[i][j], dp[i][k] + values[i] * values[k] * values[j] + dp[k][j])
    
    """
    n = len(values)
    dp = [[float('inf')] * n for _ in range(n)]
    for i in range(n-1):
        dp[i][i+1] = 0
    
    for l in range(2, n):
        for i in range(n-l):
            j = i + l
            value = values[i] * values[j]
            for k in range(i+1, j):
                dp[i][j] = min(dp[i][j], dp[i][k] + value * values[k] + dp[k][j])
    
    return dp[0][-1]
```

## [Burst Balloons](https://leetcode.com/problems/burst-balloons/)
*Return the maximum coins you can collect by bursting the balloons wisely.*


```python
def maxCoins(self, nums: List[int]) -> int:
    """
    brute force: n! 
    n = n + 2
    dp[i][j]: the maximum coins you can collect by bursting the balloons 
                between nums[i] to nums[j] exclusively (i, j)
    
    
    nums = [1,3,1,5,8,1]
          i\j 0 1 2 3 4 5
            0 _ _      ans
            1 _ _ _ 
            2 _ _ _ _ 
            3 _ _ _ _ _ 
            4 _ _ _ _ _ _ 
            5 _ _ _ _ _ _
    
    - l: the length of subarray, l=2, ..., n-1 (min=3)
    - i: start point exclusively, i=0, ..., n-l-1
    - j: end point exclusively, j=i+l=l, ..., n-1
    - k: last bursting point, k = i+1, ..., j-1
    k=i+1, ..., j-1
    
    dp[i][j] = max(dp[i][j], dp[i][k] + nums[i]*nums[k]*nums[j] + dp[k][j])
          
    """
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    
    for l in range(2, n):
        for i in range(n - l):
            j = i + l
            for k in range(i+1, j):
                dp[i][j] = max(dp[i][j], dp[i][k] + nums[i] * nums[k] * nums[j] + dp[k][j])
    
    nums.pop()
    nums = nums[1:]
    
    return dp[0][-1]
```

## [Remove Boxes](https://leetcode.com/problems/remove-boxes/)
*Return the maximum points you can get.*


```python
def removeBoxes(self, boxes: List[int]) -> int:
    """
    dp[i][j][k]: the maximum points you can get from i to j
                    with k boxes "right before" i that has the same color as i
            k=2  i  j     
    [1, 2, 2, 2, 3, 1]

    l = 1, ..., n-1
    i = 0, ..., n-1-l
    j = i + l = l, ..., n-1
    k = 0, ..., i
    
    m = i+1, ..., j
    
    i, i+1, ...,m-1, m, m+1, ..., j-1, j
    
    Choice: either to remove or keep ith box
    1) remove ith box: k + 1 continuous boxes until i and 0 continuous box before (i+1)th box 
    dp[i][j][k] = (k + 1) * (k + 1) + dp[i+1][j][0]
    2) keep ith box: find mth box from i+1 to j that has the same color as ith box
    dp[i][j][k] = dp[i+1][m-1][0] (exclusively take into account all boxes from i+1 to m-1) 
                + dp[m+1][j][k+1] (mergingly take into account k continuous boxes before mth box)
    
    return dp[0][-1][0]: return the maximum points from 0 to n-1 with 0 boxes same color as 0th box
    
    """
    
    n = len(boxes)
    dp = [[[0] * n for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for k in range(i+1):
            dp[i][i][k] = (k+1)*(k+1)
    
    for l in range(1, n):
        for i in range(n-l):
            j = i + l
            for k in range(i+1):
                # remove ith box 
                dp[i][j][k] = (k + 1) * (k + 1) + dp[i+1][j][0]
                
                # keep ith box
                for m in range(i+1, j+1):
                    if boxes[i] == boxes[m]:
                        dp[i][j][k] = max(dp[i][j][k], dp[i+1][m-1][0] + dp[m][j][k+1])
                    
    return dp[0][n-1][0]
```

## [Guess Number Higher or Lower II](https://leetcode.com/problems/guess-number-higher-or-lower-ii/)
*Return the minimum amount of money you need to guarantee a win regardless of what number I pick.*


```python
def getMoneyAmount(self, n: int) -> int:
    """
    dp[i][j]: the minimum amount of money you need to guarantee a win from i to j
    
    k = i, ..., j
    dp[i][j] = min(dp[i][j], k+1 + max(dp[i][k-1], dp[k+1][j]))
    n = 5
      i\j 0 1 2 3 4
        0 0 1
        1 _ 0 2
        2 _ _ 0 3
        3 _ _ _ 0 4
        4 _ _ _ _ 0
    
    l = 2, ..., n-1
    i = 0, ..., n-1-l
    j = i + l = l, ..., n-1
    k = i+1, ..., j-1
    
    """
    
    dp = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 0
        if i < n-1:
            dp[i][i+1] = i+1
    
    for l in range(2, n):
        for i in range(n-l):
            j = i + l
            for k in range(i+1, j):
                dp[i][j] = min(dp[i][j], k+1 + max(dp[i][k-1], dp[k+1][j]))

    return dp[0][-1]
```

## [Minimum Cost to Merge Stones](https://leetcode.com/problems/minimum-cost-to-merge-stones/)
*Return the minimum cost to merge all piles of stones into one pile.*


```python
def mergeStones(self, stones: List[int], K: int) -> int:
    """
    
    # edge case
    n  - k(K - 1) = 1
    n - 1 = k * (K - 1)
    (n - 1) % (K - 1) = 0
    
    dp[i][j]: the minimum cost to merge piles from stones[i] to stones[j] into one pile
    l = K-1, ..., n-1
    i = 0, ..., n - l - 1
    j = i + l = l, ..., n - 1
    i, i+1, ..., j-1, j
    |_______ l+1______|
    k = i, i + (K - 1), i + 2*(K - 1), ..., j-1 # size: (1, l) (1 + (K-1), rest), (1 + 2 * (K-1), rest)
    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j])
    
    K = 3
      i\j 0 1 2 3 4
        0 0 0    ans
        1 _ 0 0
        2 _ _ 0 0
        3 _ _ _ 0 0
        4 _ _ _ _ 0
    
    """
    n = len(stones)
    # edge case
    if (n - 1) % (K - 1) != 0:
        return -1
    
    # partial sum
    p = [0]
    for i in range(n):
        p.append(p[i] + stones[i])
    
    # dp
    dp = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, min(n, i + K - 1)):
            dp[i][j] = 0
    
    for l in range(K-1, n):
        for i in range(n - l):
            j = i + l
            for k in range(i, j, K-1):
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j])
            if (j - i + 1 - 1) % (K - 1) == 0:
                dp[i][j] += p[j+1] - p[i]

    return dp[0][-1]
```

[<-PREV](dsa.md)
