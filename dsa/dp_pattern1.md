# Dynamic Programming 
# Pattern 1. Min/Max value to reach a target

Given a target, find the minimum/maximum cost/path/sum to reach the target.
- 0/1 Knapsack
- Unbounded knapsack
- Coin Change
- Min Cost Climbing Stairs
- Minimum Path Sum
- Minimum Cost For Tickets
- 2 Keys Keyboard
- Perfect Squares

## 0/1 Knapsack 
Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value in the knapsack. Find the maximum value subset of value such that sum of the weights of this subset is smaller than or equal to W. You cannot break an item, either pick the complete item or don’t pick it (0-1 property).

value = [10, 15, 40]

weight = [1, 2, 3]

W = 6


```python
def zero_one_knapsack(value, weight, W):
    """
    dp[i][j] the maximum value to make weight j using 0~i-1 items
    dp[i][j] = max(dp[i-1][j], value[i-1] + dp[i-1][j-weight[i-1]]) 
             = dp[i-1][j] 
    item\W   0 1 2 3 4 5 6 
        0    0 0 0 0 0 0 0 
        1    0 
        2    0 
        3    0
    """
    n = len(value)
    dp = [[0] * (W+1) for _ in range(n+1)]

    for i in range(1, n+1): # item
        for j in range(1, W+1): # weight
            if weight[i-1] <= j:
                dp[i][j] = max(dp[i-1][j], value[i-1] + dp[i - 1][j-weight[i-1]])
            else:
                dp[i][j] = dp[i-1][j]

    return dp[-1][-1]

def optimized_zero_one_knapsack(value, weight, W):
    """
    dp[i] the maximum value to make weight j using 0~i-1 items

    """
    n = len(value)
    dp = [0] * (W+1)

    for i in range(1, n+1):
        for j in range(W, 0, -1):
            if weight[i-1] <= j:
                dp[j] = max(dp[j], value[i-1] + dp[j-weight[i-1]])

    return dp[-1]

value = [10, 15, 40]
weight = [1, 2, 3]
W = 6

zero_one_knapsack(value, weight, W)
optimized_zero_one_knapsack(value, weight, W)

```




    65



## Unbounded Knapsack

Given a knapsack weight W and a set of n items with certain value and weight, find the maximum value that could make up this quantity exactly when repetition of items allowed.


```python
def unbounded_knapsack(value, weight, W):
    """
    dp[i]: the maximum value to make weight i using items
    dp[i] = max(dp[i], value[j] + dp[i - weight[j]])
    weight 0 1 2 3 4 5 6

    """
    n = len(value)
    dp = [0] * (W + 1)
    for i in range(1, W + 1):
        for j in range(n):
            if weight[j] <= i:
                dp[i] = max(dp[i], value[j] + dp[i-weight[j]])
    
    return dp[-1]

value = [10, 15, 40]
weight = [1, 2, 3]
W = 6
unbounded_knapsack(value, weight, W)
```




    80



## [Coin Change](https://leetcode.com/problems/coin-change/)
*Return the fewest number of coins that you need to make up that amount.*


```python
def coinChange(self, coins: List[int], amount: int) -> int:
    """
    dp[i]: the min number of coins to make amount i using all coins 
    dp[i] = min(dp[i], 1 + dp[i-coins[j]]) if coins[j] <= i
    amount 0 1 2 3 4 5 6 7 8 9 10 11
           0 
    """
    
    n = len(coins)
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for j in range(n):
            if coins[j] <= i:
                dp[i] = min(dp[i], 1 + dp[i-coins[j]])
    return dp[-1] if dp[-1]!= float('inf') else -1
```

## [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/)
*Return the minimum cost to reach the top of the floor.*


```python
def minCostClimbingStairs(self, cost: List[int]) -> int:
    """
    dp[i]: the minimum cost to reach i
    dp[i] = cost[i] + min(dp[i-2], dp[i-1])
    
    """
    n = len(cost)
    dp = [0] * n
    dp[0], dp[1] = cost[0], cost[1]
    for i in range(2, n):
        dp[i] = cost[i] + min(dp[i-2], dp[i-1])

    return min(dp[-2], dp[-1])
```

## [Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)
*Find a path from top left to bottom right, which minimizes the sum of all numbers along its path.*


```python
def minPathSum(self, grid: List[List[int]]) -> int:
    """
    grid[i][j]: the minimum sum upto the path (i, j)
        1 3 1
        1
        4
    grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        1 4 5
        2
        6
    
    """
    m, n = len(grid), len(grid[0])
    for i in range(1, m):
        grid[i][0] += grid[i-1][0]
    for i in range(1, n):
        grid[0][i] += grid[0][i-1]
    
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    
    return grid[-1][-1]
```

## [Minimum Falling Path Sum](https://leetcode.com/problems/minimum-falling-path-sum/)
*Return the minimum sum of any falling path through matrix.*


```python
def minFallingPathSum(self, A: List[List[int]]) -> int:
    """
    A[i][j]: the minimum path sum upto (i, j)
    
    """
    
    n = len(A)
    
    for i in range(1, n):
        for j in range(n):
            
            A[i][j] += min(A[i-1][max(0, j-1):min(n-1, j+1) + 1])
    
    return min(A[n-1])
```

## [Minimum Cost For Tickets](https://leetcode.com/problems/minimum-cost-for-tickets/)
*Return the minimum number of dollars you need to travel every day in the given list of days.*


```python
def mincostTickets(self, days: List[int], costs: List[int]) -> int:
    """

    dp[i]: the minimum cost to travel until day i
    
    dp[i] = min(dp[i-1] + costs[0], dp[max(0, i-7)] + costs[1], dp[max(0, i-30)] + costs[2])
    dp[i] = dp[i-1]
    
    day 0 1 2 3 4 5 6 7 8 9 ... 20
        0 inf inf 
    
    """
    n = days[-1]
    
    days = set(days)
    
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n+1):
        if i not in days:
            dp[i] = dp[i-1]
        else:
            dp[i] = min(dp[i-1] + costs[0], dp[max(0, i-7)] + costs[1], dp[max(0, i-30)] + costs[2])
            
    return dp[-1]
```

## [2 Keys Keyboard](https://leetcode.com/problems/2-keys-keyboard/)
*Return the minimum number of operations to get the character 'A' exactly n times on the screen.*


```python
def minSteps(self, n: int) -> int:
    """
    dp[i]: the minimum number of operations to get 'A' i times

    """
    # Solution 1
    dp = [ i for i in range(n+1)]
    dp[0] = dp[1] = 0
    for i in range(2, n+1):
        
        j = 2
        while i * j < n + 1:
            dp[i * j] = min(dp[i * j], dp[i] + j)
            j += 1
            
    return dp[-1]

    # Solution 2
    cnt = 0
    i = 2
    while i < n + 1:
        if n % i == 0:
            cnt += i 
            n //= i
        else:
            i += 1
            
    return cnt
```

## [Perfect Squares](https://leetcode.com/problems/perfect-squares/)
*Return the least number of perfect square numbers that sum to n*


```python
def numSquares(self, n: int) -> int:
    """
    dp[i]: the minimum number of perfect square numbers that sum to i

    """
    squares = [ i*i for i in range(1, int(sqrt(n)) + 1)]
    
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        for square in squares:
            if square <= i:
                dp[i] = min(dp[i], dp[i - square] + 1)
    
    return dp[-1]
```

## [Last Stone Weight II](https://leetcode.com/problems/last-stone-weight-ii/)
*Return the smallest possible weight of the left stone.(The minimum difference between the sum of two groups.)*


```python
def lastStoneWeightII(self, stones: List[int]) -> int:
    """
    dp = {0}
    stones = 2, 7, 4, 1, 8, 1
                +  +  +  +  +  +
                -  -  -  -  -  -
    """
    
    dp = {0}
    for stone in stones:
        temp_dp = set()
        for i in dp:
            temp_dp.add(stone + i)
            temp_dp.add(abs(stone - i))
        dp = temp_dp
    return min(dp)
```

## [Triangle](https://leetcode.com/problems/triangle/)
*Return the minimum path sum from top to bottom.*


```python
def minimumTotal(self, triangle: List[List[int]]) -> int:
    """
    triangle[i][j]: the minimum path sum to (i, j)

    """
    n = len(triangle)
    
    for i in range(1, n):
        triangle[i][0] += triangle[i-1][0]
        triangle[i][i] += triangle[i-1][i-1] 
        for j in range(1, i):
            triangle[i][j] += min(triangle[i-1][j-1], triangle[i-1][j])
    
    return min(triangle[-1])
```

## [Ones and Zeros](https://leetcode.com/problems/ones-and-zeroes/)
*Return the size of the largest subset*


```python
def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
    """
    dp[k][i][j]: the maximum number of strings that can be included in the subset 
                    given only i 0's and j 1's using 0 to k-1 items in strs
    
    """
    # Basic Solution: O(k * m * n) & O(k * m * n)
    len_strs = len(strs)
    for i in range(len_strs):
        strs[i] = collections.Counter(strs[i])
    
    dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(len_strs + 1)]
            
    for k in range(1, len_strs + 1):
        zeros = strs[k-1]['0']
        ones = strs[k-1]['1']
        for i in range(m + 1):
            for j in range(n + 1):
                if i >= zeros and j >= ones:
                    dp[k][i][j] = max(dp[k-1][i][j], dp[k-1][i - zeros][j - ones] + 1)
                else:
                    dp[k][i][j] = dp[k-1][i][j]

    return dp[-1][-1][-1]


    """
    dp[i][j]: the maximum number of strings that can be included in the subset 
                given only i 0's and j 1's
    """
    # Optimized Solution: O(k * m * n) & O(m * n)

    len_strs = len(strs)
    for i in range(len_strs):
        strs[i] = collections.Counter(strs[i])
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
            
    for k in range(len_strs):
        zeros = strs[k]['0']
        ones = strs[k]['1']

        for i in range(m, -1, -1):
            for j in range(n, -1, -1):
                if i >= zeros and j >= ones:
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)

    return dp[-1][-1]
```

## [Maximal Square](https://leetcode.com/problems/maximal-square/)
*Find the largest square containing only 1's*


```python
def maximalSquare(self, matrix: List[List[str]]) -> int:
    """
    dp[i][j]: the longest length of square at (i+1, j+1)
    
    """
    # Basic Solution: O(m * n) & O(m * n)
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    maxlen = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i-1][j-1] == '1':
                dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1
                maxlen = max(maxlen, dp[i][j])
    
    return maxlen ** 2
    
    """
    dp[i]: the longest length of square at column i 
    """
    # Optimized Solution 1. O(m * n) & O(m + n)
    m, n = len(matrix), len(matrix[0])
    dp = [0] * (n + 1)
    maxlen = 0
    for i in range(1, m + 1):
        temp_dp = [0] * (n + 1)
        for j in range(1, n + 1):
            if matrix[i-1][j-1] == '1':
                temp_dp[j] = min(dp[j-1], dp[j], temp_dp[j-1]) + 1
                maxlen = max(maxlen, temp_dp[j])
        dp = temp_dp
    return maxlen ** 2
            
    # Optimized Solution 2. O(m * n) & O(n)
    m, n = len(matrix), len(matrix[0])
    dp = [0] * (n + 1)
    maxlen = 0
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            if matrix[i-1][j-1] == '1':
                dp[j] = min(dp[j-1], dp[j], prev) + 1
                maxlen = max(maxlen, dp[j])
                prev = dp[j]
            else:
                prev = 0
    return maxlen ** 2
```
