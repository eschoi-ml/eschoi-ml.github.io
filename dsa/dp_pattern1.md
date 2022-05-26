[<-PREV](dsa.md)

# Dynamic Programming 
# Pattern 1. Min/Max value

Given a target, find the minimum/maximum cost/path/sum to reach the target.
- 0/1 Knapsack
- Ones and Zeros
- Unbounded knapsack
- Coin Change
- Min Cost Climbing Stairs
- Minimum Path Sum
- Minimum Cost For Tickets
- 2 Keys Keyboard
- Perfect Squares
- Triangle
- Maximal Square
- Last Stone Weight II
- Partition Equal Subset Sum

## 0/1 Knapsack 
Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value in the knapsack. Find the maximum value subset of value such that sum of the weights of this subset is smaller than or equal to W. You cannot break an item, either pick the complete item or don’t pick it (0-1 property).

value = [10, 15, 40]

weight = [1, 2, 3]

W = 6


```python
def zero_one_knapsack(value, weight, W):
    """
    dp[k][i]: the max value to make i weights using k items  
    
    k\i 0 1 2 3 ... W
    0
    1
    2
    .
    n
    
    """
    
    n = len(value)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    
    for k in range(1, n + 1):
        v, w = value[k-1], weight[k-1]
        for i in range(W + 1):
            if w <= i:
                dp[k][i] = max(dp[k-1][i], dp[k-1][i-w] + v)
            else:
                dp[k][i] = dp[k-1][i]
    return dp[-1][-1]

def optimized_zero_one_knapsack(value, weight, W):
    """
    dp[i]: the max value to make i weights using 1 to k items  
    
    i 0 1 2 3 ... W
              <-start
    """
    
    n = len(value)
    dp = [0] * (W + 1)
    for k in range(1, n + 1):
        v, w = value[k-1], weight[k-1]
        for i in range(W, -1, -1):
            if w <= i:
                dp[i] = max(dp[i], dp[i-w] + v)
    return dp[-1]

zero_one_knapsack(value, weight, W)
optimized_zero_one_knapsack(value, weight, W)

```




    65



## [Ones and Zeros](https://leetcode.com/problems/ones-and-zeroes/)
*Return the size of the largest subset*


```python
def findMaxForm(self, strs: List[str], m: int, n: int) -> int:

    # Sol1. Basic Solution O(nstr * m * n) & O(nstr * m * n)
    """
    dp[k][i][j]: max # of subset of strs from 0 to k-1, with i 0s and j 1s 

    """
    nstr = len(strs)
    cnt_str = [0] * nstr
    for i in range(nstr):
        cnt_str[i] = collections.Counter(strs[i])


    dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(nstr + 1)]

    for k in range(1, nstr+1):
        zeros = cnt_str[k-1]['0']
        ones = cnt_str[k-1]['1']
        for i in range(m + 1):
            for j in range(n + 1):
                if zeros <= i and ones <= j:
                    dp[k][i][j] = max(dp[k-1][i][j], dp[k-1][i-zeros][j-ones] + 1)
                else:
                    dp[k][i][j] = dp[k-1][i][j]
    return dp[-1][-1][-1]

    # Sol2. Optimization O(nstr * m * n) & O(m * n)
    """
    dp[i][j]: max # of subset of strs from 0 to k-1 total nstr items, with i 0s and j 1s 

    """

    nstr = len(strs)
    cnt_str = [0] * nstr
    for i in range(nstr):
        cnt_str[i] = collections.Counter(strs[i])

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for k in range(1, nstr+1):
        zeros = cnt_str[k-1]['0']
        ones = cnt_str[k-1]['1']

        for i in range(m, -1, -1):
            for j in range(n, -1, -1):
                if zeros <= i and ones <= j:
                    dp[i][j] = max(dp[i][j], dp[i-zeros][j-ones] + 1)
    return dp[-1][-1]
```

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
        dp[i]: the min # of coins that you need to make up the amount i using coins
        
        if coin <= amount:
            dp[i] = min(dp[i], dp[i-coin]) 
    """
    
    dp = [float('inf')]*(amount + 1)
    dp[0] = 0


    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i-coin] + 1)

    return dp[-1] if dp[-1]!=float('inf') else -1
```

## [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/)
*Return the minimum cost to reach the top of the floor.*


```python
def minCostClimbingStairs(self, cost: List[int]) -> int:
    """
    dp[i]: the min cost to reach the i steps

    dp[0] = dp[1] = 0

    dp[i] = min(dp[i], dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])

    """

    n = len(cost)
    dp = [float('inf')] * (n + 1)
    dp[0] = dp[1] = 0

    for i in range(2, n+1):
        dp[i] = min(dp[i], dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])

    return dp[-1]
```

## [Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)
*Find a path from top left to bottom right, which minimizes the sum of all numbers along its path.*


```python
def minPathSum(self, grid: List[List[int]]) -> int:
    """
    dp[i][j]: the min path sum to (i, j)

    dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

    """
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]


    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

    return dp[-1][-1]
```

## [Minimum Falling Path Sum](https://leetcode.com/problems/minimum-falling-path-sum/)
*Return the minimum sum of any falling path through matrix.*


```python
def minFallingPathSum(self, matrix: List[List[int]]) -> int:
    """
    dp[i][j]: the min sum of falling path to (i, j)

    dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1]) + matrix[i][j]

    """
    n = len(matrix)
    dp = [[0] * n for _ in range(n)]
    for j in range(n):
        dp[0][j] = matrix[0][j]


    for i in range(1, n):
        for j in range(n):
            dp[i][j] = min(dp[i-1][max(0, j-1):min(j+2, n)]) + matrix[i][j]

    return min(dp[-1])
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

    # Solution 2 O(sqrt(n)) & O(1)
    res = 0
    d = 2
    while n > 1:
        while n % d == 0:
            res += d
            n //= d
        d += 1
    return res
```

## [Perfect Squares](https://leetcode.com/problems/perfect-squares/)
*Return the least number of perfect square numbers that sum to n*


```python
def numSquares(self, n: int) -> int:
    """
     dp[i]: the min # of perfect square that sum to i

    """
    # Sol1. DP O(n*sqrt(n)), O(n) 

    dp = [float('inf')] * (n+1)
    dp[0] = 0

    for i in range(1, n+1):
        j = 1
        while j*j <= i:
            dp[i] = min(dp[i], dp[i-j*j] + 1)
            j += 1
    return dp[-1]


    # Sol2. BFS

    q = collections.deque([n])
    visited = set([n])
    depth = -1
    while q:
        depth += 1
        size = len(q)

        for _ in range(size):

            num = q.popleft()
            if num == 0:
                return depth

            j = 1
            while j * j <= num:
                neighbor = num - j*j
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
                j += 1
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
        for j in range(i+1):
            triangle[i][j] += min(triangle[i-1][max(0, j-1)], triangle[i-1][min(j, i-1)]) 
    return min(triangle[-1])
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

## [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)
*Find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.*

```python
def canPartition(self, nums: List[int]) -> bool:
    """
    h = {0}
    nums = [1,  5,  11,  5]
            +/0 +/0 +/0 +/0
    """
    
    target = sum(nums)
    if target % 2 == 1:
        return False
    target //= 2
    
    h = {0}
    for num in nums:
        curr_h = set()
        for i in h:
            curr_sum = num + i
            if curr_sum == target:
                return True
            if curr_sum < target:
                curr_h.add(curr_sum)
        h = h | curr_h

    return False
```


[<-PREV](dsa.md)
