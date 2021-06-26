[<-PREV](dsa.md)

# Dynamic Programming
# Pattern 2. Distinct ways

Given a target, find the number of distinct ways to reacch the target.

**Pattern 2.1 dp[i] += dp[i-k]**
- Climbing Stairs
- Combination Sum IV
- Number of Dice Rolls With Target Sum
- Unique Paths
- Unique Paths II
- Soup Servings
- Longest Increasing Subsequence Series
    - Longest Continuous Increasing Subsequence
    - Longest Increasing Subsequence
    - Number of Longest Increasing Subsequence
- Dice Roll Simulation

**Pattern 2.2 dp = { initial_state : 1 }, curr_dp = collections.defaultdict(int)**
- Target Sum
- Knight Probability in Chessboard
- Out of Boundary Paths
- Number of Ways to Stay in the Same Place After Some Steps

**Pattern 2.3 dp = [1, 1, ..., 1], curr_dp = [0, 0, ..., 0 ]**
- Knight Dialer
- Count Vowels Permutation
- Domino and Tromino Tiling


# Pattern 2.1 dp[i] += dp[i-k]



## [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)
*In how many distinct ways can you climb to the top?*


```python
def climbStairs(self, n: int) -> int:
    """
    dp[i]: the number of distinct ways to climb i steps
    dp[i] = dp[i-2] + dp[i-1] 
    
    """
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        for step in range(1, 3):
            dp[i] += dp[i - step]
        #dp[i] = dp[i-2] + dp[i-1]
    return dp[-1]
```

## [Combination Sum IV](https://leetcode.com/problems/combination-sum-iv/)
*Return the number of possible combinations that add up to target.*


```python
def combinationSum4(self, nums: List[int], target: int) -> int:
    """
    dp[i]: the number of possible combinations that add up to i
    
    """
    n = len(nums)
    dp = [0] * (target + 1) 
    dp[0] = 1
    
    for i in range(1, target + 1):
        for num in nums:
            if num <= i:
                dp[i] += dp[i - num]
    return dp[-1]
```

## [Number of Dice Rolls With Target Sum](https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/)
*Return the number of possible ways (out of f^d total ways) modulo 10^9 + 7 to roll the dice so the sum of the face-up numbers equals target.*


```python
def numRollsToTarget(self, d: int, f: int, target: int) -> int:
    """
    dp[i][j]: the number of possible ways to roll the dice at [i+1]th roll to make the sum [j] 
    dp[i][j] += dp[i-1][j-k] for k = 1 to f if k <= j
    
    """
    MOD = 10**9 + 7
    # edge cases
    if not d <= target <= d * f:
        return 0
    if d == 1:
        return 1 if 1 <= target <= f else 0
            
    dp = [[0] * (target + 1) for _ in range(d + 1)]
    dp[0][0] = 1

    for i in range(1, d + 1):
        for j in range(1, target + 1):
            for k in range(1, f + 1):
                if k <= j:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - k]) % MOD
    return dp[-1][-1]
```

## [Unique Paths](https://leetcode.com/problems/unique-paths/)
*How many possible unique paths are there?*


```python
def uniquePaths(self, m: int, n: int) -> int:
    """
    dp[i][j]: the number of unique paths to reach (i, j)
    dp[i][j] = dp[i][j-1] + dp[i-1][j]
    """
    # Solution 1. O(m * n) & O(m * n)
    
    dp = [[0] * n for _ in range(m)]
    for i in range(m):
        dp[i][0] = 1
    for i in range(n):
        dp[0][i] = 1
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[-1][-1]
    
    # Solution 2. O(m * n) & O(n)
    dp = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1] 
    return dp[-1]
```

## [Unique Paths II](https://leetcode.com/problems/unique-paths-ii/)
*How many unique paths would there be?*


```python
def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
    
    if obstacleGrid[0][0]==1 or obstacleGrid[-1][-1]==1:
        return 0
    
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j]==1:
                obstacleGrid[i][j] = -1
    
    i = 0
    while i < m and obstacleGrid[i][0] != -1:
        obstacleGrid[i][0] = 1
        i += 1
    i = 1
    while i < n and obstacleGrid[0][i] != -1:
        obstacleGrid[0][i] = 1
        i += 1
    
    for i in range(1, m):
        for j in range(1, n):
            if obstacleGrid[i][j] != -1:
                obstacleGrid[i][j] = max(0, obstacleGrid[i-1][j]) + max(0, obstacleGrid[i][j-1])
    return obstacleGrid[-1][-1]

def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
    
    if obstacleGrid[0][0]==1 or obstacleGrid[-1][-1]==1:
        return 0
    
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    obstacleGrid[0][0] = 1
    for i in range(1, m):
        obstacleGrid[i][0] = int(obstacleGrid[i][0]==0 and obstacleGrid[i-1][0]==1)
    for i in range(1, n):
        obstacleGrid[0][i] = int(obstacleGrid[0][i]==0 and obstacleGrid[0][i-1]==1)
        
    for i in range(1, m):
        for j in range(1, n):
            if obstacleGrid[i][j] == 0:
                obstacleGrid[i][j] = obstacleGrid[i-1][j] + obstacleGrid[i][j-1]
            else:
                obstacleGrid[i][j] = 0
    return obstacleGrid[-1][-1]
```

## [Soup Servings](https://leetcode.com/problems/soup-servings/)
*Return the probability that soup A will be empty first, plus half the probability that A and B become empty at the same time.*


```python
def soupServings(self, n: int) -> float:
    """
    1 serving = 25 ml
    dp[i][j]: the probability that i serving from A is first empty + half the probability i servings from A and j servings from B become empty at the same time
    
    ex) n = 100 -> n = 4

    i(A)\j(B) 0    1    2    3    4
            0 0.5  1    1    1    1   
            1   0
            2   0
            3   0
            4   0
            
    (100, 0) ->(4, 0)
    (75, 25) ->(3, 1) 
    (50, 50) ->(2, 2)
    (25, 75) ->(1, 3)
    
    """
    if n >= 4800:
        return 1.0
    
    n = (n + 24)//25
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 0.5
    for i in range(1, n + 1):
        dp[0][i] = 1
    
    
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            
            candidates = [
                (i-4, j),
                (i-3, j-1),
                (i-2, j-2),
                (i-1, j-3)
            ]
            
            for r, c in candidates:
                dp[i][j] += dp[max(0, r)][max(0, c)]
            dp[i][j] *= 0.25 
            
            # dp[i][j] = 0.25 * (dp[max(0, i-4)][j] + 
            #                    dp[max(0, i-3)][max(0, j-1)] + 
            #                    dp[max(0, i-2)][max(0, j-2)] + 
            #                    dp[max(0, i-1)][max(0, j-3)])
            
    
    return dp[-1][-1]
```

## Longest Increasing Subsequence Series
1. Longest Continuous Increasing Subsequence
1. Longest Increasing Subsequence
1. Number of Longest Increasing Subsequence

### [1. Longest Continuous Increasing Subsequence](https://leetcode.com/problems/longest-continuous-increasing-subsequence/)
*Return the length of the longest continuous increasing subsequence*


```python
def findLengthOfLCIS(self, nums: List[int]) -> int:
    
    n = len(nums)
    maxlen = 0
    l = 0
    while l < n:
        r = l + 1
        while r < n and nums[r-1] < nums[r]:
            r += 1
        maxlen = max(maxlen, r - l)
        l = r
    
    return maxlen
```

### [2. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)
*Return the length of the longest strictly increasing subsequence.*


```python
def lengthOfLIS(self, nums: List[int]) -> int:
    """
    Solution 1. DP
    dp[i]: the length of the longest increasing subsequence that ends at i
    
    """
    # O(n^2) & O(n)
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if dp[i] < dp[j] + 1:
                    dp[i] = dp[j] + 1
    
    return max(dp)
    
    """
    Solution 2. Monotonic increasing stack
    nums = [10,9,2,5,3,7,101,18]
    
    stack = [2, 3, 7, 18]
    
    """
    # O(n^2) & O(n)
    stack = []
    for num in nums:
        if not stack or stack[-1] < num:
            stack.append(num)
        else:
            i = 0
            while stack[i] < num:
                i += 1
            stack[i] = num
    
    return len(stack)
    
    """
    Solution 3. Solution 2 + Binary Search
    """
    # O(n*logn) & O(n)
    stack = []
    for num in nums:
        if not stack or stack[-1] < num:
            stack.append(num)
        else:
            l, r = 0, len(stack) 
            while l < r:
                mid = (l + r) // 2
                if stack[mid] >= num: 
                    r = mid
                else:
                    l = mid + 1
            stack[l] = num
    return len(stack)
```

### [3. Number of Longest Increasing Subsequence](https://leetcode.com/problems/number-of-longest-increasing-subsequence/)
*Return the number of longest increasing subsequences.*


```python
def findNumberOfLIS(self, nums: List[int]) -> int:
    """
    dp[i]: the longest length of increasing subsequences ending with nums[i]
    cnt_dp[i]: the number of longest length of increasing subsequences ending with nums[i]
    
    """
    n = len(nums)
    dp, cnt_dp = [1] * n, [1] * n
    
    maxlen, maxcnt = 1, 0        
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if dp[i] < dp[j] + 1:
                    dp[i] = dp[j] + 1
                    cnt_dp[i] = cnt_dp[j]
                elif dp[i] == dp[j] + 1:
                    cnt_dp[i] += cnt_dp[j]
                    
        if dp[i] == maxlen:
            maxcnt += cnt_dp[i]
        elif dp[i] > maxlen:
            maxlen = dp[i]
            maxcnt = cnt_dp[i]

    return maxcnt
```

## [Dice Roll Simulation](https://leetcode.com/problems/dice-roll-simulation/)
*Return the number of distinct sequences that can be obtained with exact n rolls.*


```python
def dieSimulator(self, n: int, rollMax: List[int]) -> int:
    """
    dp[i][j]: the number of distinct sequences with i rolls ending with j

rollMax = [ 1, 1, 1, 2, 2, 3], n = 3  
    i\j  0  1  2  3  4  5 total
        0  0  0  0  0  0  0     1  
        1  1  1  1  1  1  1     6 
        2  5  5  6  6  6  6    34
        3 29 29 29 
    dp[3][0] = 34 - (sum_dp[1] - dp[1][0]) = 29 _y00
    dp[3][1] = 34 - (sum_dp[1] - dp[1][1]) = 29 _y11
    dp[3][2] = 34 - (sum_dp[1] - dp[1][1]) = 29 _y22
    dp[3][3] = 34 - (sum_dp[0] - dp[0][3]) = 30 y333
    dp[3][4] = 34 - (sum_dp[0] - dp[0][4]) = 30 y444
    dp[3][5] = 34                               5555

    
    """
    MOD = 10**9 + 7
    dp =[[0] * 6 for _ in range(n + 1)]
    sum_dp = [0] * (n + 1)
    
    sum_dp[0] = 1
    for i in range(6):
        if rollMax[i] > 0:
            dp[1][i] = 1
            sum_dp[1] += dp[1][i]

    for i in range(2, n + 1):
        for j in range(6):
            dp[i][j] = sum_dp[i-1]
            k = i - rollMax[j] - 1
            if k >= 0:
                dp[i][j] -= (sum_dp[k] - dp[k][j])
            
            sum_dp[i] = (sum_dp[i] + dp[i][j]) % MOD

            
    return sum_dp[-1]
```

# Pattern 2.2 dp = { initial_state : 1 }, curr_dp = collections.defaultdict(int)

## [Target Sum](https://leetcode.com/problems/target-sum/)
*Return the number of different expressions that you can build, which evaluates to target.*


```python
def findTargetSumWays(self, nums: List[int], S: int) -> int:
    
    # edge case
    cumsum = 0
    for num in nums:
        cumsum += abs(num)
    if cumsum < S or -cumsum > S:
        return 0
    
    dp = {0:1}
    for num in nums:
        curr_dp = collections.defaultdict(int)
        for key, val in dp.items():
            curr_dp[key + num] += val
            curr_dp[key - num] += val
        dp = curr_dp
    return dp[S]
```

## [Knight Probability in Chessboard](https://leetcode.com/problems/knight-probability-in-chessboard/)
*Return the probability that the knight remains on the board after it has stopped moving.*


```python
def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
    """
    (i, j)
    
    (i-2, j-1)
    (i-2, j+1)
    (i-1, j-2)
    (i-1, j+2)
    (i+1, j-2)
    (i+1, j+2)
    (i+2, j-1)
    (i+2, j+1)
    """
    
    h = {(row, column):1}
    
    for _ in range(k):
        curr_h = collections.defaultdict(int)
        for (i, j), val in h.items():
            
            candidates = [
                (i-2, j-1),
                (i-2, j+1),
                (i-1, j-2),
                (i-1, j+2),
                (i+1, j-2),
                (i+1, j+2),
                (i+2, j-1),
                (i+2, j+1)]
            
            for r, c in candidates:
                if 0 <= r < n and 0<= c < n:
                    curr_h[(r, c)] += val
        h = curr_h

    return sum(h.values())/8**k
```

## [Out of Boundary Paths](https://leetcode.com/problems/out-of-boundary-paths/)
*Return the number of paths to move the ball out of the grid boundary.*





```python
def findPaths(self, m: int, n: int, N: int, i: int, j: int) -> int:
    """
    m = 2, n = 2, N = 2, i = 0, j = 0
    
    (i, j):
    
    (i, j + 1)
    (i, j - 1)
    (i + 1, j)
    (i - 1, j)
    
    """
    MOD = 10**9 + 7
    
    h = {(i, j):1}
    cnt = 0
    for _ in range(N):
        curr_h = collections.defaultdict(int)
        for (i, j), val in h.items():
            candidates = [
                (i, j + 1),
                (i, j - 1),
                (i + 1, j),
                (i - 1, j),
            ]
            for r, c in candidates:
                if 0 <= r < m and 0 <= c < n:
                    curr_h[(r, c)] = (curr_h[(r, c)] + val) % MOD
                else:
                    cnt = (cnt + val) % MOD
        h = curr_h
    return cnt
```

## [Number of Ways to Stay in the Same Place After Some Steps](https://leetcode.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/)
*Return the number of ways such that your pointer still at index 0 after exactly steps steps.*


```python
def numWays(self, steps: int, arrLen: int) -> int:
    """
    arrLen:2
    0 1
    
    """
    MOD = 10**9 + 7
    dp = {0:1}
    
    for _ in range(steps):
        curr_dp = collections.defaultdict(int)
        for i, val in dp.items():
            candidates = [i, i-1, i+1]
            for j in candidates:
                if 0 <= j < arrLen:
                    curr_dp[j] = (curr_dp[j] + val) % MOD
        dp = curr_dp
    return dp[0] 
```

# Pattern 2.3 dp = [1, 1, ..., 1], curr_dp = [0, 0, ..., 0 ]

## [Knight Dialer](https://leetcode.com/problems/knight-dialer/)
*Return how many distinct phone numbers of length n we can dial.*


```python
def knightDialer(self, n: int) -> int:
    
    MOD = 10**9 + 7

    validPath = {
        0: [4, 6],
        1: [6, 8],
        2: [7, 9],
        3: [4, 8],
        4: [0, 3, 9],
        5: [],
        6: [0, 1, 7],
        7: [2, 6],
        8: [1, 3],
        9: [2, 4]
    }
    
    h = [1] * 10
    
    for i in range(n-1):
        curr_h = [0] * 10
        for num, val in enumerate(h):
            for next_num in validPath[num]:
                curr_h[next_num] = (curr_h[next_num] + val) % MOD        
        h = curr_h 
    return sum(h) % MOD
```

## [Count Vowels Permutation](https://leetcode.com/problems/count-vowels-permutation/)
*Count how many strings of length n can be formed under the following rules*


```python
def countVowelPermutation(self, n: int) -> int:
    """
    a, e, i, o, u
    0  1  2  3  4
    """
    
    MOD = 10**9 + 7
    rules = [[1], #a
            [0, 2], #e 
            [0, 1, 3, 4], #i
            [2, 4], #o
            [0]] #u     
    
    dp = [1] * 5 # {'a':1, 'e':1, 'i':1, 'o':1, 'u':1}
    
    for _ in range(n-1):
        curr_dp = [0] * 5
        for i, val in enumerate(dp):
            for j in rules[i]:
                curr_dp[j] = (curr_dp[j] + val) % MOD
        dp = curr_dp

    return sum(dp) % MOD
```

## [Domino and Tromino Tiling](https://leetcode.com/problems/domino-and-tromino-tiling/)
*How many ways are there to tile a 2 x n board?*


```python
def numTilings(self, n: int) -> int:
    """
curr_state  next_state 
(0) x   1 x(0)  1 1(2)  1 x(1)  1 1(3) 
    x   1 x     1 x     1 1     2 2
    
(1) x   1 1(2)  1 1(3) 
    o   o x     o 1   
    
(2) o   o x(1)  o 1(3)
    x   1 1     1 1
    
(3) o   o x(0)
    o   o x
    
    dp[i]: the number of ways to fill tile 2 x i board
    
    """
    MOD = 10**9 + 7
    path = [[0, 1, 2, 3], [2, 3], [1, 3], [0]]
    dp = [1, 0, 0, 0]
    
    for i in range(n):
        curr_dp = [0, 0, 0, 0]
        for curr_state, val in enumerate(dp):
            if val > 0:
                for next_state in path[curr_state]:
                    curr_dp[next_state] = (curr_dp[next_state] + val) % MOD
        dp = curr_dp
    return dp[0]
```

[<-PREV](dsa.md)
