# Dynamic Programming
# Pattern 5. State machine 

**Maximum Subarray**
> dp[i][j]: the largest sum/product/... using nums[i] when j = 0 min state and j = 1 max state

- Maximum Subarray
- Maximum Product Subarray
- Maximum Absolute Sum of Any Subarray
- Maximum Subarray Sum After One Operation
- Longest Turbulent Subarray

**House Robber**
> dp[i]: dp[i][j]: the maximum amount of money you can rob on ith day when j = 0 no rob or j = 1 rob

- House Robber
- House Robber II
- House Robber III

**Best Time to Buy and Sell Stock**
> dp[i][j]: the maximum profit you can achieve upto ith transaction when j = 0 sell state or j = 1 buy state

- Best Time to Buy and Sell Stock (k = 1)
- Best Time to Buy and Sell Stock with Cooldown (k = 1 with cooldown)
- Best Time to Buy and Sell Stock II (k = inf)
- Best Time to Buy and Sell Stock with Transaction Fee (k = inf with transaction fee)
- Best Time to Buy and Sell Stock IV (arbitrary k)
- Best Time to Buy and Sell Stock III (k = 2)

## Maximum Subarray

### [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)
*Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.*


```python
def maxSubArray(self, nums: List[int]) -> int:
    """
    dp[i]: the largest sum using nums[i]
    
    """
    # Solution 1
    n = len(nums)
    dp = [0] * n        
    dp[0] = nums[0]
    max_sum = nums[0]
    for i in range(1, n):
        num = nums[i]
        dp[i] = max(num, dp[i-1] + num)
        max_sum = max(max_sum, dp[i])
    return max_sum 

    # Space-optimized Solution
    n = len(nums)
    curr_sum = nums[0]
    max_sum = curr_sum
    
    for i in range(1, n):
        
        num = nums[i]
        
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    
    return max_sum
```

### [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)
*Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product, and return the product.*


```python
def maxProduct(self, nums: List[int]) -> int:
    """
    dp[i][j]: the largest product using nums[i]
              j = 0 min, j = 1 max
    """
    # Solution 1
    n = len(nums)
    dp = [[0, 0] for _ in range(n)]
    dp[0][0] = dp[0][1] = nums[0]
    max_prod = nums[0]
    
    for i in range(1, n):
        num = nums[i]
        dp[i][0] = min(num, dp[i-1][0] * num, dp[i-1][1] * num)
        dp[i][1] = max(num, dp[i-1][0] * num, dp[i-1][1] * num)
        max_prod = max(max_prod, dp[i][1])
    
    return max_prod

    # Space-optimized solution
    Max = Min = nums[0]
    max_prod = Max
    
    for i in range(1, len(nums)):
        
        num = nums[i]
        
        curr_max = max(num, Max * num, Min * num)
        curr_min = min(num, Max * num, Min * num)
        
        Max = curr_max
        Min = curr_min
        max_prod = max(max_prod, Max)
    
    return max_prod
```

### [Maximum Absolute Sum of Any Subarray](https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/)
*Return the maximum absolute sum of any (possibly empty) subarray of nums.*


```python
def maxAbsoluteSum(self, nums: List[int]) -> int:
    """
    dp[i][j]: the maximum sum using nums[i]
              j = 0 min, j = 1 max
    """
    # Solution 1
    n = len(nums)
    dp = [[0, 0] for _ in range(n)]
    dp[0][0] = dp[0][1] = nums[0]
    max_abs_sum = abs(nums[0])
    
    for i in range(1, n):
        num = nums[i]
        dp[i][0] = min(num, dp[i-1][0] + num, dp[i-1][1] + num)
        dp[i][1] = max(num, dp[i-1][0] + num, dp[i-1][1] + num)
        max_abs_sum = max(max_abs_sum, abs(dp[i][0]), abs(dp[i][1]))
    
    return max_abs_sum
    
    # Space-optimized solution
    Max = Min = nums[0]
    max_abs_sum = abs(Max)
    
    for i in range(1, len(nums)):
        
        num = nums[i]
        
        curr_max = max(num, Max + num, Min + num)
        curr_min = min(num, Max + num, Min + num)
        
        Max = curr_max
        Min = curr_min
        max_abs_sum = max(max_abs_sum, abs(Max), abs(Min))
        
    return max_abs_sum
```

### [ Maximum Subarray Sum After One Operation](https://leetcode.com/problems/maximum-subarray-sum-after-one-operation/)
*Return the maximum possible subarray sum after exactly one operation.*


```python
def maxSumAfterOperation(self, nums: List[int]) -> int:
    """
    dp[i][j]: the maximum possible subarray sum using nums[i] 
              when j = 0 nums[i] or j = 1 squared nums[i]
                
    """
    # Solution 1 
    n = len(nums)
    dp = [[0, 0] for _ in range(n)]
    dp[0][0] = nums[0]
    dp[0][1] = nums[0]*nums[0]
    max_sum = dp[0][1]
    
    for i in range(1, n):
        num = nums[i]
        squared_num = num * num
        dp[i][0] = max(num, dp[i-1][0] + num)
        dp[i][1] = max(squared_num, dp[i-1][0] + squared_num, dp[i-1][1] + num)
        max_sum = max(max_sum, dp[i][1])
    
    return max_sum
    
    # Space-optimized solution
    n = len(nums)
    non_squared, squared = nums[0], nums[0]*nums[0]
    max_sum = squared
    
    for i in range(1, n):
        
        num = nums[i]
        num_squared = num * num
        
        curr_non_squared = max(num, non_squared + num)
        curr_squared = max(num_squared, non_squared + num_squared, squared + num)
        
        non_squared = curr_non_squared
        squared = curr_squared
        
        max_sum = max(max_sum, squared)
    
    return max_sum
```

### [Longest Turbulent Subarray](https://leetcode.com/problems/longest-turbulent-subarray/)
*Given an integer array arr, return the length of a maximum size turbulent subarray of arr.*


```python
def maxTurbulenceSize(self, arr: List[int]) -> int:
    """
    dp[i][j]: the length of a maximum size turbulent subarray using arr[i]
              when j = 0 arr[i] is smaller, j = 1 arr[i] is greater
    """
    # Solution 1
    
    n = len(arr)
    dp = [[0, 0] for _ in range(n)]
    dp[0][0] = dp[0][1] = 1
    max_len = 1
    for i in range(1, n):
        if arr[i-1] > arr[i]:
            dp[i][0] = dp[i-1][1] + 1
            dp[i][1] = 1
            
        
        elif arr[i-1] < arr[i]:
            dp[i][0] = 1
            dp[i][1] = dp[i-1][0] + 1
            
        
        else: # arr[i] == arr[i-1]
            dp[i][0] = 1
            dp[i][1] = 1
        
        max_len = max(max_len, dp[i][0], dp[i][1])
    return max_len


    # Space-optimized solution
    n = len(arr)
    low, high = 1, 1
    max_len = 1
    
    for i in range(1, n):
        
        if arr[i-1] > arr[i]:
            curr_low = high + 1
            curr_high = 1
        elif arr[i-1] < arr[i]:
            curr_low = 1
            curr_high = low + 1
        else:
            curr_low = 1
            curr_high = 1
        
        low = curr_low
        high = curr_high
        max_len = max(max_len, low, high)

    return max_len
```

## House Robber

### [House Robber](https://leetcode.com/problems/house-robber/)
*Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.*


```python
def rob(self, nums: List[int]) -> int:
    """
    dp[i][j]: the maximum amount of money you can rob on ith day 
              when j = 0 no rob or j = 1 rob
    
    dp[i][0] = max(dp[i-1])
    dp[i][1] = dp[i-1][0] + nums[i-1]
    
    """
    # Solution 1
    n = len(nums)        
    dp = [[0, 0] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1])
        dp[i][1] = dp[i-1][0] + nums[i-1]
    
    return max(dp[-1])

    # Solution 2
    n = len(nums)
    dp = [0] * (n + 1)
    dp[1] = nums[0]
    
    for i in range(2, n + 1):
        dp[i] = max(dp[i-2] + nums[i-1], dp[i-1])
    
    return dp[-1]

    # Solution 3
    n = len(nums)
    prev2 = 0
    prev1 = nums[0]
    
    for i in range(2, n + 1):
        temp = prev1
        prev1 = max(prev2 + nums[i-1], prev1)
        prev2 = temp
    return prev1
```

### [House Robber II](https://leetcode.com/problems/house-robber-ii/)
*Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police when all houses at this place are arranged in a circle.*


```python
def rob(self, nums: List[int]) -> int:
    def rob_helper(nums, l, r):
        
        n = r - l + 1
        dp = [0] * (n + 1)
        dp[1] = nums[l]
        
        for i in range(2, n + 1):
            dp[i] = max(dp[i-2] + nums[l+i-1], dp[i-1])
        
        return dp[-1]
    
    n = len(nums)
    if n == 1:
        return nums[0]
    if n == 2:
        return max(nums)
    return max(rob_helper(nums, 0, n-2), rob_helper(nums, 1, n-1))
```

### [House Robber III](https://leetcode.com/problems/house-robber-iii/)
*Given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police.*


```python
def rob(self, root: TreeNode) -> int:
    
    def helper(node):
        
        if not node:
            return [0, 0]
        
        left = helper(node.left)
        right = helper(node.right)
        
        state0 = max(left) + max(right)
        state1 = node.val + left[0] + right[0]
        
        return [state0, state1]
    
    return max(helper(root))
```

## Best Time to Buy and Sell Stock

### [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) (k = 1)
*Return the maximum profit you can achieve from this transaction.*


```python
def maxProfit(self, prices: List[int]) -> int:
    """
    dp[i][j]: the maximum profit you can achieve upto ith transaction
              when j = 0 sell state or j = 1 buy state
    
    """
    # Solution 1
    n = len(prices)
    dp = [[0, 0] for _ in range(n)]
    dp[0][1] = -prices[0]
    for i in range(1, n):
        price = prices[i]
        dp[i][0] = max(dp[i-1][0], price + dp[i-1][1])
        dp[i][1] = max(dp[i-1][1], -price)
    
    return dp[-1][0]

    # Space-optimized solution
    n = len(prices)
    sell = 0
    buy = -prices[0]
    for i in range(1, n):
        price = prices[i]
        sell = max(sell, price + buy)
        buy = max(buy, -price)
    
    return sell
```

### [Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) (k = 1 with cooldown)
*Find the maximum profit you can achieve. You may complete as many transactions as you like with the following restrictions:
After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).*


```python
def maxProfit(self, prices: List[int]) -> int:
    """
    dp[i][j]: the maximum profit you can achieve upto ith transaction
              when j = 0 sell state     <- j = 1
                   j = 1 buy state      <- j = 1, 2
                   j = 2 cooldown state <- j = 0, 2 
    """
    # Solution 1
    n = len(prices)  
    dp = [[0, 0, 0] for _ in range(n)]
    dp[0][1] = -prices[0]
    
    for i in range(1, n):
        price = prices[i]
        dp[i][0] = dp[i-1][1] + price
        dp[i][1] = max(dp[i-1][1], dp[i-1][2] - price)
        dp[i][2] = max(dp[i-1][2], dp[i-1][0])
    
    return max(dp[-1][0], dp[-1][2])

    
    # Space-optimized solution
    n = len(prices)
    sell = 0
    buy = -prices[0]
    cooldown = 0
    
    for i in range(1, n):
        price = prices[i]
        curr_sell = buy + price
        curr_buy = max(buy, cooldown - price)
        curr_cooldown = max(cooldown, sell)
        
        sell = curr_sell
        buy = curr_buy
        cooldown = curr_cooldown
    
    return max(sell, cooldown) 
```

### [Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/) (k = inf)
*Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).*


```python
def maxProfit(self, prices: List[int]) -> int:
    """
    dp[i][j]: the maximum profit you can achieve upto ith transaction
              when j = 0 sell state, j = 1 buy state        
    """
    # Solution 1
    n = len(prices)
    dp = [[0, 0] for _ in range(n)]
    dp[0][1] = - prices[0]
    
    for i in range(1, n):
        price = prices[i]
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + price)
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] - price)
    return dp[-1][0]
    
    # Space-optimized solution
    n = len(prices)
    sell = 0
    buy = -prices[0]
    for i in range(1, n):
        price = prices[i]
        curr_sell = max(sell, buy + price)
        curr_buy = max(buy, sell - price)
        
        sell = curr_sell
        buy = curr_buy
    
    return sell
    
    # Intuitive solution
    n = len(prices)
    res = 0
    for i in range(1, n):
        diff = prices[i] - prices[i-1]
        if diff > 0:
            res += diff
    return res
```

### [Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/) (k = inf with transaction fee)
*Find the maximum profit you can achieve. You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction.*


```python
def maxProfit(self, prices: List[int], fee: int) -> int:
    """
    dp[i][j]: the maximum profit you can achieve upto ith transaction
              when j = 0 sell state(+ transaction) or j = 1 buy state 
    """
    # Solution 1
    n = len(prices)
    dp = [[0, 0] for i in range(n)]
    dp[0][1] = -prices[0]
    
    for i in range(1, n):
        price = prices[i]
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + price - fee)
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] - price)
    return dp[-1][0]

    # Space-optimized solution
    n = len(prices)
    sell = 0
    buy = -prices[0]
    
    for i in range(1, n):
        price = prices[i]
        curr_sell = max(sell, buy + price - fee)
        curr_buy = max(buy, sell - price)
        
        sell = curr_sell
        buy = curr_buy
    return sell
```

### [Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/) (arbitrary k)
*Find the maximum profit you can achieve. You may complete at most k transactions.*


```python
def maxProfit(self, k: int, prices: List[int]) -> int:
    """
    dp[i][k][j]: the maximum profit you can achieve at kth transaction
                 when j = 0 sell state j = 1 buy state
                                   /*from the same transaction*/
    dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + price)
                                   /*from the previous transaction*/
    dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - price)
    """        
    n = len(prices)
    
    # edge case
    if not prices or n == 1 or k == 0: 
        return 0
    
    if n < 2 * k:
        res = 0
        for i in range(1, n):
            diff = prices[i] - prices[i-1]
            if diff > 0:
                res += diff
        return res
    
    # Solution 1
    dp = [[[0, 0] for _ in range(k + 1)] for _ in range(n)]
    
    for j in range(1, k + 1):
        dp[0][j][1] = -prices[0]
    
    for i in range(1, n):
        price = prices[i]
        for j in range(1, k + 1):
            dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + price)
            dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - price)
    return dp[-1][-1][0]
    
    # Space-optimized solution 
    n = len(prices)
    sell = [0] * (k + 1)
    buy = [0] * (k + 1)
    for i in range(1, k + 1):
        buy[i] = -prices[0]
    
    for i in range(1, n):
        price = prices[i]
        for j in range(1, k + 1):
            sell[j] = max(sell[j], buy[j] + price) 
            buy[j] = max(buy[j], sell[j-1] - price)
            
    return sell[-1]
```

### [Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/) (k = 2)
*Find the maximum profit you can achieve. You may complete at most two transactions.*


```python
def maxProfit(self, prices: List[int]) -> int:
    """
    dp[i][k][j]: the maximum profit you can achieve at kth transaction 
                    when j = 0 sell state and j = 1 buy state
    
    """
    # Solution 1
    k = 2
    n = len(prices)
    dp = [[[0, 0] for _ in range(k + 1)] for _ in range(n)]
    
    for i in range(1, k + 1):
        dp[0][i][1] = -prices[0]
    
    for i in range(1, n):
        price = prices[i]
        for j in range(1, k + 1):
            dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + price)
            dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - price)
    
    return dp[-1][-1][0]

    # Space-optimized solution
    k = 2
    n = len(prices)
    sell = [0] * (k + 1)
    buy = [0] * (k + 1)
    for i in range(1, k+1):
        buy[i] = -prices[0]
    
    for i in range(1, n):
        price = prices[i]
        for j in range(1, k + 1):
            sell[j] = max(sell[j], buy[j] + price)
            buy[j] = max(buy[j], sell[j-1] - price)
            
    return sell[-1]
```
