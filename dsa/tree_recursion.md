# Tree Recursion Top-down & Bottom-up


```python
def TopDown(node, params):

    if not node:
        return specific value if needed
    
    update the answer if needed

    left = TopDown(node.left, left_params)
    right = TopDown(node.right, right_params)

    return answer if needed
```


```python
def BottomUp(node):

    if not node:
        return specific value

    left = BottomUp(node.left)
    right = BottomUp(node.right)

    return answer
```


```python
def Advanced_BottomUp(node):

    if not node:
        return specific value

    left = Advanced_BottomUp(node.left)
    right = Advanced_BottomUp(node.right)

    update_left, update_right = 0, 0
    update_left = if needed
    update_right = if needed

    self.global_variable = update_function(self.global_variable, update_left + update_right(+ node.val))

    return return_one_subtree_function(update_left, update_right)(+ node.val)
```

## Basic application

### [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)


```python
def maxDepth(self, root: TreeNode) -> int:
    
    # Top down
    def helper(node, depth):
        
        if not node:
            return 
        
        if not node.left and not node.right:
            self.max_depth = max(self.max_depth, depth)
        
        helper(node.left, depth + 1)
        helper(node.right, depth + 1)
        
    
    self.max_depth = 0
    helper(root, 1)
    return self.max_depth
    
    # Bottom up
    if not root:
        return 0
    
    left = self.maxDepth(root.left)
    right = self.maxDepth(root.right)
    
    return max(left, right) + 1
```

### [111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)


```python
def minDepth(self, root: TreeNode) -> int:
    
    # Top down
    def helper(node, depth):
        
        if not node:
            return 
        
        if not node.left and not node.right:
            self.min_depth = min(self.min_depth, depth)
        
        helper(node.left, depth + 1)
        helper(node.right, depth + 1)
    
    if not root: return 0
    self.min_depth = float('inf')
    helper(root, 1)
    return self.min_depth
    

    # Bottom up
    if not root:
        return 0
    
    left = self.minDepth(root.left)
    right = self.minDepth(root.right)
    
    if root.left and root.right:
        return min(left, right) + 1
    return max(left, right) + 1
```

### [559. Maximum Depth of N-ary Tree](https://leetcode.com/problems/maximum-depth-of-n-ary-tree/)


```python
def maxDepth(self, root: 'Node') -> int:
    
    def helper(node, depth):
        if not node:
            return 
        
        if not node.children:
            self.max_depth = max(self.max_depth, depth)
        
        for child in node.children:
            helper(child, depth + 1)
    
    self.max_depth = 0
    helper(root, 1)
    return self.max_depth
```

### [129. Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/)


```python
def sumNumbers(self, root: TreeNode) -> int:
    
    def helper(node, num):
        
        if not node: 
            return 
        
        num = 10*num + node.val
        if not node.left and not node.right:
            self.sum_nums += num
        
        helper(node.left, num)
        helper(node.right, num)
    
    if not root:
        return 0
    self.sum_nums = 0
    helper(root, 0)
    return self.sum_nums
```

### [112. Path Sum](https://leetcode.com/problems/path-sum/)


```python
def hasPathSum(self, root: TreeNode, sum: int) -> bool:
    
    if not root: 
        return False
    
    sum -= root.val
    
    if not root.left and not root.right:
        return sum == 0
    
    left = self.hasPathSum(root.left, sum)
    right = self.hasPathSum(root.right, sum)
    
    return left or right
```

### [113. Path Sum II](https://leetcode.com/problems/path-sum-ii/)


```python
def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
    
    def helper(node, ts, path):
        
        if not node:
            return 
        
        ts -= node.val
        path.append(node.val)
        
        if not node.left and not node.right and ts == 0:
            res.append(path[:])
        
        helper(node.left, ts, path)
        helper(node.right, ts, path)
        
        path.pop()
                
    res = []
    helper(root, targetSum, [])
    return res
```

## Advanced application

### [124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)


```python
def maxPathSum(self, root: TreeNode) -> int:  
    
    def helper(node):
        
        if not node:
            return 0
        
        left = helper(node.left)
        right = helper(node.right)
        
        sum_left, sum_right = 0, 0
        if left > 0:
            sum_left = left
        if right > 0:
            sum_right = right
        
        self.max_sum = max(self.max_sum, sum_left + sum_right + node.val)
        
        return max(sum_left, sum_right) + node.val
        
    self.max_sum = float('-inf')
    helper(root)
    return self.max_sum
```

### [687. Longest Univalue Path](https://leetcode.com/problems/longest-univalue-path/)


```python
def longestUnivaluePath(self, root: TreeNode) -> int:
    
    def helper(node):
        """
        return maximum uni_depth either from left or right path            
        """
        if not node:
            return 0
        
        left = helper(node.left)
        right = helper(node.right)
        
        uni_left, uni_right = 0, 0
        if node.left and node.val == node.left.val:
            uni_left = left + 1
        if node.right and node.val == node.right.val:
            uni_right = right + 1
        
        self.max_path = max(self.max_path, uni_left + uni_right)
        
        return max(uni_left, uni_right) 
        
    self.max_path = 0
    helper(root)
    return self.max_path
```

### [250. Count Univalue Subtrees](https://leetcode.com/problems/count-univalue-subtrees/)


```python
def countUnivalSubtrees(self, root: TreeNode) -> int:
    
    def helper(node):
        
        if not node:
            return True
        
        left = helper(node.left)
        right = helper(node.right)
        
        if not left or not right:
            return False
        
        if node.left and node.left.val != node.val:
            return False
        if node.right and node.right.val != node.val:
            return False
        
        self.cnt += 1
        
        return True
        
    self.cnt = 0
    helper(root)
    return self.cnt
```

### [508. Most Frequent Subtree Sum](https://leetcode.com/problems/most-frequent-subtree-sum/)


```python
def findFrequentTreeSum(self, root: TreeNode) -> List[int]:
    
    def helper(node):
        
        if not node:
            return 0
        
        left = helper(node.left)
        right = helper(node.right)
        
        s_sum = left + right + node.val
        res[s_sum] += 1
        
        return s_sum

    res = collections.Counter()
    helper(root)
    
    max_val = max(res.values())
    max_res = []
    for key, val in res.items():
        if val == max_val:
            max_res.append(key)
    return max_res
```

### [572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)




```python
def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
    
    def helper(p, q):
        
        if not p and not q:
            return True
        
        if not p or not q:
            return False
        
        if p.val != q.val:
            return False
        
        return helper(p.left, q.left) and helper(p.right, q.right)
        
    if helper(root, subRoot):
        return True
    
    if not root:
        return False
    
    left = self.isSubtree(root.left, subRoot)
    right = self.isSubtree(root.right, subRoot)
    
    return left or right
```

### [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)


```python
def subarraySum(self, nums: List[int], k: int) -> int:
    """
    nums = [1, 2, 3]
    h = [0:1, 1:1, 3:1, 6:1]
    sum nums from i to j (k) = cumsum j - cumsum i
    cumsum i = cumsum j - k
    
    """

    h = collections.defaultdict(int)
    h[0] = 1
    
    cumsum = 0
    cnt = 0
    for num in nums:
        
        cumsum += num
        if cumsum - k in h:
            cnt += h[cumsum - k]
        h[cumsum] += 1
        
    return cnt
```

### [437. Path Sum III](https://leetcode.com/problems/path-sum-iii/)


```python
def pathSum(self, root: TreeNode, targetSum: int) -> int:
    def helper(node, cumsum):
        if not node: 
            return 
        
        cumsum += node.val
        if cumsum == targetSum:
            self.cnt += 1
        if cumsum - targetSum in h:
            self.cnt += h[cumsum - targetSum]
        h[cumsum] += 1
        helper(node.left, cumsum)
        helper(node.right, cumsum)
        h[cumsum] -= 1
    
    self.cnt = 0
    h = collections.defaultdict(int)
    helper(root, 0)
    return self.cnt
```
