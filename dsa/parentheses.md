[<-PREV](string.md)

# Parentheses
- [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)
- [Minimum Remove to Make Valid Parentheses](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/)
- [Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/)
- [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)
- [Different Ways to Add Parentheses](https://leetcode.com/problems/different-ways-to-add-parentheses/)
- [Remove Invalid Parentheses](https://leetcode.com/problems/remove-invalid-parentheses/)
- [Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/)

## [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)
*Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.*


```python
class Solution:
    def isValid(self, s: str) -> bool:
        """
        pair = {')':'(', }
        
        stack = [(]
        if open bracket:
            push into the stack
        if closed bracket:
            pop and compare
        """
        # O(n), O(n)
        
        # edge
        if len(s) == 0:
            return True
        if len(s) % 2 == 1:
            return False
        
        pairs = {'(':')', '[':']', '{':'}'}
        stack = []
        
        for bracket in s:
            if bracket in pairs:
                stack.append(bracket)
            elif not stack or pairs[stack.pop()] != bracket:
                return False
        return not stack
```

## [Minimum Remove to Make Valid Parentheses](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/)
*Given a string s of '(' , ')' and lowercase English characters. Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) so that the resulting parentheses string is valid and return any valid string.*


```python
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        
        # O(n), O(n) where n is the length of s
        stack = []
        s = list(s)
        
        for i, c in enumerate(s):

            if c == '(':
                stack.append(i)
            elif c == ')':
                if stack:
                    stack.pop()
                else:
                    s[i] = ''

        while stack:
            s.pop(stack.pop())
            
        return "".join(s)
```

## [Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/)
*Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.*


```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        # Solution 1. 
        # O(n), O(n)
        maxlen = 0
        stack = [-1]
        for i, c in enumerate(s):
            if c == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    maxlen = max(maxlen, i - stack[-1])
        
        return maxlen
    
        # Solution2
        maxlen = 0
        left = right = 0
        for c in s:
            if c == '(':
                left += 1
            else:
                right += 1
                
            if left == right:
                maxlen = max(maxlen, left * 2)
            elif right > left:
                left = right = 0
        
        left = right = 0
        for i in range(len(s)-1, -1, -1):
            c = s[i]
            if c == '(':
                left += 1
            else:
                right += 1
            
            if left == right:
                maxlen = max(maxlen, right * 2)
            elif left > right:
                left = right = 0
        return maxlen
```

## [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)
*Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.*



```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        """
        
        if left < n: # ( 
             
        if left > right: # ')'
            
        """
        # O(), O()
        def helper(left, right, curr=""):
            
            if len(curr) == 2 * n:
                res.append(curr)
                return                 
            
            if left < n:
                helper(left + 1, right, curr + '(')
            
            if left > right:
                helper(left, right + 1, curr + ')')

        res = []
        helper(0, 0, "")
        return res
```

## [Different Ways to Add Parentheses](https://leetcode.com/problems/different-ways-to-add-parentheses/)
*Given a string expression of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. You may return the answer in any order.*


```python
class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        
        # Solution 1
        if expression.isdigit():
            return [int(expression)]
        
        res = []
        for i, c in enumerate(expression):
            if c in {'+', '-', '*'}:
                ls = self.diffWaysToCompute(expression[:i])
                rs = self.diffWaysToCompute(expression[i+1:])
                for l in ls:
                    for r in rs:                                        
                        if c == '+':
                            res.append(l + r)
                        elif c == '-':
                            res.append(l - r)
                        else: # c == '*'
                            res.append(l * r)      
        
        return res
        
        # Solution 2
        def helper(s):
            
            if s in memo:
                return memo[s]
            
            if s.isdigit():
                return [int(s)]
            
            res = []
            for i, c in enumerate(s):
                if c in {'+', '-', '*'}:
                    ls = helper(s[:i])
                    rs = helper(s[i+1:])
                    
                    for l in ls:
                        for r in rs:
                            if c == '+':
                                res.append(l + r)
                            elif c == '-':
                                res.append(l - r)
                            else: # c == '*'
                                res.append(l * r)
            memo[s] = res
            return res
                    
        memo = {}
        return helper(expression)

```

## [Remove Invalid Parentheses](https://leetcode.com/problems/remove-invalid-parentheses/)
*Given a string s that contains parentheses and letters, remove the minimum number of invalid parentheses to make the input string valid. Return all the possible results.*


```python
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        
        def isValid(st):
            balance = 0
            for c in st:
                if c == '(':
                    balance += 1
                elif c == ')':
                    balance -= 1
                if balance < 0:
                    return False
            return balance == 0
            
        q = collections.deque()
        q.append(s)
        
        res = []
        
        while q:
            size = len(q)
            visited = set()
            for _ in range(size):
                currs = q.popleft() 
                if isValid(currs):
                    res.append(currs)
                elif not res:
                    for i in range(len(currs)):
                        if currs[i] in {'(',')'}:
                            nexts = currs[:i] + currs[i+1:]
                            if nexts not in visited:
                                q.append(nexts)
                                visited.add(nexts)
        return res
```

## [Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/)
*Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.*


```python
class Solution:
    def checkValidString(self, s: str) -> bool:
        
        cmin = cmax = 0
        
        for c in s:
            
            if c == '(':
                cmin += 1
                cmax += 1
            
            elif c == ')':
                cmin = max(0, cmin - 1)
                cmax -= 1
                
                if cmax < 0:
                    return False
                    
            else: # c == '*'
                cmax += 1
                cmin = max(0, cmin - 1)

        return cmin == 0 
```

[<-PREV](string.md)
