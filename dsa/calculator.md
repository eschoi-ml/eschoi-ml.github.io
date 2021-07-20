# Calculator

- Basic Calculator: Given a string s which represents an expression, evaluate this expression and return its value. 
    - [Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/): +, - , *, /, ' '
    - [Basic Calculator III](https://leetcode.com/problems/basic-calculator-iii/): +, - , *, /, ' ', (, )
    - [Basic Calculator](https://leetcode.com/problems/basic-calculator/): +, - , ' ', (, )
- [Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)
- [Expression Add Operators](https://leetcode.com/problems/expression-add-operators/)
- [Different Ways to Add Parentheses](https://leetcode.com/problems/different-ways-to-add-parentheses/)

## [Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/)


```python
def calculate(self, s: str) -> int:
    """
    operation = +, -, *, /, ' '
    """
    # Solution 1. Use a stack
    # O(n) & O(n)
    
    stack = []
    
    num = 0
    op = '+'
    
    for c in s + '+':
        
        if c.isdigit():
            num = 10 * num + int(c)
        elif c == ' ':
            pass
        else:
            if op == '+':
                stack.append(num)                
            elif op == '-':
                stack.append(-num)
            elif op == '*':
                stack[-1] *= num
            else:# op == '/'
                sign = 1 if stack[-1] >= 0 else -1
                stack[-1] = sign * (abs(stack[-1])//num) 
            num = 0
            op = c
    
    return sum(stack)
    
    # Solution 2. O(n) & O(1)
    stack = num = res = 0
    op = '+'

    for c in s + '+':
            
        if c.isdigit():
            num = 10 * num + int(c)
        elif c == ' ':
            pass
        else:
            if op == '+':
                res += stack
                stack = num
            elif op == '-':
                res += stack
                stack = -num
            elif op == '*':
                stack *= num
            else: # op == '/'
                sign = 1 if stack >= 0 else -1
                stack = sign * (abs(stack)//num)
            num = 0
            op = c
    
    return res + stack
```

## [Basic Calculator III](https://leetcode.com/problems/basic-calculator-iii/)


```python
def calculate(self, s: str) -> int:
            
    """
    op = +, -, *, /, (, )
    
    """
    # Solution 1. Iterative + stack
    # O(n) & O(n)
    
    def update(op, num):
        if op == '+':
            stack.append(num)
        elif op == '-':
            stack.append(-num)
        elif op == '*':
            stack[-1] *= num
        elif op == '/':
            stack[-1] = int(stack[-1]/num)
    
    stack = []
    num = 0
    op = '+'
    for c in s + '+':
        
        if c.isdigit():
            num = num * 10 + int(c)
        elif c == ' ':
            pass
        elif c == '(':
            stack.append(op) # only operator for stack push
            num = 0
            op = '+'
        else: # c == '+, -, *, /, )':
            update(op, num)
            if c == ')':
                num = 0
                while isinstance(stack[-1], int):
                    num += stack.pop()
                op = stack.pop()
                update(op, num)
            num = 0
            op = c

    return sum(stack)
    
    
    # Solution 2. Recursive + stack
    
    def helper(s, start):
        
        stack = []
        num = 0
        op = '+'
        
        i = start
        while i < len(s):
            c = s[i]
            if c.isdigit():
                num = num * 10 + int(c)
            elif c == ' ':
                pass
            elif c == '(':
                num, i = helper(s, i + 1)
            else:
                if op == '+':
                    stack.append(num)
                elif op == '-':
                    stack.append(-num)                        
                elif op == '*':
                    stack[-1] *= num
                elif op == '/':
                    stack[-1] = int(stack[-1] / num)
                num = 0
                op = c
                
                if c == ')':
                    return sum(stack), i
            i += 1
        return sum(stack), i
    
    
    return helper(s + '+', 0)[0]
    
    # Solution 3. Recursive + no stack
    
    def helper(s, start):
        
        stack = num = res = 0
        op = '+'
        
        i = start
        while i < len(s):
            c = s[i]
            if c.isdigit():
                num = num * 10 + int(c)
            elif c == ' ':
                pass
            elif c == '(':
                num, i = helper(s, i + 1)
            else:
                if op == '+':
                    res += stack
                    stack = num
                elif op == '-':
                    res += stack
                    stack = -num                   
                elif op == '*':
                    stack *= num
                elif op == '/':
                    stack = int(stack / num)
                num = 0
                op = c
                
                if c == ')':
                    return res + stack, i
            i += 1
        return res + stack, i
    
    
    return helper(s + '+', 0)[0]
```

## [Basic Calculator](https://leetcode.com/problems/basic-calculator/)


```python
def calculate(self, s: str) -> int:
    """
    operation = +, -, (, ), ' '
    
    """
    # Solution 1. Stack + iterative
    # O(n) & O(n)
    
    def update(op, num):
        
        if op == '+':
            stack.append(num)
        elif op == '-':
            stack.append(-num)

    stack = []
    
    num = 0
    op = '+'

    for c in s + '+':
                
        if c.isdigit():
            num = num * 10 + int(c)
        elif c == ' ':
            pass
        elif c == '(':
            stack.append(op)
            num = 0
            op = '+'
        else:
            update(op, num)
            if c == ')':
                num = 0
                while isinstance(stack[-1], int):
                    num += stack.pop()
                op = stack.pop()
                update(op, num)
            num = 0
            op = c
            
    return sum(stack)          
    
    # Solution 2. recursive + no stack
    def helper(s, start):
        
        stack = num = res = 0
        op = '+'
        
        i = start
        while i < len(s):
            c = s[i]
            if c.isdigit():
                num = 10 * num + int(c)
            elif c == ' ':
                pass
            elif c == '(':
                num, i = helper(s, i+1)
            else:
                if op == '+':
                    res += stack
                    stack = num
                elif op == '-':
                    res += stack
                    stack = -num
                
                num = 0
                op = c
                
                if c == ')':
                    return res + stack, i
            i += 1
        return res + stack, i

    return helper(s + '+', 0)[0]
```

## [Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)



```python
def evalRPN(self, tokens: List[str]) -> int:
    
    # O(n) & O(n)
    stack = []
    for token in tokens:
        if token not in {'+', '-', '*', '/'}:
            stack.append(int(token))
        else:
            n2 = stack.pop()
            n1 = stack.pop()
            if token == '+':
                stack.append(n1 + n2)
            elif token == '-':
                stack.append(n1 - n2)
            elif token == '*':
                stack.append(n1 * n2)
            else:
                stack.append(int(n1/n2))
    
    return stack.pop()
```

## [Expression Add Operators](https://leetcode.com/problems/expression-add-operators/)


```python
def addOperators(self, num: str, target: int) -> List[str]:
    
    # Solution 1. Use a stack 
    def helper(level=0, curr="", stack=[]):
        
        if level == n:
            if sum(stack) == target:
                res.append(curr)
            return 
        
        for i in range(level, n):
            
            curr_num = num[level:i+1]
            int_curr_num = int(curr_num)
            
            
            if num[level]=="0" and i!=level:
                break
            
            if level == 0:
                # +
                helper(i+1, curr + curr_num, stack + [int_curr_num]) 
            
            if level > 0:
                
                # +
                helper(i+1, curr + '+' + curr_num, stack + [int_curr_num]) 
                
                # -
                helper(i+1, curr + '-' + curr_num, stack + [-int_curr_num])
            
                # *
                temp = stack[-1]
                stack[-1] *= int_curr_num
                helper(i+1, curr + '*' + curr_num, stack)
                stack[-1] = temp                    
            
    n = len(num)
    res = []
    
    helper()
    
    return res
    
    # Solution 2. No stack
    def helper(level=0, curr="", stack=0, res=0):
        
        if level == n:
            if res + stack == target:
                ans.append(curr)
            return 
        
        for i in range(level, n):
            
            curr_num = num[level:i+1]
            int_curr_num = int(curr_num)
            
            
            if num[level]=="0" and i!=level:
                break
            
            if level == 0:
                # +
                helper(i+1, curr + curr_num, int_curr_num, res + stack) 
            
            if level > 0:
                
                # +
                helper(i+1, curr + '+' + curr_num, int_curr_num, res + stack) 
                
                # -
                helper(i+1, curr + '-' + curr_num, -int_curr_num, res + stack)
            
                # *
                helper(i+1, curr + '*' + curr_num, stack * int_curr_num, res)                 
            
    n = len(num)
    ans = []
    
    helper()
    
    return ans
```

## [Different Ways to Add Parentheses](https://leetcode.com/problems/different-ways-to-add-parentheses/)


```python
def diffWaysToCompute(self, expression: str) -> List[int]:
    
    # Solution 1. Recursion
    def helper(s):
        
        if s.isdigit():
            return [int(s)]
        
        res = []
        for i, c in enumerate(s):
            if c in {'+', '-', '*'}:
                l = helper(s[:i]) 
                r = helper(s[i+1:])
                for n1 in l:
                    for n2 in r:  
                        if c == '+':
                            res.append(n1 + n2)                            
                        elif c == '-':
                            res.append(n1 - n2)
                        else:
                            res.append(n1 * n2)
        return res
        
    return helper(expression)
    
    
    # Solution 2. Recursion + memoization 
    def helper(s):
        
        if s in memo:
            return memo[s]
        
        if s.isdigit():
            memo[s] = [int(s)]
            return memo[s]
        
        res = []
        for i, c in enumerate(s):
            if c in {'+', '-', '*'}:
                l = helper(s[:i]) 
                r = helper(s[i+1:])
                for n1 in l:
                    for n2 in r:  
                        if c == '+':
                            res.append(n1 + n2)                            
                        elif c == '-':
                            res.append(n1 - n2)
                        else:
                            res.append(n1 * n2)
        memo[s] = res
        return memo[s]
        
    memo = {}
    return helper(expression)
```
