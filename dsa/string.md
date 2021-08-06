[<-PREV](dsa.md)

# String
- [Calculator](#calculator)
- [Parentheses](#parentheses)
- [Justification](#justification)
- [Integer conversion](#integer-conversion)
- [Palindrome](#palindrome)
    
    

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

# Justification
- [Text Justification](https://leetcode.com/problems/text-justification/)
- [Rearrange Spaces Between Words](https://leetcode.com/problems/rearrange-spaces-between-words/)

## [Text Justification](https://leetcode.com/problems/text-justification/)
*Given an array of strings words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.*


```python
class Solution:
    def fullJustify(self, words, maxWidth):
        """
        full jutify
        - when not even, more on left
        
        left justify
        - one word
        - last line
        
        
        """

        # O(n), 
        n = len(words)
        res = []
        
        i = 0
        while i < n:
            
            j = i           # index
            wordlen = 0     # words length without space in between
            while j < n and wordlen + len(words[j]) + j - i <= maxWidth:
                wordlen += len(words[j])
                j += 1
            
            # left justify
            if j == i+1 or j == n:
                res.append(" ".join(words[i:j]) + " " * (maxWidth - wordlen - (j-1-i)))
 
            # full justify
            else:
                space = " " * ((maxWidth - wordlen) // (j - i - 1))
                tail = (maxWidth - wordlen) % (j - i - 1)            

                curr_res = ""
                while i < j-1:
                    curr_res += words[i] + space
                    if tail > 0:
                        curr_res += " "
                        tail -= 1
                    i += 1
                curr_res += words[i]
                res.append(curr_res)
            
            i = j
        
        return res
```

## [Rearrange Spaces Between Words](https://leetcode.com/problems/rearrange-spaces-between-words/)
*Return the string after rearranging the spaces.*


```python
class Solution:
    def reorderSpaces(self, text):
        
        n = len(text)
        word = text.split()
        nword = len(word)
        nspace = text.count(" ")
        
        space = 0 if nword == 1 else nspace // (nword-1)
        tailspace = nspace - space * (nword-1) 
        return (" " * space).join(word) + " "*tailspace
```

# Integer conversion
- [Integer to English Words](https://leetcode.com/problems/integer-to-english-words/)
- [Integer to Roman](https://leetcode.com/problems/integer-to-roman/)

## [Integer to English Words](https://leetcode.com/problems/integer-to-english-words/)
*Convert a non-negative integer num to its English words representation.*


```python
class Solution:
    def numberToWords(self, num):
        """
        10^0
        10^3: Thousand
        10^6: Million
        10^9: Billion
        
        """
        def helper(num):
            
            num2word = {1:'One', 2:'Two', 3:'Three', 4:'Four', 5:'Five', 6:'Six', 7:'Seven', 8:'Eight', 9:'Nine'}
            num2word10 = {10:'Ten', 11:'Eleven', 12:'Twelve', 13:'Thirteen', 14:'Fourteen', 15:'Fifteen', 16:'Sixteen', 17:'Seventeen', 18:'Eighteen', 19:'Nineteen', 2:'Twenty', 3:'Thirty', 4:'Forty', 5:'Fifty', 6:'Sixty', 7:'Seventy', 8:'Eighty', 9:'Ninety'}
            
            word = []
            i = 2
            while num > 0:
                q = num // 10**i
                if q > 0:
                    if i == 2:
                            word.extend([num2word[q], 'Hundred'])
                    elif i == 1:
                        if q == 1:
                            word.append(num2word10[num])
                            return word
                        else:
                            word.append(num2word10[q])
                    else:
                        word.append(num2word[q])
                
                num %= 10**i
                i -= 1
            
            return word
        
        if num == 0:
            return 'Zero'
                        
        h = {9:'Billion', 6:'Million', 3:'Thousand'}
        res = []
        i = 9
        while num > 0:
            q, num = divmod(num, 10**i)
            if q > 0:
                res.extend(helper(q))
                if i > 0:
                    res.append(h[i])
            i -= 3

        return " ".join(res)
```

## [Integer to Roman](https://leetcode.com/problems/integer-to-roman/)
*Given an integer, convert it to a roman numeral.*


```python
class Solution:
    def intToRoman(self, num: int) -> str:
        dic = {
            1:'I',
            4:'IV',
            5:'V',
            9:'IX',
            10:'X',
            40:'XL',
            50:'L',
            90:'XC',
            100:'C',
            400:'CD',
            500:'D',
            900:'CM',
            1000:'M'
        }
        res = ""
        i = 3
        while num > 0:
            div = 10**i
            q, num = divmod(num, div)
            
            if q > 0:
                currnum = q*div
                if currnum in dic: # q = 1, 4, 5, 9
                    res += dic[currnum]
                elif q < 5: # 2, 3
                    res += dic[div]*q
                else: # 6, 7, 8
                    res += dic[5*div] + dic[div]*(q-5)
            i -= 1
        return res
```

# Palindrome
-[Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)
-[Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/)
-[Palindrome Permutation](https://leetcode.com/problems/palindrome-permutation/)
-[Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)
-[Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)
-[Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)
-[Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/)
-[Palindrome Pairs](https://leetcode.com/problems/palindrome-pairs/)
-[Shortest Palindrome](https://leetcode.com/problems/shortest-palindrome/)


## Valid Palindrome
*Given a string s, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.*


```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        
        l, r = 0, len(s)-1
        while l < r:
            while l < r and not s[l].isalnum():
                l += 1
            while l < r and not s[r].isalnum():
                r -= 1
            
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
            
        return True
```

## Valid Palindrome II
*Given a string s, return true if the s can be palindrome after deleting at most one character from it.*


```python
class Solution:
    def validPalindrome(self, s: str) -> bool:

        l, r = 0, len(s)-1
        while l < r:
            if s[l] == s[r]:
                l += 1
                r -= 1
            else:
                return self._validPalindrome(s, l+1, r) or self._validPalindrome(s, l, r-1)
        
        return True
    
    def _validPalindrome(self, s, l, r):

        while l < r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True
```

## Palindrome Permutation
*Given a string s, return true if a permutation of the string could form a palindrome.*


```python
class Solution:
    def canPermutePalindrome(self, s: str) -> bool:
        
        # Solution1. double pass with dic
        dic = collections.defaultdict(int)
        
        for ch in s:
            dic[ch] += 1
        
        oddcnt = 0
        for cnt in dic.values():
            if cnt % 2 == 1:
                oddcnt += 1
                if oddcnt > 1:
                    return False
        return True
    
    
        # Solution2. single pass with dic
        dic = collections.defaultdict(int)
        oddcnt = 0
        for ch in s:
            dic[ch] += 1
            if dic[ch] % 2 == 0:
                oddcnt -= 1
            else:
                oddcnt += 1
        return oddcnt <= 1
    
        
        # Solution3. single pass with set
        h = set()
        for ch in s:
            if ch in h:
                h.remove(ch)
            else:
                h.add(ch)
        return len(h) <= 1
```

## Palindromic Substrings
*Given a string s, return the number of palindromic substrings in it.*


```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        """
        dp[i][j]: # of palidromic substrings in s[i:j+1]
        l = 1, ... n-1
        i = 0, ..., n-l-1
        j = i + l
        
        """
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        
        cnt = n
        for l in range(1, n):
            for i in range(n - l):
                j = i + l
                if s[i] == s[j] and (j - i + 1 <= 3 or dp[i+1][j-1] == True):
                    dp[i][j] = True
                    cnt += 1
        return cnt
```

## Longest Palindromic Substring
*Given a string s, return the longest palindromic substring in s.*


```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        """
        dp[i][j]: whether s[i:j+1] is palindromic or not
        
        l = 1, ..., n-1
        i = 0, ..., n-1-l
        j = i + l
        """
        n = len(s)
        if n == 1:
            return s
        
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
            
        maxstr = s[0]
        for l in range(1, n):
            for i in range(n - l):
                j = i + l
                if s[i] == s[j] and (j - i + 1 <= 3 or dp[i+1][j-1] == True):
                    dp[i][j] = True
                    maxstr = s[i:j+1]
                else:
                    dp[i][j] = False
        return maxstr
```

## Longest Palindromic Subsequence
*Given a string s, find the longest palindromic subsequence's length in s.*


```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        """
        dp[i][j]: the longest palindromic subsequence's length in s[i:j+1]
        
        l = 1, ..., n-1
        i = 0, ..., n-l-1
        j = i + l
        
        """
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        
        for l in range(1, n):
            for i in range(n - l):
                j = i + l
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
                    
        return dp[0][-1]
```

## Palindrome Partitioning
*Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.*


```python
class Solution:
    
    def validPalindrome(self, st):
        
        l, r = 0, len(st) - 1
        while l < r:
            if st[l] == st[r]:
                l += 1
                r -= 1
            else:
                return False
        return True
            
        
    def partition(self, s: str) -> List[List[str]]:
        
        def helper(i=0, curr=[]):
            
            if i == n:
                res.append(curr)
                return 
            
            for j in range(i, n):
                if self.validPalindrome(s[i:j+1]):
                    helper(j+1, curr + [s[i:j+1]])
            
        n = len(s)
        res = []
        helper()
        return res
```

## Palindrome Pairs
*Given a list of unique words, return all the pairs of the distinct indices (i, j) in the given list, so that the concatenation of the two words words[i] + words[j] is a palindrome.*


```python
class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        """
        (i, j), (j, i) 
        """
        # Solution1 Brute Force. TLE
        # O(n^2 * k), O(n)
        n = len(words)
        res = []
        for i in range(n-1):
            for j in range(i+1, n):
                if self.validPalindrome(words[i] + words[j]):
                    res.append([i, j])
                if self.validPalindrome(words[j] + words[i]):
                    res.append([j, i])
        return res
        
        
        """
        i = 0, ..., n
        rev_suff pref        suff
                 word[:i]    word[i:]
        dcba     i=0 ""      abcd           
        dcb      i=1 a       bcd    
                 i=2 ab      
                 i=3 abc
                 i=4 abcd
        
        pref     suff        rev_pref
        word[:i] word[i:]
                 i=0 abcd
                 i=1 bcd
                 i=2 cd
        abc      i=3 d       cba
   
        """
        # Solution2 dict
        n = len(words)
        wordsdic = {word:i for i, word in enumerate(words)}
        res = []
        for idx, word in enumerate(words):
            
            reword = word[::-1]
            if reword in wordsdic and wordsdic[reword]!= idx:
                res.append([idx, wordsdic[reword]])
            if word == reword and word !="" and "" in wordsdic:
                res.extend([[idx, wordsdic[""]],[wordsdic[""], idx]])
            
            for i in range(1, len(word)):
                pref = word[:i]
                suff = word[i:]
                rev_pref = pref[::-1]
                rev_suff = suff[::-1]
                
                if self.validPalindrome(pref) and rev_suff in wordsdic:
                    res.append([wordsdic[rev_suff], idx])
                if self.validPalindrome(suff) and rev_pref in wordsdic:
                    res.append([idx, wordsdic[rev_pref]])

        return res

    def validPalindrome(self, st):
        
        l, r = 0, len(st) - 1
        while l < r:
            if st[l] == st[r]:
                l += 1
                r -= 1
            else:
                return False
        return True
```

## Shortest Palindrome
*You are given a string s. You can convert s to a palindrome by adding characters in front of it. Return the shortest palindrome you can find by performing this transformation.*


```python
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        
        # Solution1 TLE
        for i in range(len(s) + 1, 0, -1):
            pref = s[:i]
            suff = s[i:]
            rev_suff = suff[::-1]
            if self.validPalindrome(pref):
                return rev_suff + s
        return ""

    def validPalindrome(self, st):
        
        l, r = 0, len(st)-1
        while l < r:
            if st[l] == st[r]:
                l += 1
                r -= 1
            else:
                return False
        return True
```

[<-PREV](dsa.md)
