[<-PREV](dsa.md)

# Dynamic Programming
# Pattern 4. DP on strings

- Substring: a contiguous sequence of characters within the string.
- Subsequence: a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.
- Supersequence: the shortest string that has both str1 and str2 as subsequences.  

One string: Palindromic Substring and Subsequence
```python
for l in range(1, n):
    for i in range(n-l):
        j = i + l
        if s[i] == s[j] and ... :
            dp[i][j] = update
        else:
            dp[i][j] = update
```
- Palindromic Substrings
- Longest Palindromic Substring
- Longest Palindromic Subsequence


Two strings: Subsequence or Supersequence
```python
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if word1[i-1] == word2[j-1]:
            dp[i][j] = dp[i-1][j-1] + ...
        else:
            dp[i][j] = ...
```
- Longest Common Subsequence
- Shortest Common Supersequence
- Distinct Subsequences
- Delete Operation for Two Strings
- Edit Distance
- Minimum ASCII Delete Sum for Two Strings



## One string: Palindromic Substring and Subsequence

### [Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)
*Given a string s, return the number of palindromic substrings in it.*


```python
    def countSubstrings(self, s: str) -> int:
        """
        dp[i][j]: boolean of palindromic substrings from s[i] to s[j]
            a a a
        i\j 0 1 2
        a 0 1 
        a 1 0 1
        a 2 0 0 1

        l = 1, ..., n-1
        i = 0, ..., n-l-1
        j = i + l = l, ..., n-1
        
        if s[i] == s[j] and ...:
            dp[i][j] = dp[i+1][j-1] + 2
            
        """
        
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        
        res = n
        for l in range(1, n):
            for i in range(n-l):
                j = i + l
                if s[i] == s[j] and (j - i + 1 <= 3 or dp[i+1][j-1] == j - i - 1):
                    dp[i][j] = dp[i+1][j-1] + 2
                    res += 1
        return res
```

### [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)
*Given a string s, return the longest palindromic substring in s.*


```python
def longestPalindrome(self, s: str) -> str:
    
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
        

    res = s[0]
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            if s[i] == s[j] and (j - i + 1 <= 3 or dp[i+1][j-1] == j - i - 1):
                dp[i][j] = dp[i+1][j-1] + 2
                res = s[i:j+1]
    return res
```

### [Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)
*Given a string s, find the longest palindromic subsequence's length in s.*


```python
def longestPalindromeSubseq(self, s: str) -> int:
    """
    dp[i][j]: the longest palindromic subsequences's length from s[i] to s[j]
    
    l = 1, ..., n-1
    i = 0, ..., n-l-1
    j = i + l = l, ..., n-1
    
    1) s[i] == s[j]:
    dp[i][j] = dp[i+1][j-1] + 2
    2) s[i] != s[j]
    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    
    
    """
    
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    
    for l in range(1, n):
        for i in range(n-l):
            j = i + l
            if s[i]==s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])

    return dp[0][-1]
```

## Two strings: Subsequence or Supersequence


### [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
*Given two strings text1 and text2, return the length of their longest common subsequence.*


```python
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    """
    dp[i][j]: the length of the longest common subsequence 
                from text1[:i] and text2[:j]
    
          a c e
    i\j 0 1 2 3
      0      ans
    a 1
    b 2
    c 3
    d 4
    e 5
    
    if text1[i-1] == text2[j-1]:
        dp[i][j] = dp[i-1][j-1] + 1
    else:
        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    """
    # Solution 1. O(m*n) & O(m*n)
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[-1][-1]
    
    
    # Space optimized solution: O(m*n) & O(min(m, n))
    
    if len(text1) < len(text2):
        text1, text2 = text2, text1
    m, n = len(text1), len(text2)
    
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        curr_dp = [0] * (n + 1)
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                curr_dp[j] = dp[j-1] + 1
            else:
                curr_dp[j] = max(dp[j], curr_dp[j-1])
        dp = curr_dp
    return dp[-1]
        
```

### [Shortest Common Supersequence](https://leetcode.com/problems/shortest-common-supersequence/)
*Given two strings str1 and str2, return the shortest string that has both str1 and str2 as subsequences.*


```python
def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
    """
    dp[i][j]: the longest common subsequence string between str1[:i] and str[:j]
    
    """
    m, n = len(str1), len(str2)
    dp = [[""] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + str1[i-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1], key=len)
    
    res = ""
    i = j = 0
    for ch in dp[m][n]:

        while i < m and str1[i] != ch:
            res += str1[i]
            i += 1
        while j < n and str2[j] != ch:
            res += str2[j]
            j += 1
        
        res += str1[i]
        i += 1
        j += 1
    
    if i < m:
        res += str1[i:]
    
    if j < n:
        res += str2[j:] 

    return res
```

### [Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/)
*Given two strings s and t, return the number of distinct subsequences of s which equals t.*


```python
def numDistinct(self, s: str, t: str) -> int:
    """
    dp[i][j]: the number of distinct subseuqences of s[:i] which equals t[:j]
            r a b b i t
    i\j 0 1 2 3 4 5 6
        0 1 0 0 0 0 0 0 
    r 1 0 1 0 0 0 0 0 
    a 2 0 1 1 0 0 0 0 
    b 3 0 1 1 1 0 0 0 
    b 4 0 1 1 2 1 0 0 
    b 5 0 1 1 3 3 0 0
    i 6 0 1 1 3 3 3 0
    t 7 0 1 1 3 3 3 3
    
    
    if s[i-1] == t[i-1]:
        dp[i][j] = dp[i-1][j] + dp[i-1][j-1]
    else:
        dp[i][j] = dp[i-1][j]
    
    """
    # Solution: O(m*n) & O(m*n)
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = 1
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
            else:
                dp[i][j] = dp[i-1][j]
    return dp[-1][-1]

    # Space optimized solution: O(m*n) & O(n)
    m, n = len(s), len(t)
    dp = [0] * (n + 1)
    dp[0] = 1
    
    for i in range(1, m + 1):
        curr_dp = [1] + [0] * n
        for j in range(1, n + 1):
            if s[i-1] == t[j-1]:
                curr_dp[j] = dp[j-1] + dp[j]
            else:
                curr_dp[j] = dp[j]
        dp = curr_dp
    return dp[-1]
```

### [Delete Operation for Two Strings](https://leetcode.com/problems/delete-operation-for-two-strings/)
*Given two strings word1 and word2, return the minimum number of steps required to make word1 and word2 the same.*



```python
def minDistance(self, word1: str, word2: str) -> int:
    """
    dp[i][j]: the length of the longest common subsequence
    
    """
    
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return m + n - 2 * dp[-1][-1]
```

### [Edit Distance](https://leetcode.com/problems/edit-distance/)
*Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.*


```python
def minDistance(self, word1: str, word2: str) -> int:
    """
    dp[i][j]: the minimum number of operations required 
                to convert word1[:i] to word2[:j]
    
    """
    m, n = len(word1), len(word2)
    if m == 0 or n == 0:
        return m or n
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for i in range(1, n + 1):
        dp[0][i] = i
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1])
            else:
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)
    return dp[-1][-1]
```

### [Minimum ASCII Delete Sum for Two Strings](https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/)
*Given two strings s1 and s2, return the lowest ASCII sum of deleted characters to make two strings equal.*


```python
def minimumDeleteSum(self, s1: str, s2: str) -> int:
    """
    dp[i][j]: the lowest ASCII sum of deleted characters to make s1[:i] and s2[:j] equal
    
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + ord(s1[i-1]) 
    for i in range(1, n + 1):
        dp[0][i] = dp[0][i-1] + ord(s2[i-1])
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j] + ord(s1[i-1]), dp[i][j-1] + ord(s2[j-1]))

    return dp[-1][-1]
```

[<-PREV](dsa.md)
