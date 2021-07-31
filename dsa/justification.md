[<-PREV](string.md)

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

[<-PREV](string.md)
