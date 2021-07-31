[<-PREV](string.md)

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



[<-PREV](string.md)
