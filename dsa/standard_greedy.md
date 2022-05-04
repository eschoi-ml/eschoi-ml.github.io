# Standard Greedy Algorithm

1. Maximum number of activity selelction 
1. Maximum profit in job sequencing with deadlines
1. Generate egyption fraction
1. Maximum value in Fractional Knapsack
1. Minimum empty space on wall by fitting shelves
1. Minimum number of coloring graph
1. Minimum number of platforms
1. Minimum number of meeting rooms
1. Minimum time to assign mice to holes
1. Minimum number of swaps to make the string balanced
1. Maximum number that policemen catch thieves
1. Water connection (dfs)
1. Huffman Coding



# 1. Maximum number of activity selection
You are given n activities with their start and finish times. Select the maximum number of activities that can be performed by a single person, assuming that a person can only work on a single activity at a time. 

    1. Sort the activities in ascending order based on their finish times.
    2. Select the first activity from this sorted list.
    3. Select a new activity from the list if its start time is greater than or equal to the finish time of the previously selected activity.
    4. Repeat the last step until all activities in the sorted list are checked.


```python
def activity_selection(start, finish):
    cnt = 1
    n = len(start)
    i = 0
    for j in range(n):
        if start[j] >= finish[i]:
            cnt += 1
            i = j
    return cnt

start = [10, 12, 20]
finish = [20, 25, 30]
activity_selection(start, finish)

start = [1, 3, 0, 5, 8, 5]
finish = [2, 4, 6, 7, 9, 9]
activity_selection(start, finish)
```




    4



# 2. Maximum profit in job sequencing with deadlines

Given an array of jobs where every job has a deadline and associated profit if the job is finished before the deadline. Find the maximum total profit earned by executing the tasks within the specified deadlines. Assume that a task takes one unit of time to execute, and it can't execute beyond its deadline. Also, only a single task will be executed at a time.

    1. Sort all jobs in decreasing order of profit
    2. Iterate on jobs in decreasing order of profit. For each job, do the following
        - For each job, find the latest empty time slot from j = min(deadline - 1, job[i][1] - 1) to 0
        - If found empty slot, put the job in the slot and mark this slot filled.


```python
def job_sequencing(job, deadline):

    job.sort(key = lambda x: x[2], reverse=True)

    res = [-1] * deadline

    for i in range(len(job)):

        j = min(deadline - 1, job[i][1] - 1)
        while j >= 0 and res[j] != -1:
            j -= 1
        res[j] = job[i][0]

    return res

job = [['a', 2, 100],  # Job Array [job_id, deadline, profit]
       ['b', 1, 19],
       ['c', 2, 27],
       ['d', 1, 25],
       ['e', 3, 15]]
deadline = 3
job_sequencing(job, deadline)
```




    ['c', 'a', 'e']



# 3. Generate egyptian fraction
Generate Egyptian fraction.Every positive fraction can be represented as sum of unique unit fractions. A fraction is unit fraction if numerator is 1 and denominator is a positive integer, for example 1/3 is a unit fraction.

    1. Find the greates possible unit fraction from n/d, ceiling (d/n): 6/14 -> ceil(14/6) = 3
    2. Update n and d as n*x - d and d*x: 6/14 - 1/3 = n/d - 1/x = (n*x - d)/(d*x) 
    3. Repeat 1-2 until n = 0


```python
from math import ceil

def egyptian_fraction(n, d):

    res = []
    while n != 0:

        x = ceil(d/n)
        res.append(x)
        n = n * x - d
        d = d * x
    
    for i, x in enumerate(res):
        if i != len(res)-1:
            print(f"1/{x} + ", end=" ")
        else:
            print(f"1/{x}")
egyptian_fraction(6, 14)
```

    1/3 +  1/11 +  1/231


# 4. Maximum value in Fractional Knapsack 
Given weights and values of n items, we need to put these items in a knapsack of capacity W to get the maximum total value in the knapsack. In Fractional Knapsack, we can break items for maximizing the total value of knapsack. 0/fractional/1 Knapsack!


    1. Calculate cost as (value / weight, index) 
    2. Sort cost 
    3. Select one or fractional weight & value depending on the remaining capacity

    value = [60, 40, 100, 120]
    weight = [10, 40, 20, 30]
    capacity = 50


```python
def fractiona_knapsack(value, weight, capacity):

    n = len(value)
    cost = [(value[i]/weight[i], i) for i in range(n)]
    cost.sort(reverse=True)

    total_value = 0
    for i in range(n):
        idx = cost[i][1]
        curr_value = value[idx]
        curr_weight = weight[idx]

        if curr_weight < capacity:
            capacity -= curr_weight
            total_value += curr_value
        else:
            total_value += curr_value * capacity / curr_weight
            break
    
    return total_value

value = [60, 40, 100, 120]
weight = [10, 40, 20, 30]
# cost = [(6, 0), (1, 1), (5, 2), (4, 3)]
# cost = [(6, 0), (5, 2), (4, 3), (1, 1)]
capacity = 50
fractiona_knapsack(value, weight, capacity)
```




    240.0



# 5. Minimum empty space on wall by fitting shelves
Given length of wall w and shelves of two lengths m and n, find the number of each type of shelf to be used and the remaining empty space in the optimal solution so that the empty space is minimum. The larger of the two shelves is cheaper so it is preferred. However cost is secondary and first priority is to minimize empty space on wall.

    1. Increase the number of longer shelf from 0
    2. Fit the remaining wall with the shorter shelf
    3. Calculate minimum length of the remaining wall


```python
def fit_shelves(w, m, n):

    longS = shortS = 0
    if m > n:
        longS, longI = m, 0
        shortS, shortI = n, 1
    else:
        longS, longI = n, 1
        shortS, shortI = m, 0

    res = [0, 0]
    min_wall = w
    i = 0
    while i*longS <= w and min_wall != 0: 

        q, r = divmod((w - i*longS), shortS)

        if r < min_wall:
            min_wall = min(min_wall, r)
            res[longI], res[shortI] = i, q
        i += 1
    
    return res + [min_wall]

w = 29
m = 9
n = 3
print(fit_shelves(w, m, n))

w = 24
m = 4
n = 7
print(fit_shelves(w, m, n))
```

    [0, 9, 2]
    [6, 0, 0]


# 6. Minimum number of coloring graph
Graph coloring (also called vertex coloring) is a way of coloring a graph's vertices such that no two adjacent vertices share the same color. 
1. K-colorable graph: Given K colors, find a way of coloring the vertices. 
2. K-chromatic graph: Find the minimum number K of colors used.
 

    1. Color first vertex with first color: res[0] = 0
    2. Do the following for remaining V-1 vertices
        - Consider the currently picked vertex and color it with the lowest numbered color that has NOT been used on any previously colored vertices adjacent to it. 
        - If all previously used colors appear on vertices adjacent to v, assign a new color to it. 


```python
def graph_coloring(graph, V):

    res = [-1] * V 
    res[0] = 0

    for i in range(1, V):
        
        available_color = [True] * V

        for j in graph[i]:
            if res[j] != -1:
                available_color[res[j]] = False

        color = 0
        while color < V and not available_color[color]:
            color += 1
        
        res[i] = color
    
    return res

graph = {
    0: [1, 4, 5],
    1: [0, 3, 4],
    2: [3, 4],
    3: [1, 2],
    4: [0, 1, 2, 5],
    5: [0, 4]
}
print(graph_coloring(graph, len(graph)))

graph = [[1, 2],[0, 2, 3],[0, 1, 3],[1, 2, 4],[3]]
print(graph_coloring(graph, len(graph)))
```

    [0, 1, 0, 2, 2, 1]
    [0, 1, 2, 0, 1]


# 7. Minimum number of platforms

Given a schedule containing the arrival and departure time of trains in a station, find the minimum number of platforms needed to avoid delay in any train’s arrival.


    1. sort arr, dep
    2. For each time, platform += 1 if arr platform -= 1 if dep




```python
def minimum_platform(arr, dep):

    arr.sort()
    dep.sort()

    n = len(arr)

    cnt = curr_cnt = 0
    i = j = 0
    while i < n and j < n:

        if arr[i] < dep[j]:
            curr_cnt += 1
            i += 1
            cnt = max(cnt, curr_cnt)
        else: # arr[i] >= dep[j] 
            curr_cnt -= 1
            j += 1
    return cnt


arr = [900, 940, 950, 1100, 1500, 1800]
dep = [910, 1200, 1120, 1130, 1900, 2000]
print(minimum_platform(arr, dep))

arr = [2.00, 2.10, 3.00, 3.20, 3.50, 5.00]
dep = [2.30, 3.40, 3.20, 4.30, 4.00, 5.20]
print(minimum_platform(arr, dep))
```

    3
    2


# 8. Minimum number of meeting rooms ([Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/submissions/))
Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.



```python
def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    # O(nlogn) & O(n)
    n = len(intervals)
    arr = []
    dep = []
    for i in range(n):
        arr.append(intervals[i][0])
        dep.append(intervals[i][1])
    arr.sort()
    dep.sort()
    
    cnt = curr_cnt = 0
    i = j = 0
    while i < n and j < n:
        if arr[i] < dep[j]:
            curr_cnt += 1
            i += 1
            cnt = max(cnt, curr_cnt)
        else:
            curr_cnt -= 1
            j += 1
            
    return cnt
```

# 9. Minimum time to assign mice to holes
There are N Mice and N holes are placed in a straight line. Each hole can accommodate only 1 mouse. A mouse can stay at his position, move one step right from x to x + 1, or move one step left from x to x -1. Any of these moves consumes 1 minute. Assign mice to holes so that the time when the last mouse gets inside a hole is minimized.

    1. Sort positions of mice and holes
    2. Return the maximum difference between mice[i] and holes[i]


```python
def assign_holes(mices, holes):

    mices.sort()
    holes.sort()

    max_difftime = 0
    for mice, hole in zip(mices, holes):
        max_difftime = max(max_difftime, abs(mice - hole))
    
    return max_difftime

mices = [4, -4, 2]
holes = [4, 0, 5]
print(assign_holes(mices, holes))

mices = [-10, -79, -79, 67, 93, -85, -28, -94] 
holes = [-2, 9, 69, 25, -31, 23, 50, 78]
print(assign_holes(mices, holes))
```

    4
    102


# 10. [Minimum number of swaps to make the string balanced](https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/)
You are given a 0-indexed string s of even length n. The string consists of exactly n / 2 opening brackets '[' and n / 2 closing brackets ']'. Return the minimum number of swaps to make s balanced.


```python
def minSwaps(s):

    bal = res = 0
    for c in s:
        if c == '[':
            bal += 1
        else:
            bal -= 1
            if bal < 0: # swap ] -> [
                res += 1
                bal = 1
    return res

s = "][]["
print(minSwaps(s))

s = "]]][[["
print(minSwaps(s))
```

    1
    2


# 11. Maximum number that policemen catch thieves
Given an array of size n that has the following specifications: 
Each element in the array contains either a policeman or a thief.
Each policeman can catch only one thief. A policeman cannot catch a thief who is more than K units away from the policeman.Find the maximum number of thieves that can be caught.

    1. Get the lowest index of policeman p and thief t. Make an allotment if |p-t| <= k and increment to the next p and t found. 
    2. Otherwise increment min(p, t) to the next p or t found. 
    3. Repeat above two steps until next p and t are found. 
    4. Return the number of allotments made.


```python
def police_thief(arr, k):
    
    n = len(arr)
    police = []
    thief = []
    for i, c in enumerate(arr):
        if c == 'P':
            police.append(i)
        else:
            thief.append(i)

    res = 0
    i = j = 0
    while i < len(police) and j < len(thief):

        if abs(police[i] - thief[j]) <= k:
            res += 1
            i += 1
            j += 1
        elif police[i] < thief[j]:
            i += 1        
        else: # police[i] > thief[j]
            j += 1
    return res

arr = ['P', 'T', 'T', 'P', 'T']
k = 2
print(police_thief(arr, k))

arr = ['P', 'T', 'P', 'T', 'T', 'P']
k = 3
print(police_thief(arr, k))
```

    2
    3


# 12. Water connection

Every house in the colony has at most one pipe going into it and at most one pipe going out of it. Tanks and taps are to be installed in a manner such that every house with one outgoing pipe but no incoming pipe gets a tank installed on its roof and every house with only an incoming pipe and no outgoing pipe gets a tap.

Given two integers n and p denoting the number of houses and the number of pipes. The connections of pipe among the houses contain three input values: a_i, b_i, d_i denoting the pipe of diameter d_i from house a_i to house b_i, find out the efficient solution for the network. 

The output will contain the number of pairs of tanks and taps t installed in first line and the next t lines contain three integers: house number of tank, house number of tap and the minimum diameter of pipe between them.


```python
def water_connection(n, p, pipes):

    def dfs(node, path, mindia):

        if not graph[node]:
            res.append([path[0], path[-1], mindia])
            return 

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                mindia = min(mindia, graph[node][neighbor])
                path[-1] = neighbor
                dfs(neighbor, path, mindia)

    # graph
    graph = {i: {} for i in range(1, n+1)}
    innodes = set()
    outnodes = set()
    for u, v, d in pipes:
        
        graph[u][v] = d

        innodes.add(v)
        outnodes.add(u)
    outnodes -= innodes
    
    # dfs
    visited = set()
    res = []
    for node in outnodes:
        if node not in visited:            
            visited.add(node)
            dfs(node, [node, node], float('inf'))
    return res

n = 9
p = 4
pipes = [
[7, 4, 98],
[5, 9, 72],
[4, 6, 10],
[2, 8, 22]
]
print(water_connection(n, p, pipes))

n = 9
p = 6
pipes = [
[7, 4, 98], 
[5, 9, 72], 
[4, 6, 10],
[2, 8, 22], 
[9, 7, 17], 
[3, 1, 66]
]
print(water_connection(n, p, pipes))
```

    [[2, 8, 22], [5, 9, 72], [7, 6, 10]]
    [[2, 8, 22], [3, 1, 66], [5, 6, 10]]


# 13. Huffman coding
The basic idea is to use 'variable-length encoding.' We assign a variable number of bits to characters depending upon their frequency in the given text. Thus, the more frequently used characters are, the shorter their encodings are resulting in overal small file size. The condition is the 'prefix rule,' which no code is is a prefix of another code.
 
- a 0
- b 11
- c 100
- d 011


```python
import heapq
from collections import Counter

# Tree node
class Node:
    def __init__(self, ch, freq, left=None, right=None):
        self.ch = ch
        self.freq = freq
        self.left = left
        self.right = right

    # Override the '__lt__' funtion to make 'Node' class work with mean heap
    def __lt__(self, other):
        return self.freq < other.freq

def Encode_HuffmanCode(text):

    # Build Huffman Tree from text: word in leaf nodes
    def build_HuffmanTree(text):
        if len(text)==0:
            return 

        freq = Counter(text)
        pq = [Node(ch, freq) for ch, freq in freq.items()]
        heapq.heapify(pq)

        while len(pq) != 1:
            left = heapq.heappop(pq)
            right = heapq.heappop(pq)
            total = left.freq + right.freq
            heapq.heappush(pq, Node(None, total, left, right))
        root = pq[0]
        return root
    
    def encode_HuffmanTree(node, s):

        if not node:
            return
        
        if not node.left and not node.right: # leaf
            huffman_code[node.ch] = s if len(s) > 0 else '1'
        
        encode_HuffmanTree(node.left, s + '0')
        encode_HuffmanTree(node.right, s + '1')

    def encode(text, huffman_code):

        encoded_text = ""
        for ch in text:
            encoded_text += huffman_code.get(ch)
        return encoded_text 

    root = build_HuffmanTree(text)

    huffman_code = {}
    encode_HuffmanTree(root, "")
    encoded_text = encode(text, huffman_code)

    return encoded_text, root

def Decode_Huffmancode(encoded_text, root):

    # Traverse the Huffman Tree and decode the encoded text
    # return the index
    def decode(encoded_text, index, node):
        
        if not node.left and not node.right:
            decoded_text.append(node.ch)
            return index

        if encoded_text[index] == '0':
            return decode(encoded_text, index+1, node.left)
        else:
            return decode(encoded_text, index+1, node.right)
    
    #When the tree has only one root node 
    if root and not root.left and not root.right:
        decoded_text = ""
        while root.freq > 0:
            decoded_text += root.ch
            root.freq -= 1
    else:
        decoded_text = []
        index = 0
        while index < len(encoded_text):
            index = decode(encoded_text, index, root)
        decoded_text = "".join(decoded_text)
    return decoded_text

text = 'Huffman coding is a data compression algorithm.'
print('Original text: ', text)
encoded_text, root = Encode_HuffmanCode(text)
print('Encoded text: ', encoded_text)
decoded_text = Decode_Huffmancode(encoded_text, root)
print('Decoded text: ', decoded_text)

text = 'aaa'
print('Original text: ', text)
encoded_text, root = Encode_HuffmanCode(text)
print('Encoded text: ', encoded_text)
decoded_text = Decode_Huffmancode(encoded_text, root)
print('Decoded text: ', decoded_text)
```

    Original text:  Huffman coding is a data compression algorithm.
    Encoded text:  11111100001111001110001110100110101000111001111000101101101110100110011010101011111001010001010101000111000111111110110101000010011001001110001101010101110111101111001101000110001111010011100000
    Decoded text:  Huffman coding is a data compression algorithm.
    Original text:  aaa
    Encoded text:  111
    Decoded text:  aaa

