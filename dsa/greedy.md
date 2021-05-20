[<-PREV](dsa.md)

# Greedy Algorithm
Let's try solving Top 8 Greedy algorithm questions!

1. Activity selelction 
1. Fractional Knapsack
1. Graph coloring
1. Job sequencing with deadlines
1. Minimum number of platforms/ meeting rooms
1. Huffman Coding
1. Dijkstra's algorithm for shortest paths from a single source (in-progress)
1. Kruska's and Prim's minimum spanning tree (in-progress)

Inspired by these two articles [[Top 7]](https://medium.com/techie-delight/top-7-greedy-algorithm-problems-3885feaf9430) & [[Top 20]](https://www.geeksforgeeks.org/top-20-greedy-algorithms-interview-questions/?ref=rp).


### Activity selection
You are given n activities with their start and finish times. Select the maximum number of activities that can be performed by a single person, assuming that a person can only work on a single activity at a time. 

---
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



### Fractional Knapsack 
Given weights and values of n items, we need to put these items in a knapsack of capacity W to get the maximum total value in the knapsack. In Fractional Knapsack, we can break items for maximizing the total value of knapsack. 0/fractional/1 Knapsack!

---
- Calculate cost as (value / weight, index) 
- Sort cost 
- Select one or fractional weight & value depending on the remaining capacity

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



### Graph coloring
Graph coloring (also called vertex coloring) is a way of coloring a graph's vertices such that no two adjacent vertices share the same color. 
1. K-colorable graph: Given K colors, find a way of coloring the vertices. 
2. K-chromatic graph: Find the minimum number K of colors used. 

---
1. Color first vertex with first color. 
2. Do the following for remaining V-1 vertices
    - Consider the currently picked vertex and color it with the lowest numbered color that has NOT been used on any previously colored vertices adjacent to it. If all previously used colors appear on vertices adjacent to v, assign a new color to it. 


```python
def graph_coloring(graph, V):

    res = [-1] * V 
    res[0] = 0

    for i in range(1, V):
        
        available = [True] * V

        for j in graph[i]:
            if res[j] != -1:
                available[res[j]] = False

        color = 0
        while color < V and not available[color]:
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
graph_coloring(graph, len(graph))

graph = [[1, 2],[0, 2, 3],[0, 1, 3],[1, 2, 4],[3]]
graph_coloring(graph, len(graph))
```




    [0, 1, 2, 0, 1]



### Job sequencing with deadlines

Given an array of jobs where every job has a deadline and associated profit if the job is finished before the deadline. Find the maximum total profit earned by executing the tasks within the specified deadlines. Assume that a task takes one unit of time to execute, and it can't execute beyond its deadline. Also, only a single task will be executed at a time.

---

1. Sort all jobs in decreasing order of profit
2. Iterate on jobs in decreasing order of profit. For each job, do the following
    - For each job, find an empty time slot from deadline to 0. If found empty slot, put the job in the slot and mark this slot filled.


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

job = [['a', 2, 100],  # Job Array
       ['b', 1, 19],
       ['c', 2, 27],
       ['d', 1, 25],
       ['e', 3, 15]]
deadline = 3
job_sequencing(job, deadline)
```




    ['c', 'a', 'e']



### Minimum number of platforms

Given a schedule containing the arrival and departure time of trains in a station, find the minimum number of platforms needed to avoid delay in any train’s arrival.
___
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
        else:
            curr_cnt -= 1
            j += 1
    return cnt


arr = [900, 940, 950, 1100, 1500, 1800]
dep = [910, 1200, 1120, 1130, 1900, 2000]
minimum_platform(arr, dep)

arr = [2.00, 2.10, 3.00, 3.20, 3.50, 5.00]
dep = [2.30, 3.40, 3.20, 4.30, 4.00, 5.20]
minimum_platform(arr, dep)
```




    2



#### [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/submissions/)
Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.



```python
def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    
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

### Huffman Coding
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

    # Build Huffman Tree from text
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
        
        if not node.left and not node.right:
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

    # Traverse ethe Huffman Tree and decode the encoded text
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

#text = 'Huffman coding is a data compression algorithm.'
text = 'I love estelle!'
#text = 'aaa'
encoded_text, root = Encode_HuffmanCode(text)
print('Original text: ', text)
print('Encoded text: ', encoded_text)

decoded_text = Decode_Huffmancode(encoded_text, root)
print('Decoded text: ', decoded_text)
```

    Original text:  I love estelle!
    Encoded text:  10011100011101010011100110001111010000011011
    Decoded text:  I love estelle!


### Dijkstra's algorithm for shortest paths from a single source
given a source vertex s from a set of vertices V in a weighted graph where all its edge weights w(u, v) are non-negative, find the shortest/smallest path weights d(s, v) for all vertices v present in the graph.




```python
import heapq
def get_route(target, previous):

    route = []
    v = target
    while v != -1:
        route.append(v)
        v = previous[v]
    return route[::-1]

def Dijkstra(source, graph):

    N = len(graph)
    pq = [(0, source)]

    distance = [float('inf')] * N
    distance[source] = 0

    visited = [False] * N
    visited[source] = True

    previous = [-1] * N
    route = []

    while pq:

        d, v = heapq.heappop(pq)
        visited[v] = True

        for n, w in graph[v].items():
            new_dist = d + w
            if not visited[n] and new_dist < distance[n]:
                distance[n] = new_dist
                previous[n] = v
                heapq.heappush(pq, (distance[n], n))

    for i in range(1, N):
        if i!= source and distance[i]!=float('inf'):
            route = get_route(i, previous)
            print(f'Path from {source} to {i}: {route} with the minimum cost {distance[i]}')

graph = {0:{1:10, 4:3},
         1:{2:2, 4:4},
         2:{3:9},
         3:{2:7},
         4:{1:1, 2:8, 3:2}
         }
Dijkstra(0, graph)
```

    Path from 0 to 1: [0, 4, 1] with the minimum cost 4
    Path from 0 to 2: [0, 4, 1, 2] with the minimum cost 6
    Path from 0 to 3: [0, 4, 3] with the minimum cost 5
    Path from 0 to 4: [0, 4] with the minimum cost 3


### Minimum Spanning Tree (MST)

#### Kruska's MST
#### Prim's MST




[<-PREV](dsa.md)
