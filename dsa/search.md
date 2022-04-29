[<-PREV](dsa.md)

# Search with DFS, BFS, Topological Sort, Dijkstra, Bellman-Ford, Union-Find, and MST algorithms


1. Depth First Search algorithm - Recursive & Iterative using **Stack**
    - 1.1 Depth First Search: Return True or False if you can reach a target node('F') from a starting node ('A')
    - 1.2 Depth First Path Search: Return the path from a starting node ('A') to a target node('F') 
        - 1.2.1 Find a single path
        - 1.2.2 Find all paths: **Backtracking**. 

2. Breath First Search algorithm - Iterative using **Queue**
    - 2.1 Breath First Search - Return the shortest depth if you can reach a target node('F') from a starting node ('A')
    - 2.2 Breath First Path Search - Return the shortest path from a starting node ('A') to a target node('F')

3. Topological Sort - Iterative using **Queue** (Sort)
4. Dijkstra algorithm - Iterative using **Heap Queue** (Greedy)
    - 3.1 Find a single path & min distance with a distance variable
    - 3.2 Find a single path & min distance without a distance variable
    - 3.3 Find all paths & their min distances with a distance variable

5. Bellman-Ford algorithm(Dynamic Programming)
6. Union Find (Greedy)
7. Minimum Spanning Tree algorithm (Greedy)
    - 7.1 Kruskal’s Minimum Spanning Tree algorithm
    - 7.2 Prim’s Minimum Spanning Tree algorithm


```python
graph = {
    'A' : ['B', 'C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : ['F'],
    'F' : []
}
```


```python
def reconstruct_path(v1, v2, previous):
    v = v2
    path = [v]
    while  v!= v1:
        v = previous[v]
        path.append(v)
    return "->".join(path[::-1])   
```

# 1. Depth First Search algorithm

## 1.1 DFS recursive & iterative


```python
def dfs(graph, v1, v2):
    
    def dfs_recursive(v, v2):

        if v == v2:
            return True
        
        for neighbor in graph[v]:
            if neighbor not in visited:
                visited.add(neighbor)
                if dfs_recursive(neighbor, v2):
                    return True
        return False
    
    visited = set([v1])
    print("dfs recursive: ", dfs_recursive(v1, v2))
    
    def dfs_iterative(v1, v2):

        stack = [v1]
        visited = set([v1])

        while stack:

            v = stack.pop()

            if v == v2:
                return True
            for neighbor in graph[v]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return False
    
    print("dfs iterative: ", dfs_iterative(v1, v2))           

dfs(graph, 'A', 'F')
```

    dfs recursive:  True
    dfs iterative:  True


## 1.2 DFS **Path** recursive & iterative


```python
def dfs_path(graph, v1, v2):

    def dfs_singlePath_recursive(v, v2, path):

        if v == v2:
            res[:] = path
            return True

        for neighbor in graph[v]:
            if neighbor not in visited:
                visited.add(neighbor)
                if dfs_singlePath_recursive(neighbor, v2, path + [neighbor]):
                    return True
        return False

    res = []
    visited = set([v1])
    dfs_singlePath_recursive(v1, v2, [v1])
    print('dfs_singlePath_recursive: ', res)

    def dfs_singlePath_iterative(v1, v2):

        res = []

        previous = {v: None for v in graph}
        stack = [v1]
        visited = set([v1])

        while stack:
            v = stack.pop()
            if v == v2:
                res.append(reconstruct_path(v1, v2, previous))
                return res

            for neighbor in graph[v]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    previous[neighbor] = v
                    stack.append(neighbor)
        return res
    print('dfs_singlePath_iterative: ', dfs_singlePath_iterative(v1, v2))


    def dfs_allPath_recursive(v, v2, path): # backtracking 

        if v == v2:
            res.append(path)
            return 

        for neighbor in graph[v]:
            if neighbor not in visited:
                visited.add(neighbor)
                dfs_allPath_recursive(neighbor, v2, path + [neighbor])
                visited.remove(neighbor)

    res = []
    visited = set([v1])
    dfs_allPath_recursive(v1, v2, [v1])
    print('dfs_allPath_recursive: ', res)

dfs_path(graph, 'A', 'F')
```

    dfs_singlePath_recursive:  ['A', 'B', 'E', 'F']
    dfs_singlePath_iterative:  ['A->C->F']
    dfs_allPath_recursive:  [['A', 'B', 'E', 'F'], ['A', 'C', 'F']]


# 2. Breath First Search algorithm

## 2.1 BFS - Iterative



```python
from collections import deque

def bfs(graph, v1, v2):

    def bfs_iterative1(graph, v1, v2):
        
        queue = deque([(v1, 0)])
        visited = set([v1])

        while queue:
            v, depth = queue.popleft()
            if v == v2:
                return depth
            for neighbor in graph[v]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return depth
    print("bfs_iterative1: ", bfs_iterative1(graph, v1, v2))
    
    def bfs_iterative2(graph, v1, v2):
        
        queue = deque([v1])
        visited = set([v1])
        depth = -1

        while queue:

            depth += 1
            size = len(queue)

            for _ in range(size):
                v = queue.popleft()
                if v == v2:
                    return depth
                for neighbor in graph[v]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return -1
    print("bfs_iterative2: ", bfs_iterative2(graph, v1, v2))

bfs(graph, 'A', 'F')
```

    bfs_iterative1:  2
    bfs_iterative2:  2


## 2.2 BFS **Path** - Iterative


```python
from collections import deque

def bfs_path(graph, v1, v2):

    def bfs_singlePath_iterative1(graph, v1, v2):
        
        previous = {v: None for v in graph}
        queue = deque([(v1, 0)])
        visited = set([v1])

        while queue:
            v, depth = queue.popleft()
            if v == v2:
                return reconstruct_path(v1, v2, previous)
            for neighbor in graph[v]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    previous[neighbor] = v
                    queue.append((neighbor, depth + 1))
        return None
    print("bfs_singlePath_iterative1: ", bfs_singlePath_iterative1(graph, v1, v2))
    
    def bfs_singlePath_iterative2(graph, v1, v2):
        
        previous = {v:None for v in graph}
        queue = deque([v1])
        visited = set([v1])
        depth = -1

        while queue:

            depth += 1
            size = len(queue)

            for _ in range(size):
                v = queue.popleft()
                if v == v2:
                    return reconstruct_path(v1, v2, previous)
                for neighbor in graph[v]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        previous[neighbor] = v
                        queue.append(neighbor)
        return -1
    print("bfs_singlePath_iterative2: ", bfs_singlePath_iterative2(graph, v1, v2))

    def bfs_allPath_iterative(graph, v1, v2):
        res = []
        queue = deque([[v1]])
        while queue:
            path = queue.popleft()
            v = path[-1]
            if v == v2:
                res.append(path)
            else:
                for neighbor in graph[v]:
                    if neighbor not in path:
                        queue.append(path + [neighbor])
        return res
    print("bfs_allPath_iterative: ", bfs_allPath_iterative(graph, v1, v2))

    def bfs_shortestAllPath_iterative(graph, v1, v2):
        res = []
        queue = deque([[v1]])
        while queue and not res:
            size = len(queue)
            for _ in range(size):
                path = queue.popleft()
                v = path[-1]
                if v == v2:
                    res.append(path)
                else:
                    for neighbor in graph[v]:
                        if neighbor not in path:
                            queue.append(path + [neighbor])
        return res
    print("bfs_shortestAllPath_iterative: ", bfs_shortestAllPath_iterative(graph, v1, v2))

bfs_path(graph, 'A', 'F')
```

    bfs_singlePath_iterative1:  A->C->F
    bfs_singlePath_iterative2:  A->C->F
    bfs_allPath_iterative:  [['A', 'C', 'F'], ['A', 'B', 'E', 'F']]
    bfs_shortestAllPath_iterative:  [['A', 'C', 'F']]


# 3. Topological Sort
Directed Acyclic Graph(DAG)


```python
directed_acyclic_graph = {
    0:[],
    1:[],
    2:[3],
    3:[1],
    4:[0, 1],
    5:[0, 2]
}
```


```python
import collections

def kahn_topological_sort(graph):
    def topological_sort1(graph):

        # Initialize inDegree
        n = len(graph)
        inDegree = {v:0 for v in graph}
        for v in graph:
            for neighbor in graph[v]:
                inDegree[neighbor] += 1

        # Initialize Queue
        q = collections.deque()
        for v in graph:
            if inDegree[v] == 0:
                q.append(v)
        
        res = []
        while q:

            v = q.popleft()
            res.append(v)
            
            for neighbor in graph[v]:
                inDegree[neighbor] -= 1
                if inDegree[neighbor] == 0:
                    q.append(neighbor)
        
        return res if len(res) == n else []

    def topological_sort2(graph):

        # Initialize inDegree
        n = len(graph)
        inDegree = {v:0 for v in graph}
        for v in graph:
            for neighbor in graph[v]:
                inDegree[neighbor] += 1

        # Initialize Queue
        q = collections.deque()
        for v in graph:
            if inDegree[v] == 0:
                q.append(v)
        
        res = []
        depth = -1
        while q:

            depth += 1
            size = len(q)

            for _ in range(size):
                
                v = q.popleft()    
                res.append(v)    
                
                for neighbor in graph[v]:
                    inDegree[neighbor] -= 1
                    if inDegree[neighbor] == 0:
                        q.append(neighbor)
            
        
        return res if len(res) == n else []

    print("Topological sort1: ", topological_sort1(graph))
    print("Topological sort2: ", topological_sort2(graph))
kahn_topological_sort(directed_acyclic_graph)

```

    Topological sort1:  [4, 5, 0, 2, 3, 1]
    Topological sort2:  [4, 5, 0, 2, 3, 1]


# 4. Dijkstra's algorithm
Given a source vertex s from a set of vertices V in a weighted graph where all its edge weights w(u, v) are non-negative, find the shortest/smallest path weights d(s, v) for all vertices v present in the graph.


```python
nonnegative_graph = {
    0:{1:10, 4:3},
    1:{2:2, 4:4},
    2:{3:9},
    3:{2:7},
    4:{1:1, 2:8, 3:2}
}
```


```python
def reconstruct_path(v1, v2, previous):
    v = v2
    path = [str(v)]
    while v != v1:
        v = previous[v]
        path.append(str(v))
    return "->".join(path[::-1])
```


```python
import heapq
def dijkstra(graph, v1, v2):
    def dijkstra_singlePath(graph, v1, v2):
        # return min distance, path
        # keep track of distance with a variable "distance"

        previous = {v: None for v in graph}
        
        distance = {v: float('inf') for v in graph}
        distance[v1] = 0
        pq = [(0, v1)]
        visited = set()

        while pq:

            dist, v = heapq.heappop(pq)
            visited.add(v)

            if v == v2:
                return distance[v2], reconstruct_path(v1, v2, previous)

            for neighbor, w in graph[v].items():
                newdist = dist + w
                if neighbor not in visited and newdist < distance[neighbor]:
                    distance[neighbor] = newdist
                    previous[neighbor] = v
                    heapq.heappush(pq, (distance[neighbor], neighbor))
        return -1, None
    print("dijkstra_singlePath: ", dijkstra_singlePath(graph, v1, v2))

    def dijkstra_singlePath2(graph, v1, v2):
        # return min distance, path
        # keep track of distance without a variable "distance" 

        previous = {v: None for v in graph}
        pq = [(0, v1)]
        visited = set()

        while pq:

            dist, v = heapq.heappop(pq)
            visited.add(v)

            if v == v2:
                return dist, reconstruct_path(v1, v2, previous)

            for neighbor, w in graph[v].items():
                newdist = dist + w
                if neighbor not in visited:
                    previous[neighbor] = v
                    heapq.heappush(pq, (newdist, neighbor))
        return -1, None
    print("dijkstra_singlePath2: ", dijkstra_singlePath2(graph, v1, v2))

    def dijkstra_allPath(graph, v1):

        previous = {v: None for v in graph}
        distance = {v: float('inf') for v in graph}
        distance[v1] = 0
        pq = [(0, v1)]
        visited = set()

        while pq:

            dist, v = heapq.heappop(pq)
            visited.add(v)
            
            for neighbor, w in graph[v].items():
                newdist = dist + w
                if neighbor not in visited and newdist < distance[neighbor]:
                    distance[neighbor] = newdist
                    previous[neighbor] = v
                    heapq.heappush(pq, (distance[neighbor], neighbor))
        
        print("dijkstra_allPath: ")
        for v in graph:
            if v == v1:
                continue
            print(distance[v], reconstruct_path(v1, v, previous))
    dijkstra_allPath(graph, v1)       
    
dijkstra(nonnegative_graph, 0, 3)
```

    dijkstra_singlePath:  (5, '0->4->3')
    dijkstra_singlePath2:  (5, '0->4->3')
    dijkstra_allPath: 
    4 0->4->1
    6 0->4->1->2
    5 0->4->3
    3 0->4


# 5. Bellman-Ford algorithm
After relaxing each edge N-1 times, perform the Nth relaxation. According to the “Bellman-Ford algorithm”, all distances must be the shortest after relaxing each edge N-1 times. However, after the Nth relaxation, if there exists distances[u] + weight(u, v) < distances(v) for any edge(u, v), it means there is a shorter path . At this point, we can conclude that there exists a “negative weight cycle”.

Shortest distance to all vertices from src. If there is a negative weight cycle, then shortest distances are not calculated, negative weight cycle is reported.


```python
negative_graph = {
    'A': {'B':-1, 'C':4},
    'B': {'C':3, 'D':2, 'E':2},
    'C': {},
    'D': {'B':1, 'C':5},
    'E': {'D':-3}
}
```


```python
def bellmanford(graph, v1, v2):

    distance = {v:float('inf') for v in graph}
    distance[v1] = 0
    previous = {v: None for v in graph}

    N = len(graph)
    for i in range(N-1):
        for v in graph:
            for u, weight in graph[v].items():
                newdist = distance[v] + weight
                if newdist < distance[u]:
                    distance[u] = newdist
                    previous[u] = v
    
    for v in graph:
        for u, weight in graph[v].items():
            if distance[v] + weight < distance[u]: # negative weight cycle exist
                return None, None
    return distance[v2], reconstruct_path(v1, v2, previous)
bellmanford(negative_graph, 'A', 'D')
```




    (-2, 'A->B->E->D')



# 6. Union Find


```python
class UnionFind:
    def __init__(self, n):
        # O(n), O(n)
        self.parent = [i for i in range(n)]
        self.rank = [1] * n
        self.component = n
        
    def union(self, x, y):
        # O(alpha(n)) = O(1) where alpha is the inverse Ackermann function more efficient than O(log(n))
        # 1. union x and y
        # 2. return True if x and y are not currently connected/ so just connected/ not cycled
        px = self.find(x)
        py = self.find(y)
        
        if px == py:
            return False
        
        if self.rank[px] > self.rank[py]:
            self.parent[py] = px
            self.rank[px] += self.rank[py]
        elif self.rank[px] < self.rank[py]:
            self.parent[px] = py
            self.rank[py] += self.rank[px]
        else:
            self.parent[py] = px
            self.rank[px] += 1
        
        self.component -= 1
    
        return True
    
    def find(self, x):
        # O(alpha(n)) = O(1)
        # return x's parent 
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]    
```

# 7. Minimum Spanning Tree algorithm
A minimum spanning tree (MST) or minimum weight spanning tree for a **weighted, connected, undirected graph** is a spanning tree with a weight less than or equal to the weight of every other spanning tree. 


```python
edges1 = [(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)]
edges2 = [(0, 1, 2), (0, 3, 6), (1, 2, 3), (1, 3, 8), (1, 4, 5), (2, 4, 7), (3, 4, 9)]
```

## 7.1 Kruskal's MST


```python
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n
        self.component = n
    
    def union(self, x, y):

        px = self.find(x)
        py = self.find(y)

        if px == py:
            return False
        
        if self.rank[px] > self.rank[py]:
            self.parent[py] = px
            self.rank[px] += self.rank[py]
        elif self.rank[px] < self.rank[py]:
            self.parent[px] = py
            self.rank[py] += self.rank[px]
        else:
            self.parent[py] = px
            self.rank[px] += 1
        
        self.component -= 1
        return True

    def find(self, x):

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
```


```python
def kruskal_mst(vertices, edges):
    
    # O(e*(log(e) + alpha(n))) & O(n)

    edges.sort(key=lambda x:x[2]) # O(e*loge)
    
    uf = UnionFind(vertices) 
    
    res = []
    minCost = 0
    for u, v, w in edges: # O(e*alpha(n))

        if uf.union(u, v):
            minCost += w
            res.append([u, v, w])

            if uf.component == 1:
                return minCost, res
    return -1, None

print(kruskal_mst(4, edges1))
print(kruskal_mst(5, edges2))
```

    (19, [[2, 3, 4], [0, 3, 5], [0, 1, 10]])
    (16, [[0, 1, 2], [1, 2, 3], [1, 4, 5], [0, 3, 6]])


## 7.2 Prim’s MST


```python
def prim_mst(n, edges):
    # O(n^2) & O(n^2)
    graph = [[0] * n for _ in range(n)]
    for u, v, w in edges:
        graph[u][v] = w
        graph[v][u] = w

    parent = [None] * n
    parent[0] = -1

    key = [float('inf')] * n
    key[0] = 0

    mstSet = [False] * n

    res = []
    minCost = 0    
    for _ in range(n):

        # find min_key(minkey), min_node(u)
        u = -1
        minkey = float('inf')
        for v in range(n):
            if not mstSet[v] and key[v] < minkey:
                minkey = key[v]
                u = v

        mstSet[u] = True
        minCost += minkey
        res.append([parent[u], u, minkey])

        # update u's neighbors' key
        for v in range(n):
            if not mstSet[v] and 0 < graph[u][v] < key[v]:
                key[v] = graph[u][v]
                parent[v] = u

    return minCost, res[1:]
        
print(prim_mst(4, edges1))
print(prim_mst(5, edges2))
```

    (19, [[0, 3, 5], [3, 2, 4], [0, 1, 10]])
    (16, [[0, 1, 2], [1, 2, 3], [1, 4, 5], [0, 3, 6]])

[<-PREV](dsa.md)
