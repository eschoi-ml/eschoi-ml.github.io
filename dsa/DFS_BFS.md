[<- PREV](dsa.md)

# DFS & BFS

## 1. Depth First Search
### 1.1 Depth First Search
Problem: Return True or False if you can reach a target node(F) from a starting node (A)
  - Recursive
  - Iterative

### 1.2 Depth First **Path** Search
Problem: Return the path from a starting node (A) to a target node(F) 
- Find a single path
  - singlePath_recursive
  - singlePath_iterative 
 
- Find all paths: **Backtracking** 
  - allPath_recursive

## 2. Breath First Search
### 2.1 Breath Frist Search - Iterative
Problem: Return the depth if you can reach a target node(F) from a starting node (A). If not, return -1
### 2.2 Breath First **Path** Search - Iterative
Problem: Return the path from a starting node (A) to a target node(F)


-----
# 0. Graph initialization

```python
def print_graph(graph):
    for node in graph:
        print(f"{node}:{graph[node]}")
    return 
```


```python
graph = {
    'A' : ['B', 'C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : ['F'],
    'F' : []
}
print_graph(graph)
```

    A:['B', 'C']
    B:['D', 'E']
    C:['F']
    D:[]
    E:['F']
    F:[]


```python
def reconstruct_path(node1, node2, previous):
    
    node = node2
    path = [node]
    while  node!= node1:
        node = previous[node]
        path.append(node)
    return "->".join(path[::-1])   
```

# 1.1 DFS recursive & iterative


```python
def dfs(graph, start_node, target_node):
    
    def dfs_recursive(curr, target):

        if curr == target:
            return True
        
        for neighbor in graph[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                if dfs_recursive(neighbor, target):
                    return True
        return False
    
    visited = set(start_node)
    print("dfs recursive: ", dfs_recursive(start_node, target_node))
    
    def dfs_iterative(root, target):

        stack = []
        visited = set()

        stack.append(root)
        visited.add(root)

        while stack:

            node = stack.pop()

            if node == target:
                return True
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return False
    
    print("dfs iterative: ", dfs_iterative(start_node, target_node))           

dfs(graph, 'A', 'F')
```

    dfs recursive:  True
    dfs iterative:  True


# 1.2 DFS **Path** recursive & iterative


```python
def dfs_path(graph, start_node, target_node):

    def dfs_singlePath_recursive(curr, target, path):

        if curr == target:
            res[:] = path
            return True

        for neighbor in graph[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                if dfs_singlePath_recursive(neighbor, target, path + [neighbor]):
                    return True
        return False

    res = []
    visited = set()
    dfs_singlePath_recursive(start_node, target_node, [start_node])
    print('dfs_singlePath_recursive: ', res)

    def dfs_singlePath_iterative(start_node, target_node):

        res = []

        previous = {node:None for node in graph}
        stack = []
        visited = set()

        stack.append(start_node)
        visited.add(start_node)

        while stack:
            node = stack.pop()
            if node == target_node:
                res.append(reconstruct_path(start_node, target_node, previous))
                return res

            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    previous[neighbor] = node
                    stack.append(neighbor)
        return res
    print('dfs_singlePath_iterative: ', dfs_singlePath_iterative(start_node, target_node))


    def dfs_allPath_recursive(curr, target, path): # backtracking 

        if curr == target:
            res.append(path)
            return 

        for neighbor in graph[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                dfs_allPath_recursive(neighbor, target, path + [neighbor])
                visited.remove(neighbor)

    res = []
    visited = set([start_node])
    dfs_allPath_recursive(start_node, target_node, [start_node])
    print('dfs_allPath_recursive: ', res)

dfs_path(graph, 'A', 'F')
```

    dfs_singlePath_recursive:  ['A', 'B', 'E', 'F']
    dfs_singlePath_iterative:  ['A->C->F']
    dfs_allPath_recursive:  [['A', 'B', 'E', 'F'], ['A', 'C', 'F']]


# 2.1 BFS - Iterative 1 & 2



```python
from collections import deque

def bfs(graph, start_node, target_node):

    def bfs_iterative1(graph, start_node, target_node):
        queue = deque()
        visited = set()

        queue.append((start_node, 0))
        visited.add(start_node)

        while queue:
            node, depth = queue.popleft()
            if node == target_node:
                return depth
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return depth
    print("bfs_iterative1: ", bfs_iterative1(graph, start_node, target_node))
    
    def bfs_iterative2(graph, start_node, target_node):
        queue = deque()
        visited = set()

        queue.append(start_node)
        visited.add(start_node)

        depth = -1

        while queue:

            depth += 1
            size = len(queue)

            for i in range(size):
                node = queue.popleft()
                if node == target_node:
                    return depth
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return -1
    print("bfs_iterative2: ", bfs_iterative2(graph, start_node, target_node))

bfs(graph, 'A', 'F')
```

    bfs_iterative1:  2
    bfs_iterative2:  2


## 2.2 BFS **Path** - Iterative


```python
from collections import deque

def bfs_path(graph, start_node, target_node):

    def bfs_path_iterative1(graph, start_node, target_node):
        
        queue = deque()
        visited = set()
        previous = {node: None for node in graph}

        queue.append((start_node, 0))
        visited.add(start_node)

        while queue:
            node, depth = queue.popleft()
            if node == target_node:
                return reconstruct_path(start_node, target_node, previous)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    previous[neighbor] = node
                    queue.append((neighbor, depth + 1))
        return None
    print("bfs_path_iterative1: ", bfs_path_iterative1(graph, start_node, target_node))
    
    def bfs_path_iterative2(graph, start_node, target_node):
        queue = deque()
        visited = set()
        previous = {node:None for node in graph}

        queue.append(start_node)
        visited.add(start_node)

        depth = -1

        while queue:

            depth += 1
            size = len(queue)

            for i in range(size):
                node = queue.popleft()
                if node == target_node:
                    return reconstruct_path(start_node, target_node, previous)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        previous[neighbor] = node
                        queue.append(neighbor)
        return -1
    print("bfs_path_iterative2: ", bfs_path_iterative2(graph, start_node, target_node))

bfs_path(graph, 'A', 'F')
```

    bfs_path_iterative1:  A->C->F
    bfs_path_iterative2:  A->C->F

[<- PREV](dsa.md)
