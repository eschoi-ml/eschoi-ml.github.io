---
layout: default
title: Eun Soo's Machine Learning Guide
description: Data Structure and Algorithm
---

[<- PREV](../README.md)

# What kind of data structure and algorithm are you going to use?
## Data Structure
### Basic data structure
- Array (list)
- Hash Map (dict)
- Hash Set (set)

### Abstract data structure
- Stack (list)
- Simple/ Circular/ Priority/ Double-Ended Queue (deque, heap)
- Singly/ Doubly/ Circular Linked List (list)
- Graph: Directed/Undirected, Connected/ Disconnected, Cyclic/ Acyclic, Weighted/Unweighted
  - (Minimum) Spanning Tree: (Minimum sum of the weight of the edges) Undirected, Connected
- Tree
  - N-ary Tree
  - Binary Tree
  - Binary Search Tree
  - Height-balanced Binary (Search) Tree
  - AVL Tree  
  - Trie
  - Decision Tree
  - B, B+, Red-Black Tree

## Algorithm
Basic categories of algorithms: Insert, Update, Delete, Sort, and Search

### Sort
- bogo sort
- bubble sort
- insertion sort
- shell sort
- selection sort
- merge sort
- quick sort
- heap sort
- counting sort
- radix sort
- bucket sort
- topological sort
- kahn's topological sort


### Search

> **Code along Lecture Note: [Ultimate Binary Search Template](binary_search.md)**

- Linear search
- Binary search

> **Code along Lecture Note: [DFS & BFS](DFS_BFS.md)**

- Depth first search/ Backtracking
- Breath first search
- Bidirectional search (if undirected)

> **Code along Lecture Note: [Shortest Path]()**

- Bellman Ford's algorithm (if negative weighted) 
- Dijkstra's algorithm (if positive weighted) 
- A*(A star) search (Best First Search)

### Greedy algorithm
Looks for locally optimum solutions in the hopes of finding a global optimum
- Ford-Fulkerson algorithm
- Kruskal's minimum spanning tree algorithm
- Prim's minimum spanning tree algorithm
- Huffman coding

### Dynamic programming
Problems that have overlapping subproblems AND optimal substructure property (If not, use a recursive algorithm using a divide and conquer approach)
- Floyd-Warshall algorithm
- Longest Common Sequence

[<- PREV](../README.md)
