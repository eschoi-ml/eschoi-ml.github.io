---
layout: default
title: Eun Soo's Machine Learning Guide
description: Data Structure and Algorithm
---

[<- PREV](../README.md)

# Data Type
- Boolean (bool)
- Numbers: integer(int), float(float), complex number(complex)
- String (str)

> **Code along Problem Set: [String](string.md)**

# Data Structure
## Basic data structure
- Array (list, tuple)
- Hash Map (dict)
- Hash Set (set)

> **Code along Problem Set: [Design](design.md)**

## Abstract data structure
- Stack (list)
- Simple/ Circular/ Priority/ Double-Ended Queue (deque, heap)
- Singly/ Doubly/ Circular Linked List (list)

> **Code along Problem Set: [Monotonic Stack](monotonic_stack.md)**

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
  
> **Code along Lecture Note:**
> - **[Tree Traversal](tree_traversal.md)**
> - **[Tree Recursion Top-Down & Bottom-Up](tree_recursion.md)**
> - **[Binary Search Tree](bst.md)**

# Algorithm
Basic categories of algorithms: Insert, Update, Delete, Sort, and Search

## Sort
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

> **Code along Lecture Note: [Sort](sort.md)**

## Search

- Linear search
- Binary search

> **Code along Lecture Note: [Ultimate Binary Search Template](binary_search.md)**

- Depth first search/ Backtracking
- Breath first search

> **Code along Lecture Note: [DFS & BFS](DFS_BFS.md)**

- **DFS** (traverse all nodes or traverse all paths between two nodes when equally or unweighted)
- ***BFS*** (traverse all nodes or shortest path between two nodes when equally or unweighted)
- ***Dijkstra***'s algorithm (shortest path from a single source when eaqually or positively weighted) 
- ***Bellman-Ford***'s algorithm (shortest path from a single source when negatively weighted) 
- Kruskal's minimum spanning tree (**UnionFind**, DisjointSet) and Prim's minimum spanning tree 
- **Kahn**'s topological sort

> **Code along Lecture Note: DFS, BFS, Dijkstra, Bellman-Ford, UnionFind, Topological sort**

## Greedy algorithm

Looks for locally optimum solutions in the hopes of finding a global optimum
- Activity selelction 
- Fractional Knapsack
- Graph coloring
- Job sequencing with deadlines
- Minimum number of platforms/ meeting rooms
- Huffman Coding
- Dijkstra's algorithm for shortest paths from a single source 
- Kruskal's and Prim's minimum spanning tree

> **Code along Lecture Note: [Top 8 Greedy Algorithm](greedy.md)**

## Dynamic programming
Problems that have overlapping subproblems AND optimal substructure property (If not, use a recursive algorithm using a divide and conquer approach)
- Floyd-Warshall algorithm

> 5 DP patterns by [Atalyk Akash](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns)
- **[Pattern 1. Min/Max value](dp_pattern1.md)**
- **[Pattern 2. Dintinct ways](dp_pattern2.md)**
- **[Pattern 3. Merge intervals](dp_pattern3.md)**
- **[Pattern 4. DP in strings](dp_pattern4.md)**
- **[Pattern 5. State machine](dp_pattern5.md)**

[<- PREV](../README.md)
