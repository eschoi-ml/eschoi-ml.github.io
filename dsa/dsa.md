---
layout: default
title: Eunsoo's Data Science Guide
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
- Tree: Directed, Connected, Acyclic, Unweighted graph
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

- Depth First Search algorithm 
- Breath First Search algorithm 
- Topological Sort (Sort)
- Dijkstra algorithm (Greedy)
- Bellman-Ford algorithm (Dynamic programming)
- Union Find (Greedy)
- Minimum Spanning Tree algorithm (Greedy)


> **Code along Lecture Note: [DFS, BFS, Topological sort, Dijkstra, Bellman-Ford, Union Find, and MST algorithms](search.md)**

## Greedy algorithm

Looks for locally optimum solutions in the hopes of finding a global optimum

- Maximum number of activity selelction
- Maximum profit in job sequencing with deadlines
- Generate egyption fraction
- Maximum value in Fractional Knapsack
- Minimum empty space on wall by fitting shelves
- Minimum number of coloring graph
- Minimum number of platforms
- Minimum number of meeting rooms
- Minimum time to assign mice to holes
- Minimum number of swaps to make the string balanced
- Maximum number that policemen catch thieves
- Water connection (dfs)
- Huffman Coding

> **Code along Lecture Note: [Standard Greedy Algorithm](standard_greedy.md)**

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
