[<-PREV](dsa.md)

# Binary Search Tree

- Inorder traversal of BST is an array sorted in the ascending order.
- Successor = "after node", i.e. the next node, or the smallest node after the current one.
- Predecessor = "before node", i.e. the previous node, or the largest node before the current one. [[1]](https://leetcode.com/problems/delete-node-in-a-bst/solution/)

## [Search in a Binary Search Tree](https://leetcode.com/problems/search-in-a-binary-search-tree/)


```python
def searchBST(self, root: TreeNode, val: int) -> TreeNode:
    
    if not root:
        return None
    
    if root.val == val:
        return root
    
    if root.val > val:
        return self.searchBST(root.left, val)
    
    return self.searchBST(root.right, val)
```

## [Insert into a Binary Search Tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/)


```python
def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
    
    if not root:
        return TreeNode(val)

    if root.val < val:
        root.right = self.insertIntoBST(root.right, val)

    elif root.val > val:
        root.left = self.insertIntoBST(root.left, val)
        
    return root
```

## [Delete Node in a BST](https://leetcode.com/problems/delete-node-in-a-bst/)


```python
def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
    
    if not root:
        return None
    
    if root.val > key:
        root.left = self.deleteNode(root.left, key)
    
    elif root.val < key:
        root.right = self.deleteNode(root.right, key)
        
    else: # root.val == key
        if not root.left and not root.right:
            root = None
        elif root.left and not root.right:
            root = root.left
        elif not root.left and root.right:
            root = root.right
        else:
            # Find a successor
            node = root.right
            while node.left:
                node = node.left
            root.val = node.val
            root.right = self.deleteNode(root.right, root.val)
    return root
```

## [Merge Two Binary Trees](https://leetcode.com/problems/merge-two-binary-trees/)


```python
def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:

    if not root1 and not root2:
        return None
    
    if not root1 or not root2:
        return root1 or root2
    
    root1.val += root2.val
    root1.left = self.mergeTrees(root1.left, root2.left)
    root1.right = self.mergeTrees(root1.right, root2.right)
    
    return root1 
```

## [Split BST](https://leetcode.com/problems/split-bst/)


```python
def splitBST(self, root: TreeNode, target: int) -> List[TreeNode]:
    
    if not root:
        return None, None
    
    if root.val <= target:
        sub = self.splitBST(root.right, target)
        root.right = sub[0]
        return root, sub[1]
    else: # root.val > target
        sub = self.splitBST(root.left, target)
        root.left = sub[1]
        return sub[0], root
```

[<-PREV](dsa.md)
