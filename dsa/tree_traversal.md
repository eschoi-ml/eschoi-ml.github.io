[<-PREV](dsa.md)

# Tree Traversal
- **dfs**: recursive and iterative using a stack
    - **pre-order**: **root** - left - right
    - **in-order**: left - **root** - right
    - **post-order**: left - right - **root**
- **bfs**: iterative using a queue(deque)
    - **level-order**

## Pre-order tree traversal


```python
def preorder_recursive(node):
    if not node:
        return
    res.append(node.val)
    preorder_recursive(node.left)
    preorder_recursive(node.right)

def preorder_iterative(root):

    if not root:
        return []

    stack = [root]
    res = []
    
    while stack:
        node = stack.pop()
        res.append(node.val)

        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return res
```

## In-order tree traversal


```python
def inorder_recursive(node):
    if not node:
        return 

    inorder_recursive(node.left)
    res.append(node.val)
    inorder_recursive(node.right)

def inorder_iterative(root):

    if not root:
        return None
    
    stack = []
    res = []

    node = root
    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        
        node = stack.pop()
        res.append(node.val)

        node = node.right
    return res
```

## Post-order tree traversal


```python
def postorder_recursive(node):

    if not node:
        return 

    postorder_recursive(node.left)
    postorder_recursive(node.right)
    res.append(node.val)

def postorder_iterative(root):

    if not root:
        return None
    
    stack = []
    res = []

    node = root
    while True:
        while node:
            if node.right:
                stack.append(node.right)
            stack.append(node)
            node = node.left

        node = stack.pop()
        if node.right and stack and stack[-1] == node.right:
            stack.pop()
            stack.append(node)
            node = node.right
        else:
            res.append(node.val)
            node = None
        if not stack:
            break
    
    return res
```

## Level-order tree traversal


```python
def levelorder_iterative1(root):

    if not root:
        return None

    queue = collections.deque([(root, 0)])
    res = []

    while queue:
        node, depth = queue.popleft()
        if len(res) == depth:
            res.append([node.val])
        else:
            res[depth].append(node.val)

        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))
    return res

def levelorder_iterative2(root):

    if not root:
        return None
    
    queue = collections.deque([root])
    res = []
    depth = -1
    
    while queue:

        depth += 1
        size = len(queue)
        temp_res = []
        
        for i in range(size):
            node = queue.popleft()
            temp_res.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(temp_res)
    return res
```
[<-PREV](dsa.md)
