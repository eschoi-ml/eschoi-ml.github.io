# Design
- [System](#system)
- [Cache](#cache)
- [Tic Tac Toe](#tic-tac-toe)

# System
- [Design In-Memory File System](https://leetcode.com/problems/design-in-memory-file-system/): Trie
- [Design Log Storage System](https://leetcode.com/problems/design-log-storage-system/): dict & binary search

## [Design In-Memory File System](https://leetcode.com/problems/design-in-memory-file-system/)


```python
# Trie
class Node:
    def __init__(self):
        self.directory = collections.defaultdict(Node)
        self.content = ""
        
class FileSystem:

    def __init__(self):
        self.root = Node()
    
    def _find(self, path):
        
        if path == '/':
            return self.root
        
        node = self.root
        for p in path.split('/')[1:]:
            node = node.directory[p]
        return node
        
    def ls(self, path: str) -> List[str]:
        
        node = self._find(path)
        if node.content:
            return [path.split('/')[-1]]
        
        return sorted(node.directory.keys())
                        

    def mkdir(self, path: str) -> None:
        
        self._find(path)

        
    def addContentToFile(self, filePath: str, content: str) -> None:
        
        node = self._find(filePath)
        node.content += content
 

    def readContentFromFile(self, filePath: str) -> str:
        
        node = self._find(filePath)
        return node.content

```

## [Design Log Storage System](https://leetcode.com/problems/design-log-storage-system/)


```python
class LogSystem:

    def __init__(self):
        
        # {timestamp:id}
        self.dic = dict()
        # [timestamp1, timestamp2] as sorted
        self.ts = []
        
        self.gmap = {'Year': 4, 'Month': 7, 'Day': 10, 'Hour': 13, 'Minute': 16, 'Second': 19}
        self.start = '2000:00:00:00:00:00'
        self.end = '2017:12:31:23:59:59'

    def put(self, id: int, timestamp: str) -> None:
        # O(log n) & O(n)
        self.dic[timestamp] = id 
        bisect.insort_left(self.ts, timestamp)        
        
        
    def retrieve(self, start: str, end: str, granularity: str) -> List[int]:
        
        # O(log n), O(1)
        
        idx = self.gmap[granularity]
        
        start = start[:idx]  + self.start[idx:]
        end = end[:idx] + self.end[idx:]
        
        l = bisect.bisect_left(self.ts, start)
        r = bisect.bisect_right(self.ts, end)
        
        res = []
        for i in range(l, r):
            res.append(self.dic[self.ts[i]])
        
        return res
```

# Cache
- [LRU Cache](https://leetcode.com/problems/lru-cache/): OrderedDict
- [LFU Cache](https://leetcode.com/problems/lfu-cache/): dict, defaultdict(deque) 

## [LRU Cache](https://leetcode.com/problems/lru-cache/)


```python
# OrderedDict
class LRUCache:

    def __init__(self, capacity: int):
        
        self.dic = collections.OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        
        if key not in self.dic:
            return -1
        
        self.dic.move_to_end(key)
        return self.dic[key]
        
    def put(self, key: int, value: int) -> None:
        
        if key in self.dic:
            self.dic.move_to_end(key)
        self.dic[key] = value
        if len(self.dic) > self.capacity:
            self.dic.popitem(last=False)
```

## [LFU Cache](https://leetcode.com/problems/lfu-cache/)


```python
# dict, defaultdict(deque)
class LFUCache:

    def __init__(self, capacity: int):
        
        # {key:(val, freq)}
        self.key2node = {} 
        # {freq: [key1, key2, key3]}
        self.freq2key = collections.defaultdict(collections.deque) 
        self.minfreq = 0
        
        self.capacity = capacity
    
    def _update(self, key, val):
        
        _, freq = self.key2node[key]
        self.key2node[key] = [val, freq + 1]
        self.freq2key[freq].remove(key)
        if not self.freq2key[freq]:
            del self.freq2key[freq]
        self.freq2key[freq + 1].append(key)
        
        if not self.freq2key[self.minfreq]:
            self.minfreq += 1
        
    def get(self, key: int) -> int:
        
        if key not in self.key2node:
            return -1
        
        val = self.key2node[key][0]
        self._update(key, val)
            
        return val
        
    def put(self, key: int, value: int) -> None:
        
        if self.capacity == 0:
            return
        
        if key in self.key2node:
            self._update(key, value)
            return 
        
        if len(self.key2node) == self.capacity:
            oldkey = self.freq2key[self.minfreq].popleft()
            if not self.freq2key[self.minfreq]:
                del self.freq2key[self.minfreq]  
            del self.key2node[oldkey]

        self.minfreq = 1
        self.key2node[key] = [value, 1]
        self.freq2key[1].append(key)
```

# Tic Tac Toe
- [Design Tic-Tac-Toe](https://leetcode.com/problems/design-tic-tac-toe/)
- [Find Winner on a Tic Tac Toe Game](https://leetcode.com/problems/find-winner-on-a-tic-tac-toe-game/)
- [Valid Tic-Tac-Toe State](https://leetcode.com/problems/valid-tic-tac-toe-state/)


## [Design Tic-Tac-Toe](https://leetcode.com/problems/design-tic-tac-toe/)


```python
class TicTacToe:

    def __init__(self, n: int):
        """
        Initialize your data structure here.
        """
        
        self.rows = [0] * n
        self.cols = [0] * n
        self.diag = [0] * 2
        self.n = n
    
    def _winCheck(self, dic, key, player):
        
        playerval = 2 * (player - 1.5)
        return dic[key] == playerval * self.n

    def move(self, row: int, col: int, player: int) -> int:
        """
        Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins.
        """

        playerval = 2 * (player - 1.5)
        
        self.rows[row] += playerval
        self.cols[col] += playerval
        
        if row - col == 0:
            self.diag[0] += playerval
        if row + col == self.n - 1:
            self.diag[1] += playerval
        
        
        if self._winCheck(self.rows, row, player) or self._winCheck(self.cols, col, player) or self._winCheck(self.diag, 0, player) or self._winCheck(self.diag, 1, player):
            return player
        
        return 0
```

## [Find Winner on a Tic Tac Toe Game](https://leetcode.com/problems/find-winner-on-a-tic-tac-toe-game/)


```python
class Solution:
    def tictactoe(self, moves: List[List[int]]) -> str:
        
        def checkBingo(player):
            
            playerval = 2 * (player - 0.5)
            for i in range(n):
                if rows[i] == playerval * n or cols[i] == playerval * n:
                    return True
            return diag[0] == playerval * n or diag[1] == playerval * n
        
        n = 3
        rows = [0] * n 
        cols = [0] * n
        diag = [0] * 2 # r == c, r + c == n-1
        
        for i in range(len(moves)):
            player = i % 2
            playerval = 2*(player - 0.5)
            r, c = moves[i][0], moves[i][1]
            rows[r] += playerval
            cols[c] += playerval
            if r == c:
                diag[0] += playerval
            if r + c == n - 1:
                diag[1] += playerval
        
        if checkBingo(0):
            return 'A'
        elif checkBingo(1):
            return 'B'
        elif len(moves) == n*n:
            return 'Draw'
        else:
            return 'Pending'
        
```

## [Valid Tic-Tac-Toe State](https://leetcode.com/problems/valid-tic-tac-toe-state/)


```python
class Solution:
    def validTicTacToe(self, board: List[str]) -> bool:
        """
        # nx, no
        # bx, bo
        
        if (nx == no and not bx) or (nx == no + 1 and not bo):
            return True
        return False
        
        
        """
        def checkBingo(player):
            
            # row, col 0 to n-1
            diagcnt = antidiagcnt = 0
            for i in range(n):
                
                rowcnt = colcnt = 0
                for j in range(n):   
                    if board[i][j] == player:
                        rowcnt += 1
                    if board[j][i] == player:
                        colcnt += 1
                if rowcnt == n or colcnt == n:
                    return True
                
                if board[i][i] == player:
                    diagcnt += 1
                
                if board[i][n-1-i] == player:
                    antidiagcnt += 1
                    
            return (diagcnt == n) or (antidiagcnt == n)
   

        n = len(board)
        
        # nx, no
        nx = no = 0
        for i in range(n):
            for j in range(n):
                if board[i][j] == 'O':
                    no += 1
                elif board[i][j] == 'X':
                    nx += 1
        # bx, bo
        bx = bo = False
        if nx >= n:
            bx = checkBingo('X')
        if no >= n:
            bo = checkBingo('O')
        
        if (nx == no and not bx) or (nx == no + 1 and not bo):
            return True
        return False
```
