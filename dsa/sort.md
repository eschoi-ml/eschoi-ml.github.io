[<-PREV](dsa.md)

# Sort
1. Bogo Sort: best O(n), avg O(n*n!) worst O(inf) & O(1), randomly shuffle
1. Bubble Sort: best, avg, worst O(n^2) & O(1), compare arr[j] vs. arr[j+1] with j = 0 to n - 1 - i when 0 <= i <= n-1 
1. **Insertion Sort**: best O(n), avg and worst O(n^2) & O(1), key = arr[i] j = i-1 to 0
1. Shell Sort: best O(nlogn), worst O(n^2) & O(1), extension of **Insertion Sort** with gaps
1. Selection Sort: best, avg, worst O(n^2) & O(1), find the smallest one and place it in the front
1. **Merge Sort**: best, avg, worst O(nlogn) & O(n), divide and conquer
1. **Quick Sort**: best, avg, worst O(nlogn) & O(nlogn) inplace except for call stacks, [elements < pivot] + [pivot] + [elements >= pivot]
1. **Heap Sort**: heapq.heapify()-O(n), heapq.heappush() heapq.heappop()-O(nlogn), a Binary Heap is a Complete Binary Tree where items are stored in a special order such that value in a parent node is greater(max-heap)or smaller(min-heap) than the values in its two children nodes
1. Counting Sort: O(n + range_n) & O(n + range_n), by range(min_ele to max_ele, 10^n), or alphabet, count_arr, output_arr
1. Radix Sort: O(d*(n + b)) & O(n + b), repeat counting sort
1. **Bucket Sort**: worst O(n^2), best O(n+k) & O(n+k), great for uniformly distributed input, use **Insertion Sort** for each bucket
1. **Tim Sort**: O(nlogn), used in Python sorted(), list.sort(), **Insertion Sort** + Merge Sort




```python
arr = [3,2,13,4,6,5,7,8,1,20]
arr2 = [170, 45, 75, 90, 802, 24, 2, 66]
arr3 = [0.899, 0.666, 0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]
print('unsorted arr1: ', arr)
print('unsorted arr2: ', arr2)
print('unsorted arr3: ', arr3)
```

    unsorted arr1:  [3, 2, 13, 4, 6, 5, 7, 8, 1, 20]
    unsorted arr2:  [170, 45, 75, 90, 802, 24, 2, 66]
    unsorted arr3:  [0.899, 0.666, 0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]


## 1. Bogo Sort
Shuffle untile sorted


```python
import random

def bogo_sort(arr):
    def isSorted(arr):
        if len(arr) < 2: return True
        n = len(arr)
        for i in range(n-1):
            if arr[i] > arr[i+1]:
                return False
        return True
    while not isSorted(arr):
        random.shuffle(arr)
    print("Bogo Sort: ", arr)

bogo_sort(arr)

```

    Bogo Sort:  [1, 2, 3, 4, 5, 6, 7, 8, 13, 20]


## 2. Bubble Sort
    o o o o x
    o o o x
    o o x
    o x



```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n-1):
        for j in range(n-1-i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

    print('Bubble Sort: ', arr)

bubble_sort(arr)
```

    Bubble Sort:  [1, 2, 3, 4, 5, 6, 7, 8, 13, 20]


## 3. **Insertion Sort**
    o o o o o 
    o x
    o o x
    o o o x
    o o o o x


```python
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    print('Insertion Sort: ', arr) 

insertion_sort(arr)
```

    Insertion Sort:  [1, 2, 3, 4, 5, 6, 7, 8, 13, 20]


## 4. Shell Sort
Insertion Sort with gaps

    o o o o o o gap=3 
    o     x
      o     x
        o     x
    o x         gap=1
      o x
        o x
          o x
            o x
              o x


```python
def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            key = arr[i]
            j = i - gap
            while j >= 0 and arr[j] > key:
                arr[j + gap] = arr[j]
                j -= gap
            arr[j + gap] = key
        gap //= 2

    print('Shell Sort: ', arr)
shell_sort(arr)
```

    Shell Sort:  [1, 2, 3, 4, 5, 6, 7, 8, 13, 20]


## 5. Selection Sort
Find the smallest one in the front
    
    x o o o o 
    $ x o o o 
    $ $ x o o 
    $ $ $ x o 


```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n-1):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    print('Selection Sort: ', arr)
selection_sort(arr)
```

    Selection Sort:  [1, 2, 3, 4, 5, 6, 7, 8, 13, 20]


## 6. **Merge Sort**
    divide
    o o o o x x x x
    o o x x o o x x
    o x o x o x o x
    conquer
    o-x o-x o-x o-x
    o-x-o-x o-x-o-x
    o-x-o-x-o-x-o-x



```python
def merge_sort(arr):
    
    def merge(arr, left, right):
        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        if i < len(left):
            arr[k:] = left[i:]
        if j < len(right):
            arr[k:] = right[j:]
        return 

    if len(arr)>= 2:
        mid = len(arr)//2
        left = arr[:mid]
        right = arr[mid:]

        merge_sort(left)
        merge_sort(right)

        merge(arr, left, right)
    print('Merge Sort: ', arr)
merge_sort(arr)
```

    Merge Sort:  [1]
    Merge Sort:  [2]
    Merge Sort:  [1, 2]
    Merge Sort:  [3]
    Merge Sort:  [4]
    Merge Sort:  [5]
    Merge Sort:  [4, 5]
    Merge Sort:  [3, 4, 5]
    Merge Sort:  [1, 2, 3, 4, 5]
    Merge Sort:  [6]
    Merge Sort:  [7]
    Merge Sort:  [6, 7]
    Merge Sort:  [8]
    Merge Sort:  [13]
    Merge Sort:  [20]
    Merge Sort:  [13, 20]
    Merge Sort:  [8, 13, 20]
    Merge Sort:  [6, 7, 8, 13, 20]
    Merge Sort:  [1, 2, 3, 4, 5, 6, 7, 8, 13, 20]


## 7. **Quick Sort**
     o o o o o o o o 
    [o o]x[o o o o o]
     x[o] [o]x[o o o]
              [o]x[o]


```python
def quick_sort(arr):

    def partition(arr, l, r):
        # pivot1 = first element
        pivot = arr[l]
        i = l
        for j in range(l+1, r+1):
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[l], arr[i] = arr[i], arr[l]
        return i
        # pivot2 = last element
        pivot = arr[r]
        i = l-1
        for j in range(l, r):
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i+1], arr[r] = arr[r], arr[i+1]
        return i+1
        # pivot3 = random element
        # pivot4 = median element
    
    def _quick_sort(arr, l, r):
        if l < r:
            pivot = partition(arr, l, r)
            _quick_sort(arr, l, pivot-1)
            _quick_sort(arr, pivot + 1, r)
    _quick_sort(arr, 0, len(arr)-1)
    print("Quick Sort: ", arr)

quick_sort(arr)
```

    Quick Sort:  [1, 2, 3, 4, 5, 6, 7, 8, 13, 20]


## 8. **Heap Sort**



```python
def max_heap_sort(arr):

    def heapify(arr, n, i):
        largest = i
        left = 2*i + 1
        right = 2*i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)
        return 

    n = len(arr)
    for i in range(n//2-1, -1, -1):
        heapify(arr, n, i)
    
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    print('Max Heap Sort: ', arr)

max_heap_sort(arr)
```

    Max Heap Sort:  [1, 2, 3, 4, 5, 6, 7, 8, 13, 20]



```python
import heapq
heapq.heapify(arr)
print(arr)
```

    [1, 2, 5, 3, 6, 13, 7, 8, 4, 20]


## 9. Counting Sort
    1 5 4 2 2
    minval, maxval, range_val = 1, 5, 5
        i = 0(1) 1(2) 2(3) 3(4) 4(5)
    count = 1    2    0    1    1
    count = 1    3    3    4    5 
    arr[i] -> idx = arr[i] - minval -> j = count[idx]-1 -> output[j] = arr[i] -> count[idx] -= 1

            


```python
def counting_sort(arr):
    n = len(arr)
    maxval, minval = max(arr), min(arr)
    range_val = maxval - minval + 1
    count = [0] * range_val
    output = [0] * n

    for i in range(n):
        idx = arr[i] - minval
        count[idx] += 1
    for i in range(1, range_val):
        count[i] += count[i-1]
    
    for i in range(n-1, -1, -1):
        idx = arr[i] - minval
        output[count[idx] - 1] = arr[i]
        count[idx] -= 1

    print('Counting Sort: ', output)
counting_sort(arr)
```

    Counting Sort:  [1, 2, 3, 4, 5, 6, 7, 8, 13, 20]


## 10. Radix Sort
Sort 10^0, 10^1, 10^2 ... placement

Example. arr2 = [170, 45, 75, 90, 802, 24, 2, 66]


```python
def radix_sort(arr):
    def counting_sort(arr, exp):
        n = len(arr)
        count = [0] * 10
        output = [0] * n
        for i in range(n):
            idx = (arr[i]//exp) % 10
            count[idx] += 1
        for i in range(1, 10):
            count[i] += count[i-1]
        for i in range(n-1, -1, -1):
            idx = (arr[i]//exp) % 10
            j = count[idx] - 1
            output[j] = arr[i]
            count[idx] -= 1
        return output

    print('Radix Sort: ')

    maxval = max(arr)
    exp = 1
    while maxval // exp > 0:
        arr = counting_sort(arr, exp)
        exp *=10
        print(arr)

radix_sort(arr2)
    
```

    Radix Sort: 
    [170, 90, 802, 2, 24, 45, 75, 66]
    [802, 2, 24, 45, 66, 170, 75, 90]
    [2, 24, 45, 66, 75, 90, 170, 802]


## 11. **Bucket Sort**
Sort into bucket array with Insertion Sort

Example. arr3 = [0.899, 0.666, 0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]


```python
def bucket_sort(arr):

    def insertion_sort(bucket_arr):
        i = len(bucket_arr) - 1
        key = bucket_arr[i]
        j = i - 1
        while j >= 0 and bucket_arr[j] > key:
            bucket_arr[j+1] = bucket_arr[j]
            j -= 1
        bucket_arr[j + 1] = key
    """
    n_slot
    slot_range = (maxval - minval) // nslot
    idx = (val - minval) // slot_range
    """
    n_slot = 10
    bucket = [[] for _ in range(n_slot)]
    for num in arr:
        idx = int(num * 10) % 10
        bucket[idx].append(num)
        insertion_sort(bucket[idx])
    
    arr = []
    for j in range(n_slot):
        arr.extend(bucket[j])
    print('Bucket Sort: ', arr)
bucket_sort(arr3)
```

    Bucket Sort:  [0.1234, 0.3434, 0.565, 0.656, 0.665, 0.666, 0.897, 0.899]

[<-PREV](dsa.md)
