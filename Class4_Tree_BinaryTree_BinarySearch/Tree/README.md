# 题目整理

## 1. [N叉树的前序遍历](https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/)(简单)

有了二叉树那边的遍历， N叉树这边也是有递归和非递归， 递归的解法比较简单， 还是二叉树那里的逻辑， 拿到root，我先访问他，然后递归他的孩子节点即可， 和二叉树不同的就是这里的孩子节点不一定只有左右孩子， 是个列表的形式， 所以遍历一圈即可。

```python
class Solution:
    def __init__(self):
        self.res = []

    def preorder(self, root: 'Node') -> List[int]:

        if root:
            self.res.append(root.val)
            for i in root.children:
                self.preorder(i)
        
        return self.res
```

非递归遍历版本：

先理一下思路， 和二叉树那里也差不多， 需要用到栈的方式， 首先也是按照二叉树前序遍历的逻辑， 但和二叉树有些不同， 因为这里是N叉树， 没法按照二叉树的那个写法， 给定一个root， 然后去遍历左子树， 再遍历右子树，所以思路是这样：

* 先定义一个栈， 把root加进去
* 当栈非空的时候， 弹出元素， 然后访问， 这时候要把root的孩子节点**从右到左**进行入栈操作即可

```python
class Solution:
    def __init__(self):
        self.res = []

    def preorder(self, root: 'Node') -> List[int]:
        if not root:
            return []

        stack = []
        stack.append(root)

        while stack:

            # 出栈访问
            root = stack.pop()
            self.res.append(root.val)

            stack.extend(root.children[::-1])
        
        return self.res
```

## 2.[二叉树的后序遍历](https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal/)(简单)

递归版本，不解释

```python
class Solution:
    def __init__(self):
        self.res =[]

    def postorder(self, root: 'Node') -> List[int]:

        if root:
            for ch in root.children:
                self.postorder(ch)
            self.res.append(root.val)
        
        return self.res
```

非递归版本

这里和前序遍历的差不多， 只不过通过观察之后发现， 如果我们拿到了一个根节点root， 然后它的孩子节点是v1, v2, v3, 那么后序遍历的时候就是v1, v2, v3, root。 而如果逆序一下发现root, v3, v2, v1。 这样就会发现， 如果我们先访问根节点， 然后访问它的右孩子，再访问左孩子的方式进行前序遍历， 就会得到后序遍历的逆序了， 那么把结果翻过去输出即可。

```python
class Solution:
    def __init__(self):
        self.res =[]

    def postorder(self, root: 'Node') -> List[int]:

        if not root:
            return []

        stack = []
        stack.append(root)

        while stack:
            root = stack.pop()
            self.res.append(root.val)
            stack.extend(root.children)

        return self.res[::-1]
```

## 3. [N叉树的层序遍历](https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/)(简单)

这个和二叉树的层序遍历基本一致， 需要用队列实现， 且对于每一层需要先统计节点个数， 然后依次遍历每一层的节点，访问即可。

```python
from collections import deque
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:

        if not root:
            return []

        res =[]
        d = deque([root])

        while d:
            level = []
            length = len(d)
            for i in range(length):
                root = d.popleft()
                level.append(root.val)
                d.extend(root.children)
            res.append(level)
        return res
```

下面给出一个层序遍历的递归版本。

递归版本的思路就是拿到一个root， 把这个值加入对应层的结果， 然后进行递归， 有level进行标识那一串。 如果说上面那个是一行一行的把每一层的结果添加到最后的res中， 而下面这个就是一列一列的把最后的结果加到res中。

```python
from collections import deque
class Solution:
    def __init__(self):
        self.res = []
    
    def levelOrder(self, root: 'Node') -> List[List[int]]:

       def traverse_node(node, level):
           if not node:     # 如果当前节点为空， 直接返回
               return 
               
           if len(self.res) == level:     # 这个就是如果换到了新的一层， 先定义一个空列表
               self.res.append([])
                
           self.res[level].append(node.val)
           for ch in node.children:
                traverse_node(ch, level+1)
       if root:
            traverse_node(root, 0)
       return self.res

```

