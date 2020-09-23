# 题目整理

 ## 1. [验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)(中等)

这个题目又两个思路， 一个是递归， 一个是中序遍历， 中序遍历的解法是利用了二叉搜索树的性质**中序遍历递增**。 这个题目的递归解法可以参考递归那一节的题目整理， 这里只给出中序遍历的思路， 因为想通过这个延伸几个题目：

对于二叉搜索树来说， 有个非常重要的性质**二叉搜索树的中序遍历递增**， 基于这个思路， 我们就可以通过中序遍历一遍二叉树， 如果遍历途中发现当前节点的值比前面节点的值小， 那么就不是一棵二叉搜索树。

<img src="img/4.gif" style="zoom:50%;" />

那么梳理一下这个逻辑过程， 我们需要在遍历过程中看当前节点与前一节点的值的大小， 就需要用pre记录前一个节点，所以步骤如下：

1. 首先， pre=-inf
2. 然后中序遍历， 也就是先去左子树， 这时候开始递归
3. 当前逻辑就是， 如果左子树不是二叉搜索树， 返回False， 如果当前值小于pre， 返回False， **然后把pre更新为当前值**
4. 去递归右子树

递归代码如下：

```python 
class Solution:
    def __init__(self):
        self.pre = float('-inf')

    def isValidBST(self, root: TreeNode) -> bool:

        # 结束条件
        if not root:
            return True
        
        # 判断左子树是否是二叉搜索树
        if not self.isValidBST(root.left):    # 如果左子树不是二叉搜索树
            return False
        
        if root.val <= self.pre:
            return False
        
        self.pre = root.val          # 这里要更新pre才能去看右子树
        return self.isValidBST(root.right)

```

非递归代码， 这里也写一个非递归代码， 顺便复习一下非递归的中序遍历：

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:


        stack = []
        pre = float('-inf')

        while root or stack:

            # 去找最左边的节点
            while root:
                stack.append(root)
                root = root.left
            
            root = stack.pop()
            if root.val <= pre:
                return False
            pre = root.val

            # 去右边
            root = root.right
        return True
```

下面基于这个题目延伸一个题目

1. 在二叉搜索树中找最小的的第K个元素

   ```python
   class Solution:
       def isValidBST(self, root: TreeNode, k:int) -> int:
   
           stack = []
           res = 0
           while root or stack:
   
               # 去找最左边的节点
               while root:
                   stack.append(root)
                   root = root.left
               
               root = stack.pop()
               k -= 1
               if k == 0:
                   res = root.val
                   break
   
               # 去右边
               root = root.right
           return res
   ```

   