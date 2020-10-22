# 题目总结如下：

掌握了二叉树的层序遍历， 只需要一个模板， 就可以撸掉下面的七八个题目。 模板再放一边：

```python
def levelOrder(self, root: TreeNode) -> List[List[int]]:

        if not root:
            return []

        res = []
        d = deque([root])
        while d:
            size = len(d)       # 获取当前层的节点个数
            level = []

            # 遍历当前层的所有节点
            for _ in range(size):
                root = d.popleft()
                level.append(root.val)
                if root.left:
                    d.append(root.left)
                if root.right:
                    d.append(root.right)
            
            res.append(level)

        return res
```

下面是具体题目。

## 1. [二叉树的层序遍历II]()



## 2. [LeetCode103之字形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)



## 3. [找每一层的最右结点](https://leetcode-cn.com/problems/binary-tree-right-side-view/)



## 4. [计算每一层的最大值](https://leetcode-cn.com/problems/find-largest-value-in-each-tree-row/)



## 5. [计算每一层的平均值](https://leetcode-cn.com/problems/average-of-levels-in-binary-tree/)



## 6.[N叉树的前序遍历]()



## 7. [填充每个节点的下一个右侧节点指针]()



## 8. [填充每个节点的下一个右侧节点指针II]()



[题解](https://leetcode-cn.com/problems/find-largest-value-in-each-tree-row/solution/er-cha-shu-ceng-xu-bian-li-deng-chang-wo-yao-da--3/)