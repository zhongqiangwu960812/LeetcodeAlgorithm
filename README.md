# LeetcodeAlgorithm

这个仓库记录一下在LeetCode上的刷题过程， 算法和数据结构的知识点，刷题的一些知识总结等。 本次刷题和整理按照知识点由简单到困难的顺序进行， 并且是分知识点进行刷题， 这样做的好处是可以把知识点切碎 然后通过刻意练习的方式把每一块都学习好。 
![](img/1.png)

# 目录大纲：

1. [数组、链表和跳表](https://github.com/zhongqiangwu960812/LeetcodeAlgorithm/tree/master/Class1_ArrayAndLinked)
2. [栈和队列(优先队列， 双端队列)](https://github.com/zhongqiangwu960812/LeetcodeAlgorithm/tree/master/Class2_StackAndQueue)
3. [ 哈希表、映射和集合](https://github.com/zhongqiangwu960812/LeetcodeAlgorithm/tree/master/Class3_Hash_Map_Collections)
4. 树、二叉树、二叉搜索树
5. 泛型递归、树的递归
6. 分治和回溯
7. 深度优先和广度优先
8. 贪心算法
9. 二分查找
10. 动态规划
11. 字典树和并查集
12. 高级搜索
13. 红黑树和AVL树
14. 位运算
15. 布隆过滤和LRU缓存
16. 排序算法
17. 高级动态规划
18. 字符串算法
19. 大串讲
20. 其他题目和总结

# 开始之前， 需要知道的 

## 1. 如何做到精通一个领域（三步走）

1. Chunk it up   把知识点切碎

   庖丁解牛、脉络连接（通过思维导图的方式把知识点连接起来）

2. Deliberate Practicing   刻意练习

3. Feedback反馈

## 2. 分解数据结构和算法（第一步）

### 2.1. 数据结构概览：

- 一维：
  - 基础： 数组Array, string, 链表Linked list
  - 高级： 栈stack, 队列queue, 双端队列deque, 集合set， 映射map(hash)
- 二维：
  - 基础： 树Tree， 图graph
  - 高级： 二级搜索树binary search tree(red-blace tree, AVL), 堆heap, 并查集disjoint set, 字典树Trie
- 特殊：
  - 位运算 Bitwise, 布隆过滤器BloomFilter
  - LRU Cache

注意， 在后面的学习中要了解每个数据结构的原理和代码框架， 这个网站会有三张思维导图， 等把训练营学完了， 也要亲自画一遍。[https://blog.csdn.net/leacock1991/article/details/103333312](https://blog.csdn.net/leacock1991/article/details/103333312)

### 2.2 算法概览

- if-else, switch—>branch
- for, while loop —> Iteration
- 递归Recursion(Divide $ Conquer, Backtrace)
- 搜索Search: 深度优先搜索 Depth first search， 广度优先搜索 Breadth first search， A*
- 动态规划 Dynamic Programming
- 二分查找 Binary Search
- 贪心 Greedy
- 数学 Math， 几何 Geometry

注意： 在头脑中回忆上面每种算法的思想和代码模板

## 3. 刻意练习 — 过遍数

切题四件套：

1. Clearification（弄清楚题目）

2. **Possible solutions（important)**

   想可能的解法， 这个非常重要， 一定要把所有的解题方法先过一遍， 同时分析它们的时间和空间复杂度， 然后选择一个最优的

3. Coding（多写）

4. Test cases   自己想几个测试用例， 找自己程序的漏洞， 把问题想全面一些

五步刷题法（五毒神掌）：

1. 刷题第一遍：

   - 五分钟： 读题+思考
   - 如果不会，直接看解法： 注意多种解法， 比较解法优劣
   - **背诵， 默写好的解法（这个很重要）， 把好的几个解法默写背诵**

2. 刷题第二遍：

   - 马上自己写 —> 并提交查看结果

     把所有好的解法都尝试一遍， 并且必须写到LeetCode上全部通过

   - 多种解法比较， 看执行用时和内存消耗， 体会和尝试优化算法

3. 刷题第三遍

   - 过了一天后， 再重复做题
   - 不同解法的熟练程度 —> 专项练习

4. 刷题第四遍： 过了一周， 反复回来练习相同的

# 3 整理一些刷题比较好的学习资料或者网站
这些可以帮助我们速查速学， 快速掌握一些刷题的技巧， 思想和经验
* [这里是一个刷题目录，大约整理了200多道精华题目，可以学完上面的知识之后，按照这里面的分类刷题](https://github.com/CyC2018/CS-Notes/blob/master/notes/Leetcode%20%E9%A2%98%E8%A7%A3%20-%20%E7%9B%AE%E5%BD%95.md)
* [B 栈up 花花酱](https://space.bilibili.com/9880352?fr...)
* [小Q刷题](https://space.bilibili.com/149758?fro...)
