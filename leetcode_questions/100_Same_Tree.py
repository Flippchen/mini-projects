# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p == q:
            return True
        try:
            left = right = True
            if p.val == q.val:
                left = self.isSameTree(p.left, q.left)
                right = self.isSameTree(p.right, q.right)
                return (left and right)
        except:
            return False
        return False
# Modified on 2024-09-01 14:19:55.863295

# Modified on 2024-09-15 22:35:08.540433

# Modified on 2024-09-15 22:40:27.359944

# Modified on 2024-10-06 12:22:15.287206

# Modified on 2024-10-22 18:40:55.477671

# Modified on 2024-11-13 17:57:22.227407

# Modified on 2024-11-13 18:05:26.966839

# Modified on 2024-12-20 17:19:47.333963

# Modified on 2025-02-20 23:27:00.669317

# Modified on 2025-03-23 16:38:13.677327
