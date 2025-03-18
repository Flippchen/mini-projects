# class Queue(object):
#     def __init__(self):
#         """
#         initialize your data structure here.
#         """
#         self.stack1 = []
#         self.stack2 = []
#
#
#     def push(self, x):
#         """
#         :type x: int
#         :rtype: nothing
#         """
#         while len(self.stack1) > 0:
#             curr = self.stack1.pop()
#             self.stack2.append(curr)
#         self.stack1.append(x)
#         while len(self.stack2) > 0:
#             curr = self.stack2.pop()
#             self.stack1.append(curr)
#
#     def pop(self):
#         """
#         :rtype: nothing
#         """
#         self.stack1.pop()
#
#
#     def peek(self):
#         """
#         :rtype: int
#         """
#         return self.stack1[-1]
#
#     def empty(self):
#         """
#         :rtype: bool
#         """
#         return len(self.stack1) == 0

class Queue(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack1 = []
        self.stack2 = []


    def push(self, x):
        """
        :type x: int
        :rtype: nothing
        """
        self.stack1.append(x)

    def pop(self):
        """
        :rtype: nothing
        """
        if len(self.stack2) == 0:
            while len(self.stack1):
                curr = self.stack1.pop()
                self.stack2.append(curr)
        self.stack2.pop()


    def peek(self):
        """
        :rtype: int
        """
        if len(self.stack2) == 0:
            while len(self.stack1):
                curr = self.stack1.pop()
                self.stack2.append(curr)
        return self.stack2[-1]

    def empty(self):
        """
        :rtype: bool
        """
        return len(self.stack1) + len(self.stack2) == 0
# Modified on 2024-09-01 14:19:56.295251

# Modified on 2024-11-13 18:05:27.312215

# Modified on 2024-12-20 17:19:47.815862

# Modified on 2025-02-20 23:27:01.129629

# Modified on 2025-03-23 16:38:14.121281
