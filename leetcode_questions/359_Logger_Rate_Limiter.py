# class Logger(object):

#     def __init__(self):
#         """
#         Initialize your data structure here.
#         """
#         self.timestamps = {}

#     def shouldPrintMessage(self, timestamp, message):
#         """
#         Returns true if the message should be printed in the given timestamp, otherwise returns false.
#         If this method returns false, the message will not be printed.
#         The timestamp is in seconds granularity.
#         :type timestamp: int
#         :type message: str
#         :rtype: bool
#         """
#         if timestamp < self.timestamps.get(message, 0):
#             return False
#         self.timestamps[message] = timestamp + 10
#         return True


import heapq


class Logger(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.heap = []
        self.cache = {}

    def shouldPrintMessage(self, timestamp, message):
        """
        Returns true if the message should be printed in the given timestamp, otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity.
        :type timestamp: int
        :type message: str
        :rtype: bool
        """
        while len(self.heap):
            if self.heap[0][0] <= timestamp:
                temp = heapq.heappop(self.heap)
                self.cache.pop(temp[1])
            else:
                break
        if timestamp < self.cache.get(message, 0):
            return False
        self.cache[message] = timestamp + 10
        heapq.heappush(self.heap, (timestamp + 10, message))
        return True


# Your Logger object will be instantiated and called as such:
# obj = Logger()
# param_1 = obj.shouldPrintMessage(timestamp,message)

# Modified on 2024-09-01 14:19:56.243665

# Modified on 2024-11-13 18:05:27.269557

# Modified on 2024-12-20 17:19:47.746384

# Modified on 2025-02-20 23:27:01.064851

# Modified on 2025-03-23 16:38:14.059760
