class Solution(object):
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        # https://leetcode.com/discuss/111582/java-simple-easy-understand-solution-with-explanation
        # in Python this problem is much different because of the negative number
        # https://leetcode.com/discuss/111705/one-positive-one-negative-case-successful-for-python-rules
        import ctypes
        sum = 0
        carry = ctypes.c_int32(b)
        while carry.value != 0:
            sum = a ^ carry.value
            carry = ctypes.c_int32(a & carry.value)
            carry.value <<= 1
            a = sum
        return sum
# Modified on 2024-09-01 14:19:56.344425

# Modified on 2024-11-13 18:05:27.377826

# Modified on 2024-12-20 17:19:47.881691

# Modified on 2025-02-20 23:27:01.187360

# Modified on 2025-03-23 16:38:14.179975
