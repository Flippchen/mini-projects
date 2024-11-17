class Solution(object):
    def numberOfPatterns(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        used = [False] * 9
        res = 0
        for l in range(m, n + 1):
            res += self.calc_patterns(used, -1, l)
            used = [False] * 9
        return res

    def is_valid(self, used, index, last):
        # markded
        if used[index]:
            return False
        # first digit
        if last == -1:
            return True
        # adjacent cells (in a row or in a column)
        if (last + index) % 2 == 1:
            return True
        mid = (last + index) / 2
        if mid == 4:
            return used[mid]
        # adjacent cells on diagonal
        if (index % 3 != last % 3) and (index / 3 != last / 3):
            return True
        # all other cells which are not adjacent
        return used[mid]

    def calc_patterns(self, used, last, length):
        if length == 0:
            return 1
        res = 0
        for i in range(9):
            if self.is_valid(used, i, last):
                used[i] = True
                res += self.calc_patterns(used, i, length - 1)
                used[i] = False
        return res

# Modified on 2024-09-01 14:19:55.938618

# Modified on 2024-09-15 22:35:08.594879

# Modified on 2024-09-15 22:40:27.402837

# Modified on 2024-10-06 12:22:15.327011

# Modified on 2024-10-22 18:40:55.520579

# Modified on 2024-11-13 17:57:22.277938

# Modified on 2024-11-13 18:05:27.007875

# Modified on 2024-12-20 17:19:47.390449
