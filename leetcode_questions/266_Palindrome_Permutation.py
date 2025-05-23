class Solution(object):
    def canPermutePalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        dic = {}
        for c in s:
            dic[c] = dic.get(c, 0) + 1
        odd, even = 0, 0
        for c in dic:
            if dic[c] % 2 == 0:
                even += 1
            else:
                odd += 1
        if odd <= 1:
            return True
        return False

# Modified on 2024-09-01 14:19:56.171129

# Modified on 2024-09-15 22:35:08.806563

# Modified on 2024-11-13 18:05:27.200294

# Modified on 2024-12-20 17:19:47.645634

# Modified on 2025-02-20 23:27:00.968955

# Modified on 2025-03-23 16:38:13.972865
