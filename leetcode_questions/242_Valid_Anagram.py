class Solution(object):
    # def isAnagram(self, s, t):
    #     """
    #     :type s: str
    #     :type t: str
    #     :rtype: bool
    #     """
    #     # sort
    #     return sorted(s) == sorted(t)

    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        # hash
        # https://leetcode.com/articles/valid-anagram/
        if len(s) != len(t):
            return False
        counter = [0] * 26
        for i in range(len(s)):
            counter[ord(s[i]) - ord('a')] += 1
            counter[ord(t[i]) - ord('a')] -= 1
        for num in counter:
            if num != 0:
                return False
        return True
# Modified on 2024-09-01 14:19:55.796390

# Modified on 2024-09-15 22:35:08.492840

# Modified on 2024-09-15 22:40:27.318617

# Modified on 2024-10-06 12:22:15.241659

# Modified on 2024-10-22 18:40:55.428024

# Modified on 2024-11-13 17:57:22.181160

# Modified on 2024-11-13 18:05:26.924886
