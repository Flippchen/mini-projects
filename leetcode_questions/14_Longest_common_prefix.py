class Solution:
    def longestCommonPrefix(self, strs):
        if not strs:
            return ""

        # Find the shortest string in the list
        shortest = min(strs, key=len)

        # Initialize the longest common prefix as the entire shortest string
        for i, char in enumerate(shortest):
            for other in strs:
                # Compare characters from other strings at the same position
                if other[i] != char:
                    # If mismatch, return the substring from the start to the current index
                    return shortest[:i]
        return shortest