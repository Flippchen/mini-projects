class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        # Cache for dynamic programming
        cache = {}

        def compressed_length(count):
            if count == 0:
                return 0
            elif count == 1:
                return 1
            else:
                return 1 + len(str(count))

        def dp(i, last_char, last_char_count, k):
            # If all characters are processed or used all deletions, return 0
            if k < 0:
                return float('inf')
            if i == len(s):
                return compressed_length(last_char_count)

            # Check if the result is already calculated
            if (i, last_char, last_char_count, k) in cache:
                return cache[(i, last_char, last_char_count, k)]

            # Delete the current character
            delete = dp(i + 1, last_char, last_char_count, k - 1)

            # Keep the current character
            keep = 0
            if s[i] == last_char:
                # If it's the same as the last character, increase the count
                keep = dp(i + 1, last_char, last_char_count + 1, k)
            else:
                # If it's a different character, reset the count to 1
                keep = compressed_length(last_char_count) + dp(i + 1, s[i], 1, k)

            # Save and return the minimum of deleting or keeping
            result = min(delete, keep)
            cache[(i, last_char, last_char_count, k)] = result
            return result

        # Start from the first character with no last character, count of 0, and k deletions available
        return dp(0, "", 0, k)