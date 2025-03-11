class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        N = len(s)
        ans = ""
        position = 0
        while position < N:
            nx = s[position : position + k]
            ans = ans + nx[::-1] + s[position + k : position + 2 * k]
            position += 2 * k
        return ans

    # def reverseStr(self, s: str, k: int) -> str:
    #     s = list(s)
    #     for i in range(0, len(s), 2*k):
    #         s[i:i+k] = reversed(s[i:i+k])
    #     return "".join(s)

        

s1 = Solution()
s="abcdefg"
k=2
print(s1.reverseStr(s,k))

# Modified on 2024-09-01 14:19:56.121879

# Modified on 2024-09-15 22:35:08.750907

# Modified on 2024-10-22 18:40:55.661813

# Modified on 2024-11-13 18:05:27.146782

# Modified on 2024-12-20 17:19:47.582337

# Modified on 2025-02-20 23:27:00.910114

# Modified on 2025-03-23 16:38:13.916287
