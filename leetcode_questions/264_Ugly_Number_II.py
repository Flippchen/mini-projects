class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 5:
            return n
        dp = [0] * (n + 1)
        l1 = l2 = l3 = 1
        dp[1] = 1
        dp[2] = 2
        dp[3] = 3
        dp[4] = 4
        dp[5] = 5
        for i in range(6, n + 1):
            while dp[l1] * 2 <= dp[i - 1]:
                l1 += 1
            while dp[l2] * 3 <= dp[i - 1]:
                l2 += 1
            while dp[l3] * 5 <= dp[i - 1]:
                l3 += 1
            print l1, l2, l3
            dp[i] = min(dp[l1] * 2, dp[l2] * 3, dp[l3] * 5)
        # print dp
        return dp[n]

if __name__ == '__main__':
    # begin
    s = Solution()
    print s.nthUglyNumber(10)
# Modified on 2024-09-01 14:19:56.096805

# Modified on 2024-09-15 22:35:08.724274

# Modified on 2024-10-06 12:22:15.428594

# Modified on 2024-10-22 18:40:55.634084

# Modified on 2024-11-13 18:05:27.118603

# Modified on 2024-12-20 17:19:47.546288

# Modified on 2025-02-20 23:27:00.879450

# Modified on 2025-03-23 16:38:13.885893
