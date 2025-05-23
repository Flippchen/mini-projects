class Solution(object):
    def ipToInt(self, ip):
        ans = 0
        for x in ip.split('.'):
            ans = 256 * ans + int(x)
        return ans

    def intToIP(self, x):
        return ".".join(str((x >> i) % 256)
                        for i in (24, 16, 8, 0))

    def ipToCIDR(self, ip, n):
        # Start value of IP
        start = self.ipToInt(ip)
        ans = []
        while n:
            # Last 1 of start or can start from 0
            mask = max(33 - (start & -start).bit_length(),
                       33 - n.bit_length())
            ans.append(self.intToIP(start) + '/' + str(mask))
            start += 1 << (32 - mask)
            n -= 1 << (32 - mask)
        return ans

# Modified on 2024-09-01 14:19:55.820508

# Modified on 2024-09-15 22:35:08.514958

# Modified on 2024-09-15 22:40:27.339620

# Modified on 2024-10-06 12:22:15.265111

# Modified on 2024-10-22 18:40:55.452828

# Modified on 2024-11-13 17:57:22.202476

# Modified on 2024-11-13 18:05:26.946073

# Modified on 2024-12-20 17:19:47.305074

# Modified on 2025-02-20 23:27:00.640473

# Modified on 2025-03-23 16:38:13.648900
