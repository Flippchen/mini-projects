class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        sum_map = {}
        sum_map[0] = 1
        count = curr_sum = 0
        for num in nums:
            curr_sum += num
            # Check if sum - k in hash
            count += sum_map.get(curr_sum - k, 0)
            # add curr_sum to hash
            sum_map[curr_sum] = sum_map.get(curr_sum, 0) + 1
        return count

# Modified on 2024-09-01 14:19:56.269456

# Modified on 2024-11-13 18:05:27.290927

# Modified on 2024-12-20 17:19:47.780483

# Modified on 2025-02-20 23:27:01.097097
