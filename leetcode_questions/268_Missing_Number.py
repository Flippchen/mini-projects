class Solution(object):
    # def missingNumber(self, nums):
    #     """
    #     :type nums: List[int]
    #     :rtype: int
    #     """
    #     n = len(nums)
    #     return (n ** 2 + n) / 2 - sum(nums)

    def missingNumber(self, nums):
        res = len(nums)
        for i, v in enumerate(nums):
            res ^= i
            res ^= v
        return res
    
    # def missingNumber(self, nums):
    #     nums.sort()
    #     left, right = 0, len(nums) - 1
    #     while left <= right:
    #         mid = (left + right) / 2
    #         if nums[mid] <= mid:
    #             left = mid + 1
    #         else:
    #             right = mid - 1
    #     return left

# Modified on 2024-09-01 14:19:56.194668

# Modified on 2024-11-13 18:05:27.225716

# Modified on 2024-12-20 17:19:47.679100

# Modified on 2025-02-20 23:27:00.999075

# Modified on 2025-03-23 16:38:14.001067
