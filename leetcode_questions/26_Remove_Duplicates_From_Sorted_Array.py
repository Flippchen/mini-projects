class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        if not nums:
            return 0

        # Initialize the unique index pointer.
        unique_index = 0

        # Iterate through the array.
        for i in range(1, len(nums)):
            # If the current element is different from the last unique element,
            # increment the unique index and update the value at that index.
            if nums[i] != nums[unique_index]:
                unique_index += 1
                nums[unique_index] = nums[i]

        # The length of the unique elements is the unique index + 1.
        return unique_index + 1