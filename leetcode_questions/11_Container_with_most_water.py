class Solution:
    def maxArea(self, height: List[int]) -> int:
        max_area = 0
        left = 0
        right = len(height) - 1

        while left < right:
            # Calculate the area with the current left and right pointers
            current_area = min(height[left], height[right]) * (right - left)
            # Update max_area if current_area is larger
            max_area = max(max_area, current_area)

            # Move the pointer from the shorter line inwards
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return max_area
