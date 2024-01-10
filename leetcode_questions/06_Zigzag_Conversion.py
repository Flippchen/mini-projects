class Solution:
    def convert(self, s, numRows):
        # If numRows is 1 or the string is too short, return the original string
        if numRows == 1 or numRows >= len(s):
            return s

        # Initialize an array of strings for each row
        rows = [''] * numRows
        current_row = 0
        going_down = False

        # Iterate through each character in the string
        for char in s:
            # Add the character to the current row
            rows[current_row] += char

            # If we are at the top or bottom row, change direction
            if current_row == 0 or current_row == numRows - 1:
                going_down = not going_down

            # Move to the next row
            current_row += 1 if going_down else -1

        # Combine all rows and return
        return ''.join(rows)


