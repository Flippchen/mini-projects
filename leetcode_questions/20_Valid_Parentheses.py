class Solution:
    def isValid(self, s: str) -> bool:
        # Dictionary to map opening brackets to their corresponding closing brackets
        bracket_map = {")": "(", "}": "{", "]": "["}
        # Initialize an empty stack
        stack = []

        for char in s:
            # If the current character is a closing bracket
            if char in bracket_map:
                # Pop the top element from the stack if it's not empty, otherwise use a dummy value
                top_element = stack.pop() if stack else '#'

                # Check if the popped element matches the corresponding opening bracket
                if bracket_map[char] != top_element:
                    return False
            else:
                # For an opening bracket, push it onto the stack
                stack.append(char)

        # The string is valid if the stack is empty
        return not stack
