class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        # Mapping of digits to letters.
        digit_to_char = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }

        def backtrack(index, path):
            # If the path length is equal to the length of digits, we have found a combination.
            if len(path) == len(digits):
                combinations.append("".join(path))
                return

            # Get the letters that the current digit can represent.
            possible_letters = digit_to_char[digits[index]]
            for letter in possible_letters:
                # Append the letter to the current path and move to the next digit.
                path.append(letter)
                backtrack(index + 1, path)
                path.pop()  # Backtrack

        combinations = []
        backtrack(0, [])
        return combinations
