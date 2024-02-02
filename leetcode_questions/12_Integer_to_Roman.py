class Solution:
    def intToRoman(self, num: int) -> str:
        # List of Roman numerals
        values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

        roman_numeral = ""

        # Iterate through each symbol, subtracting and appending as we go
        for i in range(len(values)):
            while num >= values[i]:
                num -= values[i]
                roman_numeral += symbols[i]

        return roman_numeral
