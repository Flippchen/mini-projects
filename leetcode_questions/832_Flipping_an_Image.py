class Solution(object):
    def flipAndInvertImage(self, A):
        for row in A:
            for i in xrange((len(row) + 1) / 2):
                """
                In Python, the shortcut row[~i] = row[-i-1] = row[len(row) - 1 - i]
                helps us find the i-th value of the row, counting from the right.
                """
                row[i], row[~i] = row[~i] ^ 1, row[i] ^ 1
        return A
        # return [[1 ^ i for i in row[::-1]] for row in A]

# Modified on 2024-09-01 14:19:56.146238

# Modified on 2024-09-15 22:35:08.778300

# Modified on 2024-10-22 18:40:55.688867

# Modified on 2024-11-13 18:05:27.174434
