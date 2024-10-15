class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []
        for t in tokens:
            try:
                temp = int(t)
                stack.append(temp)
            except:
                b = stack.pop()
                a = stack.pop()
                if t == "+":
                    a += b
                elif t == "-":
                    a -= b
                elif t == "*":
                    a *= b
                else:
                    a = int(a * 1.0 / b)
                stack.append(a)
        return stack[-1]
# Modified on 2024-09-01 14:19:56.011622

# Modified on 2024-09-15 22:35:08.640982

# Modified on 2024-09-15 22:40:27.445943

# Modified on 2024-10-06 12:22:15.365558

# Modified on 2024-10-22 18:40:55.562420
