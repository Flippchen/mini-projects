class Solution:
    def defangIPaddr(self, address: str) -> str:
        # replace
        return address.replace('.', '[.]')
    # def defangIPaddr(self, address: str) -> str:
    #     # split and join
    #     return '[.]'.join(address.split('.'))
    # def defangIPaddr(self, address: str) -> str:
    #     # replace
    #     return re.sub('\.', '[.]', address)
    # def defangIPaddr(self, address: str) -> str:
    #     return ''.join('[.]' if c == '.' else c for c in address)

# Modified on 2024-09-01 14:19:56.368256

# Modified on 2024-11-13 18:05:27.405675

# Modified on 2024-12-20 17:19:47.917624

# Modified on 2025-02-20 23:27:01.217003
