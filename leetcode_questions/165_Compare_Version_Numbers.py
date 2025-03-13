class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        l1=list(map(int,version1.split('.')))
        l2=list(map(int,version2.split('.')))
        if l1==l2:
            return(0)
        
        a=len(l1)
        b=len(l2)
        
        if a>b:
            for i in range(a-b):
                l2.append("0")
        
        else:
            for i in range(b-a):
                l1.append("0")
            
        for i in range(len(l1)):
            if int(l1[i])>int(l2[i]):
                return(1)
            
            elif int(l1[i])<int(l2[i]):
                return(-1)
            
            else:
                pass
        
        return(0)

# Modified on 2024-09-01 14:19:56.218060

# Modified on 2024-11-13 18:05:27.248385

# Modified on 2024-12-20 17:19:47.712260

# Modified on 2025-02-20 23:27:01.032345

# Modified on 2025-03-23 16:38:14.029219
