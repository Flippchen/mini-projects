class Solution:
    def findJudge(self, N: int, trust: List[List[int]]) -> int:
        if N==1:
            return 1
        d1={}
        d2={}
        for i, j in trust:
            if j in d1:
                d1[j]+=1
            else:
                d1[j]=1
            if i in d2:
                d2[i]+=1
            else:
                d2[i]=1
        for i,j in d1.items():
            if j==N-1:
                if i not in d2:
                    return i
        return -1
                    

# Modified on 2024-09-01 14:19:56.320286

# Modified on 2024-11-13 18:05:27.351549

# Modified on 2024-12-20 17:19:47.848522

# Modified on 2025-02-20 23:27:01.159077

# Modified on 2025-03-23 16:38:14.151285
