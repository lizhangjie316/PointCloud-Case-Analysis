


def mincost(s1, s2, C, D):
    m, n = len(s1), len(s2)
    if not s1:
        return n*C
    if not s2:
        return m*D

    dp = [[0] * (n+1) for i in range(m+1)]
    # print(dp)
    for i in range(1,n + 1):
        dp[0][i] = C * i

    for i in range(1,m+1):
        for j in range(i,n+1):
            if i==j and s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            elif i==j and s1[i-1] != s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + D
            else:  # j>i的
                dp[i][j] = dp[i][j-1] + C
    print(dp)
    return dp[-1][-1]


s1 = 'baaba'
s2 = 'bbbbbbb'
# s1 = 'aba'
# s2 = 'bbb'
C, D = 20, 30
print(mincost(s1, s2, C, D))
