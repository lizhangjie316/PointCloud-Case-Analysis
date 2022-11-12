#
# def getValue(rowIndex,columnIndex) -> list[list[int]]:
#
#     ret = list()
#     for i in range(rowIndex):
#         row = list()
#         for j in range(0, i + 1):
#             if j == 0 or j == i:
#                 row.append(1)
#             else:
#                 row.append(ret[i - 1][j] + ret[i - 1][j - 1])
#
#     ret.append(row)
#     return ret[-1][columnIndex - 1]
#
#
# if __name__ == "__main__":
#     print(getValue(3,1))