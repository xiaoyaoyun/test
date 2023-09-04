import numpy as np

def hungarian_algorithm(cost_matrix):
    m, n = cost_matrix.shape
    # 初始化标记数组
    row_covered = np.zeros(m, dtype=bool)
    col_covered = np.zeros(n, dtype=bool)
    starred_zeros = np.zeros((m, n), dtype=bool)
    
    # Step 1: 减去每行的最小值
    for i in range(m):
        min_val = np.min(cost_matrix[i, :])
        cost_matrix[i, :] -= min_val
    
    # Step 2: 减去每列的最小值
    for j in range(n):
        min_val = np.min(cost_matrix[:, j])
        cost_matrix[:, j] -= min_val
    
    # Step 3: 在每行中找到第一个零，标记并尝试覆盖列
    for i in range(m):
        for j in range(n):
            if cost_matrix[i, j] == 0 and not row_covered[i] and not col_covered[j]:
                starred_zeros[i, j] = True
                row_covered[i] = True
                col_covered[j] = True
    
    # Step 4: 尝试覆盖未覆盖的列
    while not all(col_covered):
        # 找到未覆盖的列
        uncovered_col = np.where(~col_covered)[0][0]
        
        # 找到在该列中标记的零
        zero_indices = np.where(starred_zeros[:, uncovered_col])[0]
        
        if len(zero_indices) == 0:
            # 如果没有标记的零，转到Step 6
            return None
        
        # 标记第一个零并取消标记该行的所有其他零
        i = zero_indices[0]
        col_covered[uncovered_col] = True
        row_covered[i] = False
        for j in range(n):
            starred_zeros[i, j] = False
        
        # 取消标记同一列的其他零
        for k in range(m):
            if starred_zeros[k, uncovered_col]:
                row_covered[k] = False
                col_covered[uncovered_col] = False
                for j in range(n):
                    starred_zeros[k, j] = False
                break
    
    # Step 5: 找到最小未覆盖元素的值
    min_uncovered_val = np.inf
    for i in range(m):
        for j in range(n):
            if not row_covered[i] and not col_covered[j]:
                min_uncovered_val = min(min_uncovered_val, cost_matrix[i, j])
    
    # Step 6: 在未覆盖的行和列上添加最小值，并从已覆盖的行和列中减去最小值
    for i in range(m):
        for j in range(n):
            if not row_covered[i] and not col_covered[j]:
                cost_matrix[i, j] -= min_uncovered_val
            elif row_covered[i] and col_covered[j]:
                cost_matrix[i, j] += min_uncovered_val
    
    # 跳回Step 4
    return hungarian_algorithm(cost_matrix)

# 示例输入 - 邻接矩阵
cost_matrix = np.array([[3, 3, 2],
                        [3, 1, 2],
                        [1, 2, 2]])

# 调用匈牙利算法
matching = hungarian_algorithm(cost_matrix)

# 输出最大匹配
for i, j in enumerate(matching):
    if j is not None:
        print(f'Match {i} -> {j}')
