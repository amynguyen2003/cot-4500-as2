import numpy as np
from scipy.linalg import solve

np.set_printoptions(precision = 7, suppress = True, linewidth = 100)

# 1)

def nevilles_method(x_pts, y_pts, approx):
    size = len(x_pts)
    matrix = np.zeros((size, size))

    num_pts = len(x_pts)

    for index, row in enumerate(matrix):
        row[0] = y_pts[index]
    
    for i in range(1, num_pts):
        for j in range(1, i + 1):
            first_multiplication = (approx - x_pts[i - j]) * matrix[i][j - 1]
            second_multiplication = (approx - x_pts[i]) * matrix[i - 1][j - 1]

            denom = x_pts[i] - x_pts[i - j]

            matrix[i][j] = (first_multiplication - second_multiplication) / denom
    
 
    print(matrix[num_pts - 1][num_pts - 1])
    print("\n")

x_pts = [3.6, 3.8, 3.9]
y_pts = [1.675, 1.436, 1.318]
approx = 3.7 
nevilles_method(x_pts, y_pts, approx)


# 2) and 3)

def newt_forward():
    x1 = 7.2; x2 = 7.4; x3 = 7.5; x4 = 7.6
    y1 = 23.5492; y2 = 25.3913; y3 = 26.8224; y4 = 27.4589

    y1_prime = (y2 - y1) / (x2 - x1)
    y2_prime = (y3 - y2) / (x3 - x2)
    y3_prime = (y4 - y3) / (x4 - x3)

    y1_double_prime = (y2_prime - y1_prime) / (x3 - x1)
    y2_double_prime = (y3_prime - y2_prime) / (x4 - x2)
    y3_double_prime = (y2_double_prime - y1_double_prime) / (x4 - x1)

    # 2)
    print([y1_prime, y1_double_prime, y3_double_prime])
    print("\n")
    
    # 3)
    f_x_approx = y1 + y1_prime * (7.3 - x1) + y1_double_prime * (7.3 - x2) * (7.3 - x1)\
          + y3_double_prime * (7.3 - x3) * (7.3 - x2) * (7.3 - x1)
    print(f_x_approx)
    print("\n")

newt_forward()


# 4)

def apply_div_diff(matrix: np.array):
    for i in range(2, len(matrix)):
        for j in range(2, i + 2):

            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            numer = matrix[i][j - 1] - matrix[i - 1][j - 1]
            
            denom = matrix[i][0] - matrix[i - j + 1][0]

            matrix[i][j] = numer / denom
    return matrix

def hermite_interpolation():
    a1 = a2 = 3.6; a3 = a4 = 3.8; a5 = a6 = 3.9

    b1 = b2 = 1.675; b3 = b4 = 1.436; b5 = b6 = 1.318

    c1 = 0
    c2 = -1.195
    c3 = (b3 - b2) / (a3 - a1)
    c4 = -1.188
    c5 = (b5 - b4) / (a5 - a4)
    c6 = -1.182

    d1 = d2 = 0
    d3 = (c3 - c2) / (a3 - a1)
    d4 = (c4 - c3) / (a4 - a1)
    d5 = (c5 - c4) / (a5 - a3)
    d6 = (c6 - c5) / (a6 - a4)

    e1 = e2 = e3 = 0
    e4 = (d4 - d3) / (a4 - a1)
    e5 = (d5 - d4) / (a5 - a1)
    e6 = (d6 - d5) / (a6 - a3)

    f1 = f2 = f3 = f4 = 0
    f5 = (e5 - e4) / (a5 - a1)
    f6 = (e6 - e5) / (a6 - a1)

    print( np.matrix([[a1, b1, c1, d1, e1, f1], [a2, b2, c2, d2, e2, f2], [a3, b3, c3, d3, e3, f3], \
                   [a4, b4, c4, d4, e4, f4], [a5, b5, c5, d5, e5, f5], [a6, b6, c6, d6, e6, f6]]))
    print("\n")
    
hermite_interpolation()

# 5)   

def cubic_spline_interpolation(x, y):
    size = len(x)
    matrix: np.array = np.zeros((size, size))
    matrix[0][0] = 1
    matrix[1][0] = x[1] - x[0]
    matrix[1][1] = 2 * ((x[1] - x[0]) + (x[2] - x[1]))
    matrix[1][2] = x[2] - x[1]
    matrix[2][1] = x[2] - x[1]
    matrix[2][2] = 2 * ((x[3] - x[2]) + (x[2] - x[1]))
    matrix[2][3] = x[3] - x[2]
    matrix[3][3] = 1
    print(matrix, "\n")

    b0 = b3 = 0
    b1 = ((3 / (x[2] - x[1])) * (y[2] - y[1])) - ((3 / (x[1] - x[0])) * (y[1] - y[0]))
    b2 = ((3 / (x[3] - x[2])) * (y[3] - y[2])) - ((3 / (x[2] - x[1])) * (y[2] - y[1]))
    vector_b = np.array([b0, b1, b2, b3])
    print(vector_b)
    print("\n")

    f = [[matrix]]
    g = [[b0], [b1], [b2], [b3]]

    vector_x = solve(f, g)

    print(vector_x.T[0])
    print("\n")

x = [2, 5, 8, 10]
f_x = [3, 5, 7, 9]
cubic_spline_interpolation(x, f_x)