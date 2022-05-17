import numpy as np
np.set_printoptions(precision=1)

def format_number(number) -> str:
    if int(number) == number:
        return str(int(number))
    return str(number)

def print_matrix(a) -> str:
    result = "\\begin{bmatrix}\n"
    for row in a:
        for elem in row:
            result += format_number(elem) + " & "
        result = result[:-2] + "\\\\\n"
    print(result[:-3] + "\n\\end{bmatrix}")

a = np.array([[1,2,6,1],[0,-2,-1,0],[-1,3,3,6],[4,-2,1,1]], dtype=np.float64)
n = a.shape[0]
print_matrix(a)
counter = 1
for i in range(n):
    if abs(a[i][i]) <= 10**(-6):
        # Try to swap the rows
        ok = False
        for j in range(i+1, n):
            if abs(a[j][i]) >= 10**(-6):
                ok = True
                a[[i, j]] = a[[j, i]]
                print("\sim")
                print_matrix(a)
                counter += 1
                if counter % 4 == 0:
                    print("\\\\")
                break
        if not ok:
            raise ZeroDivisionError()
    for j in range(i+1, n):
        ratio = a[j][i]/a[i][i]
        a[j] -= ratio * a[i]
        print("\sim")
        print_matrix(a)
        counter += 1
        if counter % 4 == 0:
            print("\\\\")
# Inverse to make the top zero
for i in range(n-1, -1, -1):
    for j in range(i-1, -1, -1):
        ratio = a[j][i]/a[i][i]
        a[j] -= ratio * a[i]
        print("\sim")
        print_matrix(a)
        counter += 1
        if counter % 4 == 0:
            print("\\\\")
print("\sim")
print_matrix(np.identity(n))