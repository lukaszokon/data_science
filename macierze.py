import numpy as np
import statistics

A = np.array([3, 6, 8, 1])
# print(A)

A1 = np.array([[3],
               [6],
               [8],
               [1]])
# print(A1)

B = np.array([[3, 7, 4], [9, 3, 5], [8, 4, 3]])
# print(B)
# print(B.shape)
# print(B.size)
# print(B.ndim)
# print(B[1:, 1:])

C = np.arange(2, 8)
# print(C)
# print(C.ndim)
C = C.reshape((3, 2))
# print(C)
# print(C[1, 1])
# print(C[1][1])

D = np.linspace(0, 1, 5)
# print(D)

E = np.array([[[2, 1], [7, 4], [2, 5]], [[7, 2], [8, 4], [5, 1]], [[3, 6], [7, 0], [0, 5]]])
# print(E)
# print(E.ndim)

# print("Rozkład normalny:", np.random.normal())
# print("Układ jednolity:", np.random.uniform())
# print("Liczba całkowita:", np.random.randint(22, 65))
# print("Liczba zmiennoprzecinkowa:", np.random.random())

# wzrost_pilkarzy = np.random.uniform(low=1.65, high=1.95, size=5)
# print(wzrost_pilkarzy)
# wysocy = np.where(wzrost_pilkarzy > 1.85)
# print(wzrost_pilkarzy[wysocy])

print("Zadanie 1")
A = np.arange(1, 10)
print(A)
print(A[::-1])
print()

print("Zadanie 2")
A = np.array([1, 23, 4, 31, 1, 1, 4, 23, 4, 1])
print(A)
print(np.unique(A))
print()

# A = np.array([[3, 4, 5], [1, 2, 3], [4, 5, 6]])
# B = np.array([[-2, 5, 1], [7, 0, 2], [-1, 0, 5]])
# print('Macierz A', A)
# print('Macierz B', B)
# C = np.add(A, B)
# print('Macierz A+B', C)
# D = np.subtract(A, B)
# print('Macierz A-B', D)

A = np.array([[1, 5, 0], [2, -3, 1]])
B = np.array([[0, -2], [1, 1], [3, 4]])
# print("Mnożenie", np.dot(A,B))
# print("Macierz A",A)
# print("Transponowana A", A.T)

C = np.array([[1, 4], [2, 5]])
print(C)
print(np.linalg.inv(C))

D = np.array([[3, 4, 5], [1, 2, 3], [4, 5, 6]])
# print(D)
# print(D.diagonal())
# print(D.diagonal().sum())
# print(D.flatten())
# print(np.max(D))
# print(np.min(D))
# print(np.mean(D))
# print(np.var(D))
# print(np.std(D))
# print(np.identity(3))

# print("Zadanie 3")
# A = np.arange(2, 11).reshape((3, 3))
# print(A)
# print()

# print("Zadanie 4")
# A = np.random.randint(10, 31, size=6)
# print(A)
# print()

# A = np.array([[2,1],[5,3]])
# print(np.linalg.inv(A))

# print("Zadanie 5")
# A = np.array([23, 45, 112, 150, 43, 254, 95, 8])
# wieksze_od_sto = np.where(A > 100)
# print(f"Wartości większe od sto w macierzy A: {A[wieksze_od_sto]}")
# print()

# print("Zadanie 6")
# A = np.array([[1, 15, 4, 13], [8, 21, 3, 12], [11, 13, 11, 5], [32, 13, 0, 2]])
# print("Element drugiego wiersza, trzeciej kolumny:", A[1, 2])
# print("Wyznacznik macierzy A:", np.linalg.det(A))
# print("Ślad macierzy A:", A.diagonal().sum())
# print(f"MAX A: {A.max()}, Min A: {A.min()}")
