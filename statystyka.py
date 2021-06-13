import statistics

import numpy as np

# print("Zadanie 7")
A = np.array([2, 5, 7, 1.4, 25, 1.75, 8])
# print("Średnia:", np.mean(A))
# print("Mediana:", np.median(A))
# print()

# print("Zadanie 8")
B = (A - np.mean(A)) / np.std(A)
# print("Znormalizowana tablica wartości:", B)
# print()

print("Zadanie 9")


def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


def distance2(v1, v2):
    return np.linalg.norm(v1 - v2)


A = np.array([2, 5, 7, 1.4, 25, 1.75, 8])
B = np.array([1, 3, 6, 9.25, 17, 3, 12])
print(A)
print(B)
print(distance(A, B))
print(distance2(A, B))
print()
