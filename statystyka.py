import statistics
import scipy.stats as scs
import matplotlib.pyplot as plt

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
# print(A)
# print(B)
# print(distance(A, B))
# print(distance2(A, B))
# print()

print("Rozkład Bernoulliego")
p = 0.5
data = scs.bernoulli.rvs(p, size=1000)
mean, var, skew, kurt = scs.bernoulli.stats(p, moments='mvsk')
print(mean, var, skew, kurt)
print()

print("Rozkład dwumianowy")
n = 10
p = 0.3
k = np.arange(0, 21)
print(scs.binom.pmf(k, n, p))
print()

# ROZKŁAD POISSON
# mu = 2
# x = np.arange(scs.poisson.ppf(0.01, mu), scs.poisson.ppf(0.99, mu))
# rv = scs.poisson(mu)
# fig, ax = plt.subplots(1, 1)
# ax.vlines(x,0,rv.pmf(x), colors='b', linestyles='-', lw=1, label='frozen pmf')
# ax.legend(loc='best', frameon=False)
# plt.show()

# ROZKŁAD GAUSSA

# fig, ax = plt.subplots(1, 1)
# x = np.linspace(scs.norm.ppf(0.01), scs.norm.ppf(0.99), 100)
# ax.plot(x, scs.norm.pdf(x), 'r-', lw=6, alpha=0.3, label='Normalny - teoretyczny')
# rv = scs.norm(loc=0, scale=2)
# ax.plot(x, rv.pdf(x), 'k-', lw=3, label='Normalny - z próby')
# ax.legend(loc='best')
# plt.show()

# Test T-studenta dla 1 średniej
# print("Test T-studenta dla 1 średniej")
# data = np.loadtxt("Wzrost.csv", delimiter=',', skiprows=0, unpack=True)
# print(scs.ttest_1samp(data, 165))

# Test T-studenta dla 2 średnich
# data1 = np.loadtxt("Wzrost.csv", delimiter=',', skiprows=0, unpack=True)
# data2 = np.loadtxt("example.csv", delimiter=',', skiprows=0, unpack=True)
# results = scs.ttest_ind(data1, data2)
# print(results)

# Test normalności wyników badania
data = np.loadtxt("Wzrost.csv", delimiter=',', skiprows=0, unpack=True)
test_result = scs.normaltest(data)
print(test_result)

#Test Kołomogorowa-Smirnowa
normal_data = scs.norm.rvs(size=1000)
ks_result = scs.kstest(normal_data, 'norm')
print(ks_result)

#Test Shapiro-Wilka
shapiro_result = scs.shapiro(normal_data)
print(shapiro_result)
