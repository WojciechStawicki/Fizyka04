import numpy as np


def fit_least_squares(T: list, n: list) -> tuple:
    """
    Dopasowuje równanie n = n0 + aT + bT^2
    T: Lista wartości temperatury
    n: Lista wartości współczynnika
    """
    N = len(T)
    sum_T = sum(T)
    sum_T2 = sum(t ** 2 for t in T)
    sum_T3 = sum(t ** 3 for t in T)
    sum_T4 = sum(t ** 4 for t in T)
    sum_n = sum(n)
    sum_nT = sum(ni * ti for ni, ti in zip(n, T))
    sum_nT2 = sum(ni * ti ** 2 for ni, ti in zip(n, T))


    A = [[N, sum_T, sum_T2],
         [sum_T, sum_T2, sum_T3],
         [sum_T2, sum_T3, sum_T4]]
    B = [sum_n, sum_nT, sum_nT2]

    # Rozwiązanie układu równań metodą eliminacji Gaussa
    coefficients = np.linalg.solve(A, B)

    n0, a, b = coefficients
    return n0, a, b


def calculate_uncertainties(T: list, n: list, n0: float, a: float, b: float) -> tuple:
    N = len(T)
    sum_T = sum(T)
    sum_T2 = sum(t ** 2 for t in T)
    mean_T = sum_T / N
    mean_T2 = sum_T2 / N

    residuals = [ni - (n0 + a * ti + b * ti ** 2) for ti, ni in zip(T, n)]
    variance = sum(r ** 2 for r in residuals) / (N - 3)

    delta = N * sum_T2 - sum_T ** 2
    sigma_n0 = np.sqrt((sum_T2 * variance) / delta)
    sigma_a = np.sqrt((N * variance) / delta)
    sigma_b = np.sqrt((variance) / delta)

    return sigma_n0, sigma_a, sigma_b


# Przykładowe użycie
if __name__ == "__main__":
    # Wklej dane tutaj
    T = [25.0,  29.6,  34.1,  38.9,  42.8,  47.7,  52.1,  58.9,  62.5]  # Przykładowe wartości T
    n = [1.639, 1.636, 1.634, 1.633, 1.630, 1.628, 1.626, 1.623, 1.621]  # Przykładowe wartości n

    if len(T) != len(n):
        print("Błąd: listy T i n muszą mieć taką samą długość.")
    else:
        n0, a, b = fit_least_squares(T, n)
        sigma_n0, sigma_a, sigma_b = calculate_uncertainties(T, n, n0, a, b)

        print(
            f"Obliczone współczynniki: n0 = {n0:.15f} ± {sigma_n0:.15f}, a = {a:.15f} ± {sigma_a:.15f}, b = {b:.15f} ± {sigma_b:.15f}")
