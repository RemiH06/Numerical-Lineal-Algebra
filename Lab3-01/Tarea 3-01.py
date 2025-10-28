# Importa las funciones desde lab9.py
from lab9 import biseccion, punto_fijo, newton
import math
import numpy as np

# ==============================
# 1) BISECCIÓN para sin(x) = 6x + 5
# ==============================
def demo_biseccion_sin_eq():
    """
    Encuentra una raíz de sin(x) = 6x + 5 usando Bisección con precisión hasta 6 decimales.
    Ecuación: f(x) = sin(x) - 6x - 5
    """
    f = lambda x: math.sin(x) - 6 * x - 5
    a, b = -1.0, 0.0  # Intervalo inicial
    raiz, k = biseccion(f, a, b, tolerancia=1e-6)  # Llamada a la función de bisección con tolerancia de 6 decimales
    print(f"\n≈ Raíz (6 dec): {raiz:.6f}  | Iteraciones: {k}")

# ============================
# 2) VERIFICACIÓN de convergencia para g(x) = (2x - 1)^(1/3)
# ============================
# En la llamada a la función 'punto_fijo' debes pasar solo el valor inicial (x0)
def demo_punto_fijo_teorema():
    """
    Verifica si el método de punto fijo aplicado sobre la función g(x) = (2x-1)^(1/3)
    converge a la raíz p=1. Muestra el proceso de verificación.
    """
    g = lambda x: (2 * x - 1) ** (1/3)
    gp = lambda x: (2 / 3) * (2 * x - 1) ** (-2 / 3)
    x0 = 0.8  # Valor inicial para el método de punto fijo

    raiz, k = punto_fijo(g, x0, tolerancia=1e-6, max_iter=1000)
    print(f"\nRaíz encontrada: {raiz:.6f} | Iteraciones: {k}")

# ================================
# 3) NEWTON para sin(x) = 6x + 5
# ================================
def demo_newton_sin_eq():
    """
    Encuentra una raíz de sin(x) = 6x + 5 usando el método de Newton con precisión hasta 6 decimales.
    Ecuación: f(x) = sin(x) - 6x - 5,  f'(x) = cos(x) - 6
    """
    f = lambda x: math.sin(x) - 6 * x - 5
    df = lambda x: math.cos(x) - 6
    x0 = -1.0  # Aproximación inicial
    raiz, k = newton(f, df, x0, tolerancia=1e-6)  # Llamada a la función de Newton con tolerancia de 6 decimales
    print(f"\n≈ Raíz (6 dec): {raiz:.6f}  | Iteraciones: {k}")

print("=" * 100)
print("1) BISECCIÓN: Resolver sin(x) = 6x + 5 (precisión 6 decimales)")
print("=" * 100)
demo_biseccion_sin_eq()

print("\n" + "=" * 100)
print("2) PUNTO FIJO: Verificar convergencia de g(x) = (2x-1)^(1/3) hacia p=1")
print("=" * 100)
demo_punto_fijo_teorema()

print("\n" + "=" * 100)
print("3) NEWTON: Resolver sin(x) = 6x + 5 (precisión 6 decimales)")
print("=" * 100)
demo_newton_sin_eq()