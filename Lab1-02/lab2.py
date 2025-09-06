import numpy as np

A = np.array([
    [ 1.32,  3.25,  2.77],
    [-3.57,  2.12, -3.26],
    [-9.87,  3.33,  2.15]
], dtype=float)

# Inversa
B = np.linalg.inv(A)

# Multiplicación matricial
I = A @ B

print("B = inv(A) =\n", B)
print("\nA @ B =\n", I)   # Identidad

rng = np.random.default_rng(42)

for _ in range(5):
    print(f"Matriz {_+1}")
    M = rng.normal(0, 3, size=(3,3))
    print(M)
    
    while abs(np.linalg.det(M)) < 1e-6:
        M += rng.normal(0, 1e-3, size=(3,3))
    
    W = np.linalg.inv(M)
    print("Inversa: ")
    print(W)
    
    MW = M @ W
    print("M@W: ")
    print(MW.round(6))
