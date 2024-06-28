import numpy as np

def SimplexOneStep(A,b,c,B,N):
    matriz_B = A[:, B]
    B_inv = np.linalg.inv(matriz_B)
    x_b = np.dot(B_inv, b)
    c_B = [c[indice] for indice in B]
    c_N = [c[indice] for indice in N]
    lamda =  np.dot(B_inv.T, c_B)
    matriz_N = A[:, N]
    s_n = c_N - np.dot(matriz_N.T, lamda)


    if np.all(s_n >= 0):
        print("Punto óptimo encontrado")
        return x_b,B, N

    q = N[0]
    A_q = A[:, q].T
    d = np.dot(B_inv, A_q)

    if np.all(d <= 0):
        print("El problema no esta acotado")
        return

    residuo_min = 10000
    p = -1
    for i in range(len(d)):
        if d[i] > 0:
            cociente_q = x_b[i] / d[i]
            if cociente_q < residuo_min:
                residuo_min = cociente_q
                p = B[i]
        
    for i in range(len(B)):
        if B[i] == p:
            B[i] = q
    N = list(set(range(A.shape[1])) - set(B))

    return x_b,B, N


B = np.array([2, 3])
N = np.array([0, 1])
A = np.array([[1, 1, 1, 0],[2, 0.5, 0, 1]])
b = np.array([5,8])
c = np.array([-4,-2,0,0])

x1, B1, N1 = SimplexOneStep(A,b,c,B,N)
print(x1, B1, N1)

x2, B2, N2 = SimplexOneStep(A,b,c,B1,N1)
print(x2, B2, N2)

x2, B2, N2 = SimplexOneStep(A,b,c,B2,N2)
print(x2, B2, N2)
