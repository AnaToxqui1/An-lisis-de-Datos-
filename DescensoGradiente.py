import numpy as np

#A continuación se muestran las cuatro funciones con las que se probará el algoritmo y sus gradientes 

def HimmekblauFunction(x):
# Definida para 2 dimensiones
	f = (x[0]**2 + x[1] -11)**2 + (x[1]**2 + x[0] -7)**2 
	df_dx0 = 4*x[0] * (x[0]**2 + x[1] - 11) + 2 * (x[1]**2 + x[0] - 7)
	df_dx1 = 2*(x[0]**2 + x[1] - 11) + 4 * x[1] * (x[1]**2 + x[0] - 7)
	gradiente = np.array([df_dx0, df_dx1])
	return f, gradiente

def McCormickFunction(x):
# Definida para 2 dimensiones
	f = np.sin(x[0] + x[1]) + (x[0] - x[1])**2 -(1.5*x[0]) + (2.5*x[1]) +1
	df_dx0 = np.cos(x[0] + x[1]) + 2 * (x[0] - x[1]) - 1.5 
	df_dx1 = np.cos(x[0] + x[1]) - 2 * (x[0] - x[1]) + 2.5
	gradiente = np.array([df_dx0, df_dx1])
	return f, gradiente

def BootFunction(x):
# Definida para 2 dimensiones
	f = (x[0]+2*x[1]-7)**2 +(2*x[0]+x[1]-5)**2
	df_dx0 = 10*x[0]+ 8*x[1]-34
	df_dx1 = 8*x[0] + 10*x[1] -38
	gradiente = np.array([df_dx0, df_dx1])
	return f, gradiente

def SphereFunction(x):
# Definida para n dimensiones
	n = len(x)
	f = 0
	for i in range(n):
	  f += x[i]**2
	gradiente = 2 * x
	return f, gradiente 

# Algoritmo del descenso del gradiente IMPLEMENTADO POR MI 
def DesGradiente(x0, alpha0, n, funcion, c, p, epsilon):
	x = x0
	for k in range(n):
		fk, gradientefk = funcion(x)
		pk = -(gradientefk/np.linalg.norm(gradientefk)**2)
		alpha = alpha0
		while funcion(x + (alpha*pk))[0] > (fk + (c*alpha* np.dot(gradientefk,pk))):
			alpha = p*alpha
		alpha0 = alpha
		x = x + (alpha0*pk)
		if np.linalg.norm(gradientefk)**2 < epsilon:
			break
	return x, funcion(x)[0]

#Descenso del gradiente implementado en numpy
def PythonDesGrad(x0, tasa_aprendizaje, n, funcion):
	for i in range(n):
		gradiente = funcion(x0)[1]
		x0 = x0 - tasa_aprendizaje * gradiente
		f = funcion(x0)[0]
	return x0, f

# Punto inicial y parámetros
x0= np.array([20, -20, 20, -20, 20])
alpha0= 100
n= 10
c = 0.1
p = 0.3
epsilon = 1e-6
tasa_aprendizaje = 0.009

# Aplicar el algoritmo de descenso del gradiente

vector, valor  = DesGradiente(x0, alpha0, 10, SphereFunction, c, p, epsilon)
print("Resultado del descenso del gradiente 10:")
print(vector, valor, "\n")
vector, valor  = DesGradiente(x0, alpha0, 100, SphereFunction, c, p, epsilon)
print("Resultado del descenso del gradiente 100:")
print(vector, valor, "\n")
vector, valor  = DesGradiente(x0, alpha0, 500, SphereFunction, c, p, epsilon)
print("Resultado del descenso del gradiente 500:")
print(vector, valor, "\n")
vector, valor  = DesGradiente(x0, alpha0, 1000, SphereFunction, c, p, epsilon)
print("Resultado del descenso del gradiente 1000:")
print(vector, valor, "\n","\n","\n")
vector1, valor1  = PythonDesGrad(x0, tasa_aprendizaje, 10, SphereFunction)
print("Resultado del descenso del gradiente con numpy: 10")
print(vector1, valor1)
vector1, valor1  = PythonDesGrad(x0, tasa_aprendizaje, 100, SphereFunction)
print("Resultado del descenso del gradiente con numpy: 100")
print(vector1, valor1)
vector1, valor1  = PythonDesGrad(x0, tasa_aprendizaje, 500, SphereFunction)
print("Resultado del descenso del gradiente con numpy: 500")
print(vector1, valor1)
vector1, valor1  = PythonDesGrad(x0, tasa_aprendizaje, 1000, SphereFunction)
print("Resultado del descenso del gradiente con numpy: 1000")
print(vector1, valor1)
