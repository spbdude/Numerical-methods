N = 15  #номер в списке группы
eps = 10**(-6) #погрешность
def f(x, y, z): return 2*x**2 + (3+0.1*N)*y**2 + (4+0.1*N)*z**2 + x*y - y*z + x*z +x - 2*y + 3*z + N #целевая функция


import numpy as np



class Array:  #создаем класс матриц

    def __init__(self, a): #конструктор
        self.array = np.array(a, dtype='float')
        self.shape = self.shape_f()

    def shape_f(self):  #Размер матрицы
        return self.array.shape

    def T(self):  #Транспонирование
        res = np.zeros((self.shape[1], self.shape[0]))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res[j][i] = self.array[i][j]
        return Array(res)

    def plus(self, other):  #Сложение
        res = np.array(self.array)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res[i][j] += other.array[i][j]
        return Array(res)

    def pointwise_multiply(self, other):
        res = np.array(self.array)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res[i][j] *= other.array[i][j]
        return Array(res)


    def matrix_multiply(self, other): #матричное умножение
        res = np.zeros((self.shape[0], other.shape[1]))
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                for k in range(len(self.array[i])):
                    res[i][j] += self.array[i][k] * other.array[k][j]
        return Array(res)



    # Operators
    def __add__(self, other):
        return self.plus(other)

    def __sub__(self, other):
        return self.plus(Array(-(other.array)))
    def __mul__(self, other):
        return self.pointwise_multiply(other)
    def __matmul__(self, other):
        return self.matrix_multiply(other)

    def __getitem__(self, ij):
        (i, j) = ij
        return self.array[i][j]

def v_norm(v):
    return np.sqrt((v.T() @ v).array[0][0])

A = Array([[4,1,1], [1,2*(3+0.1*N),-1], [1,-1,2*(4+0.1*N)]]) #Создаем матрицу А
b = Array([[1, -2, 3]]).T() #Матрицу b
x_n = Array([[1, 1, 1]]).T() #Начальное значение

def diag_preob(B): #нахождение дельты
    delta = 0.0;
    sum = 0.0;
    m,n = np.shape(A)
    for i in range(m):
        for j in range(n):
            #print(A.array[i][j])
            if i != j:
                sum += abs(A.array[i][j])
        if i == 0:
            delta = abs(A.array[i][i]) - sum
        elif (abs(A.array[i][i]) - sum < delta):
            delta = abs(A.array[i][i]) - sum
        sum = 0
    return delta


def MNGS(A, b, x_n):
    n=0
    delta = diag_preob(A)
    while ( n == 0) or v_norm(A@x_n + b)/delta >= eps:
        x_o = x_n
        q = A @ x_o + b #Для квадратичной функции вычисляется по этой формуле
        Aq = A @ q
        mu = - sum((q * q).array) / sum((q * Aq).array)
        x_n = x_o + Array([mu, mu, mu]) * q #считаем новое значние
        n+=1
    print("Количество итераций:", n)
    return x_n #Возвращаем минимум

def MNPS(A, b, x_n):
    n=0
    while True:
        for i in range(3): #R(n) - n = 3
            x_o = x_n
            e = Array([[0, 0, 0]]).T() #орт пространства
            e.array[i] = [1]
            mu = - sum((e * (A @ x_o + b)).array) / sum((e * (A @ e)).array)
            x_n = x_o + Array([mu,mu,mu]) * e #ищем новое значение
            n+=1
            check = v_norm(x_n-x_o) < eps #проверка на погрешность
            if check:
                print("Количество итераций:", n)
                return x_n #Возвращаем минимум

print("MNGS")
x = MNGS(A, b, x_n)
print("Значение x:", x.array)
print("f(x) = ", f(*x.array)[0])
x_true = Array([[-46/185, 83/370, -17/74]]).T()
print('Абсолютная погрешность:', f(*x_true.array)[0] - f(*x.array)[0])

print("MNPS")
x = MNPS(A, b, x_n)
print("Значение x:", x.array)
print("f(x) = ", f(*x.array)[0])
x_true = Array([[-46/185, 83/370, -17/74]]).T()
print('Абсолютная погрешность:', f(*x_true.array)[0] - f(*x.array)[0])
