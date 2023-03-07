import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

a, b = -0.9, 5

def f(x):
    return x * np.log(x+1)

def Newton_coefficient(x, y):
    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])
    return a

def Newton(x, xs, fs):
    a = Newton_coefficient(xs, fs)
    n = len(xs) - 1
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - xs[n - k])*p
    return p

def Lagrange(x, xs, fs):
    res = 0
    n = len(xs)
    i,j = 0,0
    num, den = 1, 1
    for j in range(n):
        for i in range(n):
            if (xs[j] != xs[i]):
                num *= x - xs[i]
            if (xs[j] != xs[i]):
                den *= xs[j] - xs[i]
        res += (num / den) * f(xs[j])
        num, den = 1, 1
    return res

def optimal_points(a, b, n):
    x_opt = []
    f_opt = []
    for i in range(1, n + 1):
        x_opt.append(0.5 * ((b-a)*np.cos(((2*i+1)*np.pi)/(2*n+2)) + (b+a)))
    f_opt = list(f(np.array(x_opt)))
    x_opt = list(x_opt)
    return x_opt , f_opt


#задаем значения
x_true = np.linspace(a,b, 1000)
f_true = f(np.array(x_true))
Gr_nodes = [5, 10, 50]

#лагранж равноотстоящий
p_Lag = []
p_Lag10 = []
p_Lag50 = []
x_true = np.linspace(a,b, 1000)
for i in range(1000):
    x_sm = np.linspace(a, b, Gr_nodes[0])
    f_sm = f(np.array(x_sm))
    p_Lag.append(Lagrange(x_true[i], x_sm, f_sm))
for i in range(1000):
    x_sm10 = np.linspace(a, b, Gr_nodes[1])
    f_sm = f(np.array(x_sm10))
    p_Lag10.append(Lagrange(x_true[i], x_sm10, f_sm))
for i in range(1000):
    x_sm50 = np.linspace(a, b, Gr_nodes[2])
    f_sm = f(np.array(x_sm50))
    p_Lag50.append(Lagrange(x_true[i], x_sm50, f_sm))




title = "Lagrange red = 5, blue = 10, yellow = 50, green = true"
plt.plot(x_true, p_Lag, color='red')
plt.plot(x_true, p_Lag10, color='blue')
plt.plot(x_true, p_Lag50, color='yellow')
plt.plot(x_true, f_true, color='green')
plt.title(title)
plt.grid()
plt.show()



#Лагранж оптимальный
p_Lagopt = []
p_Lagopt10 = []
p_Lagopt50 = []
for i in range(1000):
    x_opt, f_opt = optimal_points(a, b, Gr_nodes[0])
    p_Lagopt.append(Lagrange(x_true[i], x_opt, f_opt))
for i in range(1000):
    x_opt10, f_opt10 = optimal_points(a, b, Gr_nodes[1])
    p_Lagopt10.append(Lagrange(x_true[i], x_opt10, f_opt10))
for i in range(1000):
    x_opt50, f_opt50 = optimal_points(a, b, Gr_nodes[2])
    p_Lagopt50.append(Lagrange(x_true[i], x_opt50, f_opt50))


title = "LagrangeOpt red = 5, blue = 10, yellow = 50, green = true"
plt.plot(x_true, p_Lagopt, color='red')
plt.plot(x_true, p_Lagopt10, color='blue')
plt.plot(x_true, p_Lagopt50, color='yellow')
plt.plot(x_true, f_true, color='green')
plt.title(title)
plt.grid()
plt.show()

#Ньютон равноотстоящий
p_New = []
p_New10 = []
p_New50 = []
for i in range(1000):
    x_sm = np.linspace(a, b, Gr_nodes[0])
    f_sm = f(np.array(x_sm))
    p_New.append(Newton(x_true[i], x_sm, f_sm))
for i in range(1000):
    x_sm10 = np.linspace(a, b, Gr_nodes[1])
    f_sm = f(np.array(x_sm10))
    p_New10.append(Newton(x_true[i], x_sm10, f_sm))
for i in range(1000):
    x_sm50 = np.linspace(a, b, Gr_nodes[2])
    f_sm = f(np.array(x_sm50))
    p_New50.append(Newton(x_true[i], x_sm50, f_sm))



title = "Newton red = 5, blue = 10, yellow = 50, green = true"
plt.plot(x_true, p_New, color='red')
plt.plot(x_true, p_New10, color='blue')
plt.plot(x_true, p_New50, color='yellow')
plt.plot(x_true, f_true, color='green')
plt.title(title)
plt.grid()
plt.show()

#Ньютон оптимальный
p_Newopt = []
p_Newopt10 = []
p_Newopt50 = []
for i in range(1000):
    x_opt, f_opt = optimal_points(a, b, Gr_nodes[0])
    p_Newopt.append(Newton(x_true[i], x_opt, f_opt))
for i in range(1000):
    x_opt10, f_opt10 = optimal_points(a, b, Gr_nodes[1])
    p_Newopt10.append(Newton(x_true[i], x_opt10, f_opt10))
for i in range(1000):
    x_opt50, f_opt50 = optimal_points(a, b, Gr_nodes[2])
    p_Newopt50.append(Newton(x_true[i], x_opt50, f_opt50))


title = "NewtonOpt red = 5, blue = 10, yellow = 50, green = true"
plt.plot(x_true, p_Newopt, color='red')
plt.plot(x_true, p_Newopt10, color='blue')
plt.plot(x_true, p_Newopt50, color='yellow')
plt.plot(x_true, f_true, color='green')
plt.title(title)
plt.grid()
plt.show()

#график распределения абс погрешности
p_NewoptPOG = []
x_opt, f_opt = optimal_points(a, b, 20)
for i in range(1000):
    p_NewoptPOG.append(abs(Newton(x_true[i], x_opt, f_opt) - f(x_true[i])))
plt.plot(x_true, p_NewoptPOG , color='blue')
plt.grid()
plt.show()

myTable1 = PrettyTable(["Количество узлов(n)", "Максимальное отклонение(RLn)", "Максимальное отклонение(RLopt)"])
m = 1000
max = 0
maxopt = 0
x_check = np.linspace(a , b , m)
f_check = f(np.array(x_check))
n = 100
h = 2


for h in range(2,n+1, 5):
    max = 0
    maxopt = 0
    x_smCH = np.linspace(a, b, h)
    f_smCH = f(np.array(x_smCH))
    x_optCH, f_optCH = optimal_points(a, b, h)
    for i in range(m):
        k = Lagrange(x_check[i], x_smCH, f_smCH)
        l = abs((f_check[i] - k))
        if l > max:
            max = l
    for i in range(m):
        o = Lagrange(x_check[i], x_optCH, f_optCH)
        j = abs((f_check[i] - o))
        if j > maxopt:
            maxopt = j
    myTable1.add_row([h, max, maxopt])
print(myTable1)
print('m = ' + str(m))

myTable2 = PrettyTable(["Количество узлов(n)", "Максимальное отклонение(RNew)", "Максимальное отклонение(RNewopt)"])
m = 1000
max = 0
maxopt = 0
x_check = np.linspace(a , b , m)
f_check = f(np.array(x_check))
n = 100



for h in range(2, n+1,5):
    max = 0
    maxopt = 0
    x_smCH = np.linspace(a, b, h)
    f_smCH = f(np.array(x_smCH))
    x_optCH, f_optCH = optimal_points(a, b, h)
    for i in range(m):
        k = Newton(x_check[i], x_smCH, f_smCH)
        l = abs((f_check[i] - k))
        if l > max:
            max = l
    for i in range(m):
        o = Newton(x_check[i], x_optCH, f_optCH)
        j = abs((f_check[i] - o))
        if j > maxopt:
            maxopt = j
    myTable2.add_row([h, max, maxopt])
print(myTable2)
print('m = ' + str(m))
