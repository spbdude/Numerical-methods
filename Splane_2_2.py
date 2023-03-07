import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import itertools

a, b = -0.9, 5

def f(x):
    return x * np.log(x+1)

def df(x):
    return np.log(x+1) + x/(x+1)

def optimal_points(a, b, n):
    x_opt = []
    f_opt = []
    for i in range(1, n + 1):
        x_opt.append(0.5 * ((b-a)*np.cos(((2*i+1)*np.pi)/(2*n+2)) + (b+a)))
    x_opt = np.flip(x_opt)
    f_opt = list(f(np.array(x_opt)))
    return x_opt , f_opt

def Splane1_0(x, xs, fs,n):
    k = 2*n - 2
    Arr = [[0, 0],
         [0, 0]]
    Coef_res = np.zeros((k//2, 2))
    for i in range(k//2):
        for k in range(2):
            for j in range(2):
                if j == 0 and k == 0:
                    Arr[k][j] = xs[i]
                elif j == 0 and k == 1:
                    Arr[k][j] = xs[i+1]
                if j == 1:
                    Arr[k][j] = 1
        F = [fs[i], fs[i+1]]
        coef = np.linalg.solve(Arr,F)
        for o in range(2):
            Coef_res[i][o] = coef[o]
    #print(Coef_res)



    for i in range (n-1):
        #print(x)
        #print(xs[i], i)
        if x == xs[i]:
            v = i
        elif x > xs[i] and x < xs[i+1]:
            v = i
        elif x == xs[i+1]:
            v=i+1
        elif x < xs[0]:
            v = 0
        elif x > xs[n-2]:
            v = n-2



    #print(v)
    if x != xs[n-1]:
        res = x * Coef_res[v][0] + Coef_res[v][1]
    else:
        res = x * Coef_res[v - 1][0] + Coef_res[v - 1][1]
    #print(res)
    return(res)

def Splane2_1(x, xs, fs,n):
    Arr = np.zeros((3 * (n-1), 3 * (n-1)))
    F = []

    for i in range(n-1):

        Arr[3 * i + 0][3 * i + 0] = xs[i] ** 2
        Arr[3 * i + 0][3 * i + 1] = xs[i]
        Arr[3 * i + 0][3 * i + 2] = 1
        Arr[3 * i + 1][3 * i + 0] = xs[i+1] ** 2
        Arr[3 * i + 1][3 * i + 1] = xs[i+1]
        Arr[3 * i + 1][3 * i + 2] = 1
        if i + 1 < n-1:
            Arr[3 * i + 2][3 * i + 0] = -2*xs[i+1]
            Arr[3 * i + 2][3 * i + 1] = -1
            Arr[3 * i + 2][3 * i + 2] = 0
            Arr[3 * i + 2][3 * i + 3] = 2*xs[i+1]
            Arr[3 * i + 2][3 * i + 4] = 1
            Arr[3 * i + 2][3 * i + 5] = 0

        F.append(fs[i])
        F.append(fs[i+1])
        F.append(0)

    Arr[-1][-1] = 2*xs[-1]
    F[-1] = df(b)
    #print(Arr)
    #print(F)
    Coef_res = np.linalg.solve(Arr,F).reshape((n-1 , 3))
    #print(Coef_res)

    for i in range (n-1):
        #print(x)
        #print(xs[i], i)
        if x == xs[i]:
            v = i
        elif x > xs[i] and x < xs[i+1]:
            v = i
        elif x == xs[i+1]:
            v=i+1
        elif x < xs[0]:
            v = 0
        elif x > xs[n-2]:
            v = n-2

    if x != xs[n-1]:
        res = pow(x,2) * Coef_res[v][0] + Coef_res[v][1]*x + Coef_res[v][2]
    else:
        res = pow(x,2) * Coef_res[v-1][0] + Coef_res[v-1][1]*x + Coef_res[v-1][2]
    return(res)

def Splane3_2(x, xs, fs,n):
    h = []
    for i in range(n-1):
        h.append(xs[i+1] - xs[i])
    #print(h)
    H = np.zeros((n-2,n-2))
    Y = []
    for i in range(n-3):
        if i != n-3:
            H[i][i] = 2 * (h[i] + h[i + 1])
            H[i + 1][i] = h[i + 1]
            H[i][i + 1] = h[i + 1]
            H[i + 1][i + 1] = 2 * (h[i] + h[i + 1])
    for i in range(1,n-1):
        Y.append(6 * ((fs[i + 1] - fs[i]) / h[i] - (fs[i] - fs[i - 1]) / h[i - 1]))

    #print(H)
    #print(Y)

    y2 = np.linalg.solve(H,Y)
    y1 = [0]
    yn = [0]
    y = list(itertools.chain(y1, y2, yn))
    true_y = []
    for i in range(n - 1):
        true_y.append((fs[i + 1] - fs[i]) / h[i] - y[i + 1] * (h[i] / 6) - y[i] * (h[i] / 3))


    for i in range (n-1):
        #print(x)
        #print(xs[i], i)
        if x == xs[i]:
            v = i
        elif x > xs[i] and x < xs[i+1]:
            v = i
        elif x == xs[i+1]:
            v=i+1
        elif x < xs[0]:
            v = 0
        elif x > xs[n-2]:
            v = n-2

    if x != xs[n-1]:
        res = fs[v] + true_y[v]*(x-xs[v]) + y[v]* (((x-xs[v]) * (x-xs[v]))/2) + (y[v+1] - y[v])* (((x-xs[v]) * (x-xs[v])* (x-xs[v]))/(6*h[v]))
    else:
        res = fs[v-1] + true_y[v-1]*(x-xs[v-1]) + y[v-1]* (((x-xs[v-1]) * (x-xs[v-1]))/2) + (y[v] - y[v-1])* (((x-xs[v-1]) * (x-xs[v-1])* (x-xs[v-1]))/(6*h[v-1]))
    return(res)






n = 20
x_true = np.linspace(a,b, 1000)
f_true = f(np.array(x_true))
x_sp = np.linspace(a,b, n)
f_sp = f(np.array(x_sp))
sp10 = []
for i in range(n):
    sp10.append(Splane1_0(x_sp[i], x_sp, f_sp, n))
plt.title('Splane10')
plt.plot(x_sp, sp10, color='green')
plt.plot(x_true, f_true, color = 'blue')
plt.grid()
plt.show()

sp21 = []
for i in range(n):
    sp21.append(Splane2_1(x_sp[i], x_sp, f_sp, n))
plt.title('Splane21')
plt.plot(x_sp, sp21, color='green')
plt.plot(x_true, f_true, color = 'blue')
plt.grid()
plt.show()

sp32 = []
for i in range(n):
    sp32.append(Splane3_2(x_sp[i], x_sp, f_sp, n))
plt.title('Splane32')
plt.plot(x_sp, sp32, color='green')
plt.plot(x_true, f_true, color = 'blue')
plt.grid()
plt.show()


z=10
x_optCH, f_optCH = optimal_points(a, b, z)
sp10 = []
print(x_optCH)
print(f_optCH)
for g in range(z):
    sp10.append(Splane1_0(x_optCH[g], x_optCH, f_optCH, z))
#print(sp10)
plt.title('Splane10 Opt')
plt.plot(x_optCH, sp10, color='green')
plt.plot(x_true, f_true, color = 'blue')
plt.grid()
plt.show()

x_optCH, f_optCH = optimal_points(a, b, z)
sp21 = []
print(x_optCH)
print(f_optCH)
for g in range(z):
    sp21.append(Splane2_1(x_optCH[g], x_optCH, f_optCH, z))
#print(sp10)
plt.title('Splane21 Opt')
plt.plot(x_optCH, sp21, color='green')
plt.plot(x_true, f_true, color = 'blue')
plt.grid()
plt.show()

x_optCH, f_optCH = optimal_points(a, b, z)
sp32 = []
print(x_optCH)
print(f_optCH)
for g in range(z):
    sp32.append(Splane3_2(x_optCH[g], x_optCH, f_optCH, z))
#print(sp10)
plt.title('Splane32 Opt')
plt.plot(x_optCH, sp32, color='green')
plt.plot(x_true, f_true, color = 'blue')
plt.grid()
plt.show()


q = 20
z=1000
x_optCH, f_optCH = optimal_points(a, b, 20)
sp32Check = []
for g in range(z):
    sp32Check.append(abs((Splane3_2(x_true[g], x_optCH, f_optCH, q))-f(x_true[g])))
#print(sp10)
plt.plot(x_true, sp32Check , color='green')
plt.grid()
plt.show()

myTable1 = PrettyTable(["Количество узлов(n)", "Максимальное отклонение(S10n)", "Максимальное отклонение(S10opt)"])
m = 100
max = 0
maxopt = 0
x_check = np.linspace(a , b , m)
f_check = f(np.array(x_check))
n = 20
h = 2


for h in range(4,n):
    max = 0
    maxopt = 0
    x_smCH = np.linspace(a, b, h)
    f_smCH = f(np.array(x_smCH))
    x_optCH, f_optCH = optimal_points(a, b, h)
    #print(x_optCH)
    #print(x_smCH)
    for i in range(m):
        k = Splane1_0(x_check[i], x_smCH, f_smCH,h)
        l = abs((f_check[i] - k))
        if l > max:
            max = l
    for i in range(m):
        o = Splane1_0(x_check[i], x_optCH, f_optCH,h)
        j = abs((f_check[i] - o))
        if j > maxopt:
            maxopt = j
    myTable1.add_row([h, max, maxopt])
print(myTable1)
print('m = ' + str(m))


myTable1 = PrettyTable(["Количество узлов(n)", "Максимальное отклонение(S21n)", "Максимальное отклонение(S21opt)"])
m = 100
max = 0
maxopt = 0
x_check = np.linspace(a , b , m)
f_check = f(np.array(x_check))
n = 20
h = 2


for h in range(4,n):
    max = 0
    maxopt = 0
    x_smCH = np.linspace(a, b, h)
    f_smCH = f(np.array(x_smCH))
    x_optCH, f_optCH = optimal_points(a, b, h)
    #print(x_optCH)
    #print(x_smCH)
    for i in range(m):
        k = Splane2_1(x_check[i], x_smCH, f_smCH,h)
        l = abs((f_check[i] - k))
        if l > max:
            max = l
    for i in range(m):
        o = Splane2_1(x_check[i], x_optCH, f_optCH,h)
        j = abs((f_check[i] - o))
        if j > maxopt:
            maxopt = j
    myTable1.add_row([h, max, maxopt])
print(myTable1)
print('m = ' + str(m))

myTable1 = PrettyTable(["Количество узлов(n)", "Максимальное отклонение(S32n)", "Максимальное отклонение(S32opt)"])
m = 100
max = 0
maxopt = 0
x_check = np.linspace(a , b , m)
f_check = f(np.array(x_check))
n = 20
h = 2


for h in range(4,n):
    max = 0
    maxopt = 0
    x_smCH = np.linspace(a, b, h)
    f_smCH = f(np.array(x_smCH))
    x_optCH, f_optCH = optimal_points(a, b, h)
    #print(x_optCH)
    #print(x_smCH)
    for i in range(m):
        k = Splane3_2(x_check[i], x_smCH, f_smCH,h)
        l = abs((f_check[i] - k))
        if l > max:
            max = l
    for i in range(m):
        o = Splane3_2(x_check[i], x_optCH, f_optCH,h)
        j = abs((f_check[i] - o))
        if j > maxopt:
            maxopt = j
    myTable1.add_row([h, max, maxopt])
print(myTable1)
print('m = ' + str(m))

