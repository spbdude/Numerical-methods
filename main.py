import numpy as np
import matplotlib.pyplot as plt
import random
from prettytable import PrettyTable


a = -1
b = 1
m = 120
nmax = 25
eps = 0.00001


def f(x):
    res = x * np.tan(x)
    return res

x = np.linspace(a, b)
plt.plot(x, f(x), color='green')
plt.title('True Function X*Tg(X)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

def MNK_N(x, y, n):
    Q = np.vander(x, N = n+1)
    H = Q.T @ Q
    b = Q.T @ y
    return np.linalg.solve(H, b)

def alpha(q_curr,x):
  q_i2 = np.polyval(q_curr,x)**2;
  ret = sum(x*q_i2)/sum(q_i2);
  return ret

def beta(q_prev,q_curr,x):
  qi = np.polyval(q_prev, x);
  q_i = np.polyval(q_curr, x);
  ret = sum(x*(q_i)*(qi))/sum((qi)**2);
  return ret

def MNK_O(x, y, n):
    q_prev = np.array([1])
    q_curr = np.array([1, -sum(x)/len(x)])
    a_prev = sum(np.polyval(q_prev,x)*y)/sum(np.polyval(q_prev,x)**2)
    a_curr = sum(np.polyval(q_curr,x)*y)/sum(np.polyval(q_curr,x)**2)
    q = np.polyadd(a_prev*q_prev, a_curr*q_curr)
    for i in range(2, n+1):
      alph_q = alpha(q_curr,x)*q_curr
      beta_q = beta(q_prev, q_curr, x)*q_prev
      q_next = np.hstack([q_curr, 0])
      q_next = np.polyadd(q_next, -alph_q)
      q_next = np.polyadd(q_next, -beta_q)
      q_prev = q_curr
      q_curr = q_next
      a_curr = sum(np.polyval(q_curr,x)*y)/sum(np.polyval(q_curr,x)**2)
      q = np.polyadd(q, a_curr*q_curr)
    return q

def Data(start, end, m):
    xs = np.linspace(start, end, m)
    ys = f(xs)
    x = np.repeat(xs, 3)
    y = np.repeat(ys, 3) + np.random.uniform(-eps,eps)
    return xs, ys, x, y

def Error(p, x, ys):
    error = np.polyval(p, x) - ys
    return error @ error

xs, ys, x, y = Data(a, b, m)

x_lp = np.linspace(a, b)
myTable = PrettyTable(["Степень полинома (n)", "Сумма квадратов ошибок для МНК (нормальные уравнения)", "Сумма квадратов ошибок для МНК (ортогональные полиномы)"])
for n in range(1, nmax+1):
    Normal = MNK_N(x, y, n)
    error_Normal = Error(Normal, xs, ys)
    Ort = MNK_O(x, y, n)
    error_Ort = Error(Ort, xs, ys)
    myTable.add_row([n, error_Normal, error_Ort])
print(myTable)


for n in range(1, 5+1):
    A_Norm = MNK_N(x, y, n)
    plt.plot(x_lp, np.polyval(A_Norm, x_lp), x, y,'o')
    number = str(n)
    title = "Norm Functions Approximation n = " + number
    print(title)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

for n in range(1, 5+1):
    A_Ort = MNK_O(x, y, n)
    plt.plot(x_lp, np.polyval(A_Ort, x_lp), x, y,'o')
    number = str(n)
    title = "Ort Functions Approximation n = " + number
    print(title)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()


