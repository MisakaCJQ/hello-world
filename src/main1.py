# 一元线性回归
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 设定绘制图的参数
def runplt(title, xlb, ylb):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    plt.grid(True)
    plt.xlim(0, 25)
    plt.ylim(-5, 25)
    return plt


# 直线方程，输入x以及两个参数，返回y值
def linearfunc(theta, x):
    return theta[0] + theta[1] * x


# 进行一次渐进迭代
def gradlient(x, y, theta, alpha):
    m = len(x)
    sum0 = 0.0
    sum1 = 0.0
    h = 0
    for i in range(m):
        h = linearfunc(theta, x[i])
        sum0 += h - y[i]
        sum1 += (h - y[i]) * x[i]

    theta[0] -= alpha * sum0 / m
    theta[1] -= alpha * sum1 / m
    return theta


# 批量梯度下降法计算参数
def bgd(x, y, theta, alpha=0.01, iterations=1000):
    for i in range(iterations):
        theta = gradlient(x, y, theta, alpha)
    return theta


# 使用panda读入数据
def loaddata():
    df1 = pd.read_table('ex1data1.txt', header=None, sep=',')
    df1.columns = ['population', 'profit']
    # 将两列数据读取到两个numpy数组
    data0 = np.array(df1[['population']])
    data1 = np.array(df1[['profit']])
    return data0, data1


# 读入两列数据
X, Y = loaddata()
# 设置参数并输出数据散点图
plt = runplt('Profit plotted against population', 'Population', 'Profit')
plt.scatter(X, Y, c='r', marker='o', label='real data')
plt.legend(loc='best')
plt.show()

# 初始化参数为0
theta0 = [0, 0]
# 调用批量梯度下降法求解两个参数,设置alpha为0.01,迭代次数为1500
theta0 = bgd(X, Y, theta0, 0.01, 1500)

# 设置参数并输出数据拟合图
plt = runplt('Fitting graph', 'Population', 'Profit')
plt.scatter(X, Y, c='r', marker='o', label='real data')
plt.plot(X, theta0[0] + theta0[1] * X, label='predict data')
plt.legend(loc='best')
plt.show()

print('使用梯度下降法求解得到两个参数：')
print('θ0：{:.6f} θ1:{:.6f}'.format(float(theta0[0]), float(theta0[1])))
print('人口数为35000时的利润:{:.6f}万美元'.format(float(theta0[0] + 3.5 * theta0[1])))
print('人口数为70000时的利润:{:.6f}万美元'.format(float(theta0[0] + 7 * theta0[1])))

