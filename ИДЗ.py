import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def First(data):
    # 1. Ввод данных
    print("Введите шаг: ")
    h = float(input())

    # 2. Построение вариационного ряда
    variational_series = np.sort(data)
    print("\nВариационный ряд:", variational_series)

    # 3. Построение эмпирической функции распределения
    def empirical_cdf(x, data):
        return np.sum(data <= x) / len(data)

    x_values = np.linspace(min(data) - 1, max(data) + 1, 1000)
    y_values = [empirical_cdf(x, data) for x in x_values]

    plt.figure(figsize=(10, 5))
    plt.step(x_values, y_values, where='post')
    plt.title('Эмпирическая функция распределения')
    plt.xlabel('Значения')
    plt.ylabel('F(x)')
    plt.grid(True)
    plt.show()

    # 4. Построение гистограммы и полигона частот
    bins = np.arange(min(data), max(data) + h, h)
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, density=True, edgecolor='black', alpha=0.7, label='Гистограмма')

    # Полигон частот
    hist, bin_edges = np.histogram(data, bins=bins,  density=True,)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(midpoints, hist, 'r-', marker='o', label='Полигон частот')

    plt.title('Гистограмма и полигон частот')
    plt.xlabel('Значения')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True)
    plt.show()
def Second(data):
    probsum=0
    # a) Выборочное среднее (математическое ожидание)
    mean = np.mean(data)
    print("Выборочное среднее: ", round(mean,4))
    # b) Выборочная дисперсия (смещенная и несмещенная)
    var_biased = np.var(data, ddof=0)  # Смещенная (N в знаменателе)
    print("Смещённая выборочная дисперсия: ", round(var_biased,4))
    var_unbiased = np.var(data, ddof=1)  # Несмещенная (N-1 в знаменателе)
    print("Немещённая выборочная дисперсия: ", round(var_unbiased,4))
    # c) Выборочное СКО
    std = np.std(data)
    print("Выборочное СКО: ", round(std,4))
    # d) Медиана
    median = np.median(data)
    print("Медиана: ", round(median,4))
    # e) Коэффициент асимметрии
    skewness = stats.skew(data)
    print("Коффициент ассимитрии: ", round(skewness,4))
    # f) Коэффициент эксцесса
    kurtosis = stats.kurtosis(data)
    print("Коффициент эксцесса: ", round(kurtosis,4))
    # g) Вероятность P(X ∈ [c,d])
    for i in data:
        if i>=-5.04 and i<=-4.84:
            probsum+=1
    print("Вероятность попадания в интервал: ", round(probsum/50,4))
def Fifth(data):
    sec=[(i-1)/50 for i in range(1,51)]
    th=[i/50 for i in range(1,51)]
    fo=np.sort(data)
    F0 = stats.norm.cdf(fo, loc=-5.00, scale=0.2)
    ld=[F0[i]-sec[i] for i in range(0,50)]
    ud=[th[i]-F0[i] for i in range(0,50)]
    print(round(max(max(ld),max(ud)),4))
def Sixth(data):
    X=0
    sr =np.sort(data)
    count=0
    sm=0
    buf=0
    for i in sr:
        count+=1
        if count==1:
            if i!= sr[0]:
                print(i,"), ", end="",sep="")
                buf+=stats.norm.cdf(i, loc=-5, scale=0.2)
                sm+=buf
                X+=((5-50*buf)**2)/(50*buf)
                buf=0
                print("[",i,end=",",sep="")
                buf-=stats.norm.cdf(i, loc=-5, scale=0.2)
            else:
                print("(-inf", end=",", sep="")
                buf -= stats.norm.cdf(-np.inf, loc=-5, scale=0.2)
        elif count == 5:
            if i == sr[-1]:
                print("+inf)",sep="")
                buf += stats.norm.cdf(np.inf, loc=-5, scale=0.2)
                sm += buf
                X += ((5 - 50 * buf) ** 2) / (50 * buf)
                buf = 0
            count=0
    print(sm, X)
def Seventh(data):
    X = 0
    sr = np.sort(data)
    count = 0
    sm = 0
    buf = 0
    for i in sr:
        count += 1
        if count == 1:
            if i != sr[0]:
                buf += stats.norm.cdf(i, loc=-4.9839, scale=0.2033)
                sm += buf
                X += ((5 - 50 * buf) ** 2) / (50 * buf)
                buf = 0
                buf -= stats.norm.cdf(i, loc=-4.9839, scale=0.2033)
            else:
                buf -= stats.norm.cdf(-np.inf, loc=-4.9839, scale=0.2033)
        elif count == 5:
            if i == sr[-1]:
                buf += stats.norm.cdf(np.inf, loc=-4.9839, scale=0.2033)
                sm += buf
                X += ((5 - 50 * buf) ** 2) / (50 * buf)
                buf = 0
            count = 0
    print(sm, X)
F=open("Data.txt")
a = F.readline().split()
b = [float(x) for x in a]
print("Введите номер задания:")
n=int(input())
if n==1:
    First(b)
elif n==2:
    Second(b)
elif n==5:
    Fifth(b)
elif n==6:
    Sixth(b)
elif n==7:
    Seventh(b)