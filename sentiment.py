from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# data = pd.read_csv("stocks_price.txt", usecols=["Date", "Open", "High", "Low", "Close"]).iloc[1663:0:-1].reset_index(drop=True)
data = pd.read_csv(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\YahooData\AAPL.csv",
                   usecols=["High"])

def rolling_average(d, size=30):
    out = [d[0]]*(size-1)
    for i in range(size, len(d)):
        out.append(np.mean(d[i-size: i]))
    return np.array(out)

def detect_shape(avg, size, thresh):
    ret = {"steep": [], "flat": []}
    for i in range(len(avg)//size):
        if np.std(avg[i*size: (i+1)*size]) > thresh:
            ret["steep"].append(i*size)
        else:
            ret["flat"].append(i*size)
    return ret

def rolling_std(d, size):
    out = [0]*size
    for i in range(size, len(d)):
        out.append(np.std(d[i-size: i])*10)
    out.extend([0]*size)
    return out

def plateaus(m):
    count = 0
    ind = 0
    ret = []
    for i, elem in enumerate(m):
        if elem > 0:
            count += 1
            if count == 1:
                ind = i
        else:
            if count >= 5:
                ret.append([ind, ind+count])
            ind = 0
            count = 0
    return ret

# std = rolling_std(data.High, 10)
# marks = [data.High[i] if 0 < elem < 10 else 0 for i, elem in enumerate(std)]
"""USE PLATEAUS"""
# p = plateaus(marks)

# prediction = [1 if data.High[p1] < data.High[p2-1] else 0 for [p1, p2] in p]
# truth = [1 if data.High[p2 + 4] > (data.High[p1] + data.High[p2-1])/2 else 0 for [p1, p2] in p]
# truth = []
# for [p1, p2] in p:
#     try:
#         if data.High[p2 + 4] > (data.High[p1] + data.High[p2-1])/2:
#             truth.append(1)
#         else:
#             truth.append(0)
#     except KeyError:
#         truth.append(0)
#
# score = 0
# for i in range(len(prediction)):
#     if prediction[i] == truth[i]:
#         score += 1
# print(score/len(truth))

"""THIS ALSO GOOD"""
roll1 = rolling_average(data.High, size=10)
roll2 = rolling_average(data.High, size=30)

plt.plot(roll1, label="r1")
plt.plot(roll2, label="r2")
plt.plot(data.High, label="High")
# plt.scatter(range(len(marks)), marks, marker=".", c="red")
# plt.plot(r_avg, label="RollingAverage")
# plt.scatter(x=shapes["flat"], y=data.High[shapes["flat"]], marker="+")
# plt.scatter(x=shapes["steep"], y=data.High[shapes["steep"]], marker=".")
plt.legend()
plt.show()