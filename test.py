import numpy as np
import time

import numpy as np
from scipy.stats import norm

# 已知参数
e_value = 0.0536 * 0.4  # 设置期望值 e
k = 3        # 希望 P(X > 3e) < 1%
p = 0.063     # 概率阈值

# 标准正态分布分位数
z_p = norm.ppf(1 - p)

# 解方程求 sigma
ln_k = np.log(k)

# 定义要解的方程
def equation(sigma):
    return ln_k / sigma + sigma / 2 - z_p

# 使用数值方法求解 sigma
from scipy.optimize import fsolve

sigma_initial_guess = 1
sigma_solution = fsolve(equation, sigma_initial_guess)[0]

# 计算 mu
mu = np.log(e_value) - (sigma_solution**2) / 2

# 验证结果
E_X = np.exp(mu + (sigma_solution**2) / 2)
prob = 1 - norm.cdf((np.log(k * e_value) - mu) / sigma_solution)

print(f"计算得到的 sigma: {sigma_solution}")
print(f"计算得到的 mu: {mu}")
print(f"验证期望值 E[X]: {E_X}")
print(f"验证概率 P(X > {k}e): {prob * 100:.2f}%")

T = 0.04  # 假设正常设备的执行时间为 1 秒

delay_list = []

# 生成延迟并应用
for i in range(16):
    delay = round(np.random.lognormal(mean=mu, sigma=sigma_solution),4)
    delay_list.append(delay)
print(delay_list)

'''
计算得到的 sigma: 1.150749453936873
计算得到的 mu: -4.797278709610362
验证期望值 E[X]: 0.015999999999999997
验证概率 P(X > 3e): 6.30%
[0.0071, 0.0036, 0.0154, 0.0316, 0.0065, 0.0046, 0.0205, 0.0026, 0.0017, 0.0067, 0.0042, 0.0008, 0.0177, 0.0125, 0.0057, 0.0115]

'''
