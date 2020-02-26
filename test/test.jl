# %% 测试用hydrogen使用julia内核
using CSV
print(1)
print("test")
print("成功了吗")
using PyCall
using RCall

# %%
using PyCall
py"""
import pandas
#from datetime import timedelta
print("1+2")
"""

# %%
print("I love you")
using GR
