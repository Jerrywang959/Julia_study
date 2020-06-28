# 用Python实现分词
from ltp import LTP
ltp = LTP()

segment, hidden = ltp.seg(["他叫汤姆去拿外衣。"])


segment, _ = ltp.seg(["他叫汤姆去拿外衣。"])

