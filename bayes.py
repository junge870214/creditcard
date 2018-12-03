import re,collections
text=open('/Users/wujie/学习/机器学习/python/01、【非加密】python数据分析与机器学习实战/课程资料/唐宇迪-机器学习课程资料/机器学习算法配套案例实战/贝叶斯-拼写检查器/big.txt')
wordtext1= re.findall(r'[a-z]+',text.read().lower())
def P_word(word):
    _num=1
    for x in wordtext1:
        if word ==x:
            _num+=1
    return _num

print(P_word('the'))