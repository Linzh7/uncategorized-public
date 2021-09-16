import requests
import re
import os

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36'}
name = input('image label:')
if not os.path.exists('./{}'.format(name)):
    os.mkdir(name)
num = 0
x = input('how many?')
for i in range(int(x)):
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+name+'&pn='+str(i*30)
    res = requests.get(url, headers=headers)
    htlm_1 = res.content.decode()
    a = re.findall('"objURL":"(.*?)",', htlm_1)
    for b in a:
        num = num + 1
        try:
            img = requests.get(b)
        except Exception as e:
            print('cant '+str(num))
            print(str(e))
            continue
        f = open('./{}/{}.jpg'.format(name, num), 'ab')
        print('downloading {}/{}.jpg')
        f.write(img.content)
        f.close()
print('done.')
