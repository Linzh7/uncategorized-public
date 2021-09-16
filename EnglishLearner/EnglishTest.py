import csv
import random
import requests
from playsound import playsound

ls = []


def do(ls):
    random.shuffle(ls)
    for i in ls:
        print(i[1], end='')
        playsound(i[1] + '.mp3')
        input()
        print(i[2])


with open('./1.csv', 'r') as f:
    lines = csv.reader(f)
    for i in lines:
        ls.append(i)

print("mode?")
mode = int(input())

if mode == 0:
    do(ls)
elif mode == 99:
    for i in ls:
        res = requests.get('http://dict.youdao.com/dictvoice?type=1&audio=' +
                           i[1])
        music = res.content
        with open('./' + i[1] + '.mp3', 'wb') as file:
            file.write(res.content)
            file.flush()
    print('done.')
