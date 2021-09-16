import base64
import requests

action = 'login'
username = ''
password = ''.encode("utf-8")
ac_id = '64'
user_ip = ''

post_addr = ""
post_header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'Accept-Encoding': 'gzip, deflate',
    'Content-Type': 'application/x-www-form-urlencoded',
    'X-Requested-With': 'XMLHttpRequest',
    'Origin': '',
    'Referer': '',
    'Content-Length': '140',
    'Cookie': '',
    'Host': '',
    'Connection': 'keep-alive',
}

post_data = {
    'action': action,
    'username': username,
    'password': '{B}'+str(base64.b64encode(password), 'utf-8'),
    'ac_id': ac_id,
    'user_ip': user_ip
}
z = requests.post(post_addr, data=post_data, headers=post_header)
# s = z.text.encode('utf8')
# print(s)
print("login success!")
