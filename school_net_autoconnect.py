#coding=utf-8
import requests
import time

def scho():
    addr = "http://1.1.1.3/ac_portal/login.php"

    header = {
        "Connection": "keep-alive",
        "Content-Length": "91",
        "Accept": "*/*",
        "DNT": "1",
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36 Edg/106.0.1370.37",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "http://1.1.1.3",
        "Referer": "http://1.1.1.3/ac_portal/20191008142058/pc.html?template=20191008142058&tabs=pwd-cas&vlanid=0&_ID_=0&switch_url=&url=http://1.1.1.3/homepage/index.html&cas_url=https%3A%2F%2Fauthserver.gznu.edu.cn%2Fauthserver%2Flogin%3Fservice%3Dhttp%3A%2F%2F1.1.1.3%3A80%2Fcas%5Fauth%3Furl%3Dhttp%3A%2F%2F1.1.1.3%2Fhomepage%2Findex.html&controller_type=&mac=2c-f0-5d-9c-13-b5",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6"
    }
opr=pwdLogin&userName=21010210672&pwd=ff49d95bf21f79b87d46fdc362&auth_tag=1670991341307&rememberPwd=0
    data = {'opr':'pwdLogin',
        'userName':'21010210672',
        'pwd':'ff49d95bf21f79b87d46fdc362',
        'auth_tag':'1670991341307',
        'rememberPwd':'0'}

    post1 = requests.post(addr, data = data, headers=header)
    print(post1.content.decode())
    print(post1.status_code)


while True:
    scho()
    time.sleep(3600)

