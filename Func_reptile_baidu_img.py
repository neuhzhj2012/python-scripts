## coding=utf-8
"""根据搜索词下载百度图片"""
import re
import sys
import chardet
import requests
if sys.version_info[0]==2:
    import urllib
else:
    import urllib.parse as urllib


def get_onepage_urls(onepageurl):
    """获取单个翻页的所有图片的urls+当前翻页的下一翻页的url"""
    if not onepageurl:
        print('已到最后一页, 结束')
        return [], ''
    try:
        if sys.version_info[0]==3:
            onepageurl = onepageurl.encode(encoding="utf-8")
        html = requests.get(onepageurl).text #unicode需要转换为utf-8编码格式
        if sys.version_info[0] == 2:
            html = html.encode('utf-8')
        # encode_type = chardet.detect(html)
        # html = html.decode(encode_type['encoding'])
    except Exception as e:
        print(e)
        pic_urls = []
        fanye_url = ''
        return pic_urls, fanye_url
    pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)
    # fanye_urls = re.findall(re.compile(r'<a href="(.*)" class="n">下一页</a>'), html, flags=0)
    pattern = '<a href="(.*)" class="n">下一页</a>'
    fanye_urls = re.findall(re.compile(pattern), html)
    fanye_url = 'http://image.baidu.com' + fanye_urls[0] if fanye_urls else ''
    return pic_urls, fanye_url


def down_pic(pic_urls):
    """给出图片链接列表, 下载所有图片"""
    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=15)
            string = str(i + 1) + '.jpg'
            with open(string, 'wb') as f:
                f.write(pic.content)
                print('成功下载第%s张图片: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            print('下载第%s张图片时失败: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue


if __name__ == '__main__':
    keywords = ['猫','狗','仓鼠']  # 关键词, 改为你想输入的词即可, 相当于在百度图片里搜索一样
    url_init_first = r'http://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1497491098685_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1497491098685%5E00_1519X735&word='

    all_pic_urls = []
    default_urls = 10

    for keyword in keywords:
        fanye_count = 1  # 累计翻页数
        url_init = url_init_first + urllib.quote(keyword, safe='/')
        onepage_urls, fanye_url = get_onepage_urls(url_init)
        if fanye_url == '' and onepage_urls == []:
            continue
        print('%s 第%s页' % (keyword, fanye_count))
        all_pic_urls.extend(onepage_urls)
        while 1:
            onepage_urls, fanye_url = get_onepage_urls(fanye_url)
            fanye_count += 1

            if fanye_url == '' and onepage_urls == []:
                break
            print('%s 第%s页' % (keyword, fanye_count))
            all_pic_urls.extend(onepage_urls)
            if fanye_count > default_urls:
                break

    down_pic(list(set(all_pic_urls)))