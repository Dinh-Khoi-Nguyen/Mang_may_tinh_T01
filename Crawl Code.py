import pycurl
from io import BytesIO
from pythonping import ping
from urllib.parse import urlsplit
import datetime
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "input"
OUT_DIR.mkdir(exist_ok=True)

iterations = 10
urls = [
        'https://x.com/',
        'https://www.facebook.com/',
        'https://www.google.com/',
        'https://www.tiktok.com/',
        'https://www.youtube.com/',
        'https://shopee.vn/',
        'https://tuoitre.vn/',
        'https://www.wikipedia.org/',
        'https://vietjack.com/',
        'https://baomoi.com/',
        'https://www.apple.com/',
        'https://www.pinterest.com/',
        'https://www.microsoft.com/',
        'https://coccoc.com/',
        'https://www.lazada.vn/'
        ]
categories = {
    'News': {'https://tuoitre.vn/', 'https://baomoi.com/'},
    'Technology': {'https://www.apple.com/', 'https://www.microsoft.com/'},
    'Search engine': {'https://www.google.com/', 'https://coccoc.com/'},
    'Shopping': {'https://shopee.vn/', 'https://www.lazada.vn/'},
    'Social networking': {'https://www.facebook.com/', 'https://www.tiktok.com/', 'https://x.com/', 'https://www.pinterest.com/'},
    'Entertainment': {'https://www.youtube.com/'},
    'Reference': {'https://www.wikipedia.org/'},
    'Education': {'https://vietjack.com/'}
}
dat = {
    'dns_time_ms': [],
    'tcp_connect_ms': [],
    'tls_time_ms': [],
    'rtt_ms': [],
    'http_status': [],
    'response_size': [],
    'time_of_day': [],
    'domain_category': [],
    'response_time_ms': []
}

for url in urls: 
    for i in range(iterations):
        buffer = BytesIO()
        c = pycurl.Curl()
        c.setopt(pycurl.URL, url)
        c.setopt(pycurl.WRITEDATA, buffer)
        c.setopt(pycurl.FOLLOWLOCATION, True)
        a = datetime.datetime.now()
        p = ping(urlsplit(url).netloc, count=1)
        c.perform()
        dat['dns_time_ms'].append(c.getinfo((pycurl.NAMELOOKUP_TIME))*1000)
        dat['tcp_connect_ms'].append((c.getinfo(pycurl.CONNECT_TIME) - c.getinfo(pycurl.NAMELOOKUP_TIME))*1000)
        dat['tls_time_ms'].append((c.getinfo(pycurl.APPCONNECT_TIME) - c.getinfo(pycurl.CONNECT_TIME))*1000)
        dat['http_status'].append(c.getinfo(pycurl.HTTP_CODE))
        dat['response_time_ms'].append(c.getinfo(pycurl.TOTAL_TIME)*1000)

        c.close()

        response = buffer.getvalue()
        dat['response_size'].append(len(response.decode('iso-8859-1')))
        rtt=(p.rtt_avg_ms)
        dat['rtt_ms'].append(rtt)
        dat['time_of_day'].append(int(a.strftime('%H') + a.strftime('%M') + a.strftime('%S')))
        
    for category in categories:
        if url in categories[category]:
            domain_category = category
            for i in range(iterations):
                dat['domain_category'].append(domain_category)
            categories[category].remove(url)
            break

df = pd.DataFrame(dat)
for i in range(len(dat['tcp_connect_ms'])):
    if dat['rtt_ms'][i]-dat['tcp_connect_ms'][i]>0:
        df.drop(i, inplace=True)
if Path(OUT_DIR / 'Dataset_finale.csv').exists():
    df.to_csv(OUT_DIR / 'Dataset_finale.csv', index=False, header=False, mode='a')
else:
    df.to_csv(OUT_DIR / 'Dataset_finale.csv', index=False)
