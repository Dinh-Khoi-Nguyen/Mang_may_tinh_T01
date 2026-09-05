import pycurl
from io import BytesIO
from pythonping import ping
from urllib.parse import urlsplit
import datetime
import pandas as pd
from pathlib import Path
import tldextract

iterations = 10
urls = [
        'https://x.com/',
        'https://www.amazon.com/',
        'https://www.google.com/',
        'https://www.bing.com/',
        'https://www.youtube.com/',
        'https://www.whatsapp.com/',
        'https://www.stackoverflow.com/',
        'https://www.wikipedia.org/',
        'https://www.microsoft.com/',
        'https://wordpress.org/',
        'https://www.apple.com/',
        'https://www.php.net/',
        'https://www.mozilla.org/',
        'https://github.com/',
        'https://www.slideshare.net/'
        ]
dat = {
    'dns_time_ms': [],
    'tcp_time_ms': [],
    'tls_time_ms': [],
    'rtt_ms': [],
    'http_status': [],
    'response_size_bytes': [],
    'time_of_day': [],
    'domain_type': [],
    'response_time_ms': []
}

for url in urls: 
    url_components = tldextract.extract(url, include_psl_private_domains=True)
    for i in range(iterations):
        buffer = BytesIO()
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(c.WRITEDATA, buffer)
        c.perform()
        rtt=(ping(urlsplit(url).netloc).rtt_avg_ms)
        a = datetime.datetime.now()

        dat['dns_time_ms'].append(c.getinfo((pycurl.NAMELOOKUP_TIME))*1000)
        dat['tcp_time_ms'].append(c.getinfo((pycurl.CONNECT_TIME))*1000)
        dat['tls_time_ms'].append((c.getinfo((pycurl.APPCONNECT_TIME)) - c.getinfo(pycurl.CONNECT_TIME))*1000)
        dat['http_status'].append(c.getinfo(pycurl.HTTP_CODE))
        dat['response_time_ms'].append(c.getinfo(pycurl.TOTAL_TIME)*1000)

        c.close()

        response = buffer.getvalue()
        dat['response_size_bytes'].append(len(response.decode('iso-8859-1')))
        dat['rtt_ms'].append(rtt)
        dat['time_of_day'].append(int(a.strftime('%H') + a.strftime('%M') + a.strftime('%S')))
        dat['domain_type'].append(url_components.suffix)

df = pd.DataFrame(dat)
if Path('D:/Dataset.csv').exists():
    df.to_csv('D:/Dataset.csv', index=False, header=False, mode='a')
else:
    df.to_csv('D:/Dataset.csv', index=False)
