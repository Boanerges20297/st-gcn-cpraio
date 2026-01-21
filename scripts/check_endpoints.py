import urllib.request
import urllib.parse

BASE = 'http://127.0.0.1:5000'
ENDPOINTS = [
    '/api/dashboard_data?region=CAPITAL&faccao=TODAS&tipo_crime=TODOS',
    '/api/strategic_insights',
    '/api/strategic_insights_range?data_inicio=2025-01-01&data_fim=2025-12-31&regiao=CAPITAL'
]

for ep in ENDPOINTS:
    url = BASE + ep
    print('\n> GET', url)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            ctype = resp.getheader('Content-Type')
            body = resp.read(1024*200)
            print('Status:', status)
            print('Content-Type:', ctype)
            print('Body (truncated):')
            try:
                print(body.decode('utf-8')[:10000])
            except Exception:
                print(body[:500])
    except Exception as e:
        print('Erro ao acessar endpoint:', e)
