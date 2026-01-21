import urllib.request, json
url='http://127.0.0.1:5000/api/dashboard_data?region=CAPITAL&faccao=TODAS&tipo_crime=TODOS'
with urllib.request.urlopen(url, timeout=10) as resp:
    data=json.load(resp)
feat = data['polygons']['features'][0]
props = feat['properties']
print('name:', props.get('name'))
print('risk_by raw repr:', repr(props.get('risk_by')))
print('risk_by type:', type(props.get('risk_by')))
print('predicted_target repr:', repr(props.get('predicted_target')))
print('all keys:', list(props.keys()))
