import os, time, requests, sys
H = {"x-api-key": os.environ["FINGRID_API_KEY"]}
for i in map(int, sys.argv[1:]):
    r = requests.get(f"https://data.fingrid.fi/api/datasets/{i}", headers=H, timeout=30)
    d = r.json() if r.ok else {}
    print(i, r.status_code, d.get("nameEn"), "|", d.get("dataPeriodEn"), "|", d.get("updateCadenceEn"), "|", (d.get("descriptionEn") or "")[:160].replace("\n"," "))
    time.sleep(2.1)
