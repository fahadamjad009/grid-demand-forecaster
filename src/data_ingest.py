"""Pull AEMO NEM price & demand history for NSW1 (last 36 months, free CSVs, no API key)."""
import time
import pandas as pd
import requests
from pathlib import Path
from datetime import date

REGION = "NSW1"
MONTHS_BACK = 36
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
URL_TMPL = "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{yyyymm}_{region}.csv"

def month_range(n):
    today = date.today()
    y, m = today.year, today.month
    out = []
    for _ in range(n):
        out.append(f"{y}{m:02d}")
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return out

def fetch_month(yyyymm: str) -> pd.DataFrame | None:
    fname = RAW_DIR / f"{yyyymm}_{REGION}.csv"
    if fname.exists():
        return pd.read_csv(fname)
    url = URL_TMPL.format(yyyymm=yyyymm, region=REGION)
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200 or not r.text.startswith("REGION"):
        print(f"  skip {yyyymm}: status {r.status_code}")
        return None
    fname.write_text(r.text)
    return pd.read_csv(fname)

def main():
    frames = []
    for yyyymm in month_range(MONTHS_BACK):
        print(f"Fetching {yyyymm}...")
        df = fetch_month(yyyymm)
        if df is not None:
            frames.append(df)
        time.sleep(1)

    if not frames:
        raise RuntimeError("No data fetched — check the URL pattern or network access.")

    full = pd.concat(frames, ignore_index=True)
    full["SETTLEMENTDATE"] = pd.to_datetime(full["SETTLEMENTDATE"])
    full = full.sort_values("SETTLEMENTDATE").drop_duplicates("SETTLEMENTDATE")
    out_path = Path("data/nem_price_demand.csv")
    full.to_csv(out_path, index=False)
    print(f"\nSaved {len(full):,} rows spanning {full['SETTLEMENTDATE'].min()} to {full['SETTLEMENTDATE'].max()}")
    print(f"-> {out_path}")

if __name__ == "__main__":
    main()