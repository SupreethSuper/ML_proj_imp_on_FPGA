from pygooglenews import GoogleNews
import time

gn = GoogleNews()
query = 'S&P 500 OR stock market OR wall street'

t0 = time.time()
r1 = gn.search(query, from_='2024-06-01', to_='2024-06-08')
t1 = time.time()
r2 = gn.search(query, from_='2024-06-08', to_='2024-06-15')
t2 = time.time()

print(f"Fetch 1: {len(r1['entries'])} entries in {t1-t0:.1f}s")
print(f"Fetch 2: {len(r2['entries'])} entries in {t2-t1:.1f}s")

e = r1['entries'][0]
print(f"Sample: {e['published']} | {e['title']}")
print(f"Source: {e.get('source', {}).get('title', 'N/A')}")

# Check date parsing
from datetime import datetime
import email.utils
for entry in r1['entries'][:5]:
    parsed = email.utils.parsedate_to_datetime(entry['published'])
    print(f"  {parsed.strftime('%Y-%m-%d')} | {entry['title'][:80]}")
