"""
news_catch.py - v5 (Google News via pygooglenews)

Fetches daily financial news headlines from Google News for each trading day
in SP500V3.xlsx and writes results to SP500V3_news.xlsx.

Strategy:
  - Uses pygooglenews (Google News RSS) for fast, reliable headline fetching
  - 7-day chunks with parallel ThreadPoolExecutor (4 workers)
  - JSON cache so interrupted runs resume seamlessly
  - Parses headlines, source, and publication date per entry

New columns in SP500V3_news.xlsx:
  - news_headlines       : top headlines per day (semicolon-separated)
  - news_article_count   : number of financial articles that day

Usage:
  python news_catch.py
  python news_catch.py --workers 6          # more parallel workers
  python news_catch.py --clear-cache        # force re-fetch
"""

import os
import json
import time
import random
import argparse
import email.utils
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import numpy as np
from pygooglenews import GoogleNews

CHUNK_DAYS = 30
DEFAULT_WORKERS = 16
MAX_RETRIES = 10
MAX_HEADLINES_PER_DAY = 120

QUERIES = [
    # --- MACRO (market drivers) ---
    "Federal Reserve interest rates",
    "FOMC meeting results",
    "US inflation CPI data",
    "US unemployment rate",
    "GDP growth United States",
    "bond yields US 10 year",
    "interest rate hikes US",
    "economic outlook United States",

    # --- MARKET SENTIMENT ---
    "stock market crash",
    "bullish stock market outlook",
    "bearish market sentiment",
    "market volatility spike",
    "recession fears US",
    "stock market rally",
    "market sell off",
    "investor sentiment US equities",

    # --- SECTOR LEVEL ---
    "tech stocks rally",
    "semiconductor industry news",
    "AI stocks surge",
    "energy sector oil prices",
    "banking sector crisis",
    "financial sector news US",
    "healthcare stocks news",
    "consumer spending trends US",

    # --- COMMODITIES (affect market indirectly) ---
    "crude oil prices",
    "Brent oil news",
    "gold prices US",
    "inflation commodities impact",
    "energy market outlook",

    # --- TOP S&P 500 COMPANIES ---
    "Apple earnings",
    "Microsoft earnings",
    "Nvidia stock news",
    "Amazon quarterly results",
    "Tesla stock movement",
    "Google earnings",
    "Meta earnings",
    "S&P 500 top companies earnings",

    # --- GENERAL INDEX CONTEXT ---
    "S&P 500 outlook",
    "S&P 500 forecast",
    "US stock market today",
    "Wall Street news",
    "Dow Jones and Nasdaq update",
]

# Thread-safe locks
_cache_lock = threading.Lock()
_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs, flush=True)


def fetch_gnews_chunk(gn, query, start_date, end_date):
    """Fetch articles from Google News for a date range with retries."""
    from_str = start_date.strftime("%Y-%m-%d")
    to_str = end_date.strftime("%Y-%m-%d")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = gn.search(query, from_=from_str, to_=to_str)
            return result.get("entries", [])
        except Exception as e:
            wait = 2 * attempt + random.uniform(0, 2)
            if attempt < MAX_RETRIES:
                time.sleep(wait)
            else:
                safe_print(f"      [{type(e).__name__}] All retries failed for {from_str}->{to_str}")
    return []


def parse_entries_by_date(entries):
    """Group Google News entries by date."""
    daily = defaultdict(list)
    for entry in entries:
        title = entry.get("title", "").strip()
        published = entry.get("published", "")
        source = entry.get("source", {}).get("title", "")

        if not title or not published:
            continue

        # Remove source suffix from title (e.g. "Headline - Reuters" -> "Headline")
        if source and title.endswith(f" - {source}"):
            title = title[: -(len(source) + 3)].strip()

        try:
            dt = email.utils.parsedate_to_datetime(published)
            date_key = dt.strftime("%Y-%m-%d")
        except Exception:
            continue

        daily[date_key].append({"title": title, "source": source})

    return dict(daily)


def date_chunks(start_date, end_date, chunk_days=CHUNK_DAYS):
    """Generate date range chunks."""
    chunks = []
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def fetch_chunk(chunk_info, cache, cache_path):
    """Fetch all queries for a single time chunk -- called from thread pool."""
    idx, total, cs, ce = chunk_info
    chunk_key = f"{cs.strftime('%Y%m%d')}_{ce.strftime('%Y%m%d')}"
    cs_label = cs.strftime("%Y-%m-%d")
    ce_label = ce.strftime("%Y-%m-%d")

    # Check cache first
    with _cache_lock:
        if chunk_key in cache:
            chunk_daily = cache[chunk_key]
            n_art = sum(len(v) for v in chunk_daily.values())
            safe_print(f"  [{idx:>3}/{total}] {cs_label} -> {ce_label}  CACHED  ({n_art} articles)")
            return chunk_daily

    # Fetch from Google News (one request per query, merge results)
    safe_print(f"  [{idx:>3}/{total}] {cs_label} -> {ce_label}  fetching ...")

    gn = GoogleNews()  # each thread gets own instance
    all_entries = []
    for q in QUERIES:
        time.sleep(random.uniform(0.3, 1.0))  # polite stagger between queries
        entries = fetch_gnews_chunk(gn, q, cs, ce)
        all_entries.extend(entries)

    chunk_daily = parse_entries_by_date(all_entries)

    # Deduplicate titles within each day
    for date_key in chunk_daily:
        seen_titles = set()
        unique = []
        for art in chunk_daily[date_key]:
            title_lower = art["title"].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique.append(art)
        chunk_daily[date_key] = unique

    n_art = sum(len(v) for v in chunk_daily.values())
    safe_print(f"  [{idx:>3}/{total}] {cs_label} -> {ce_label}  -> {n_art} articles, {len(chunk_daily)} days")

    # Save to cache
    with _cache_lock:
        cache[chunk_key] = chunk_daily
        with open(cache_path, "w") as f:
            json.dump(cache, f)

    return chunk_daily


def main():
    parser = argparse.ArgumentParser(description="Fetch Google News headlines for SP500V3")
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir.parent / "datasets" / "SP500V3.xlsx"
    default_output = script_dir.parent / "datasets" / "SP500V3_news.xlsx"

    parser.add_argument("--input", "-i", default=str(default_input))
    parser.add_argument("--output", "-o", default=str(default_output))
    parser.add_argument("--cache", default=str(script_dir / "gnews_cache.json"))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    # -- Read spreadsheet --
    print(f"Reading {args.input} ...", flush=True)
    df = pd.read_excel(args.input)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    start_date = df["observation_date"].min()
    end_date = df["observation_date"].max()
    print(f"  Date range : {start_date.date()} -> {end_date.date()}", flush=True)
    print(f"  Rows       : {len(df)}", flush=True)

    # -- Cache --
    cache = {}
    if not args.clear_cache and os.path.exists(args.cache):
        try:
            with open(args.cache, "r") as f:
                loaded = json.load(f)
            sample_key = next(iter(loaded), "")
            if "_" in sample_key and sample_key.replace("_", "").isdigit():
                cache = loaded
                print(f"  Cache      : {len(cache)} chunks already stored", flush=True)
            else:
                print(f"  Cache      : old format detected, starting fresh", flush=True)
        except Exception:
            print(f"  Cache      : corrupted, starting fresh", flush=True)
    else:
        print(f"  Cache      : starting fresh", flush=True)

    # -- Build chunk list --
    chunks = date_chunks(start_date, end_date)
    total_chunks = len(chunks)

    print(f"\n{'='*60}", flush=True)
    print(f"  Total chunks : {total_chunks}", flush=True)
    print(f"  Workers      : {args.workers}", flush=True)
    print(f"  Queries      : {len(QUERIES)}", flush=True)
    print(f"  Chunk size   : {CHUNK_DAYS} days", flush=True)
    print(f"{'='*60}\n", flush=True)

    # -- Parallel fetch --
    all_daily = {}
    chunk_infos = [(i, total_chunks, cs, ce) for i, (cs, ce) in enumerate(chunks, 1)]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_chunk, info, cache, args.cache): info
            for info in chunk_infos
        }
        for future in as_completed(futures):
            try:
                chunk_daily = future.result()
                # Merge: extend existing daily lists rather than overwrite
                for date_key, articles in chunk_daily.items():
                    if date_key in all_daily:
                        all_daily[date_key].extend(articles)
                    else:
                        all_daily[date_key] = list(articles)
            except Exception as e:
                info = futures[future]
                safe_print(f"  [ERROR] Chunk {info[0]} failed: {e}")

    total_articles = sum(len(v) for v in all_daily.values())

    # -- Build new columns --
    print(f"\nBuilding columns from {total_articles} total articles across {len(all_daily)} days ...", flush=True)

    headlines_list = []
    count_list = []

    for _, row in df.iterrows():
        dkey = row["observation_date"].strftime("%Y-%m-%d")
        entries = all_daily.get(dkey, [])

        if not entries:
            headlines_list.append(np.nan)
            count_list.append(0)
            continue

        titles = [e["title"] for e in entries[:MAX_HEADLINES_PER_DAY]]
        headlines_list.append(" ; ".join(titles))
        count_list.append(len(entries))

    df["news_headlines"] = headlines_list
    df["news_article_count"] = count_list

    # -- Summary --
    print(f"\n{'='*60}", flush=True)
    print("  Summary", flush=True)
    print(f"{'='*60}", flush=True)

    filled = df["news_headlines"].notna().sum()
    print(f"  news_headlines   : {filled}/{len(df)} days ({filled/len(df)*100:.1f}%)", flush=True)
    print(f"  news_article_count: total={int(df['news_article_count'].sum())}, "
          f"mean={df['news_article_count'].mean():.1f}/day", flush=True)

    # -- Save --
    print(f"\nSaving -> {args.output} ...", flush=True)
    df.to_excel(args.output, index=False, engine="openpyxl")
    print(f"Done! Shape: {df.shape}", flush=True)

    # -- Sample --
    sample = df[df["news_headlines"].notna()].head(5)
    if len(sample) > 0:
        print(f"\nSample rows:", flush=True)
        for _, r in sample.iterrows():
            hl = str(r["news_headlines"])
            preview = (hl[:100] + "...") if len(hl) > 100 else hl
            print(f"  {r['observation_date'].strftime('%Y-%m-%d')} | "
                  f"n={r['news_article_count']:.0f} | {preview}", flush=True)


if __name__ == "__main__":
    main()
