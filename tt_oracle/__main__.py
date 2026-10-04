"""
Headless usage (this is what the GitHub Actions workflows call):
    python -m tt_oracle backfill --pages 80               # pull the whole archive (minutes; run once)
    python -m tt_oracle update                            # recent + new pages
    python -m tt_oracle odds                              # DraftKings snapshot
    python -m tt_oracle train
    python -m tt_oracle predict
    python -m tt_oracle backtest --days 60
    python -m tt_oracle cycle                             # update -> odds -> grade -> learn -> retrain -> predict
"""
import argparse
import json
import logging

import pandas as pd

from . import config, model as M, pipeline, scraper, store


def _p(f, m):
    print(f"[{f:5.0%}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(prog="tt_oracle")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill"); b.add_argument("--pages", type=int, default=400); b.add_argument("--keep-raw", type=int, default=3)
    u = sub.add_parser("update"); u.add_argument("--pages", type=int, default=2)
    sub.add_parser("odds")
    sub.add_parser("train")
    pr = sub.add_parser("predict"); pr.add_argument("--no-log", action="store_true")
    bt = sub.add_parser("backtest"); bt.add_argument("--days", type=int, default=60); bt.add_argument("--step", type=int, default=7)
    bt.add_argument("--static", action="store_true")
    c = sub.add_parser("cycle"); c.add_argument("--no-scrape", action="store_true"); c.add_argument("--no-odds", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.ensure_dirs()

    if a.cmd == "backfill":
        print(json.dumps(scraper.backfill(pages=a.pages, progress=_p, keep_raw=a.keep_raw), indent=2))
    elif a.cmd == "update":
        print(json.dumps(scraper.update(pages=a.pages, progress=_p), indent=2))
    elif a.cmd == "odds":
        from .dk import snapshot_odds
        print(json.dumps(snapshot_odds(), indent=2))
    elif a.cmd == "train":
        print(pipeline.retrain().meta)
    elif a.cmd == "predict":
        P = pipeline.predict_upcoming(log_to_ledger=not a.no_log)
        with pd.option_context("display.width", 200, "display.max_rows", 300):
            print(P[["start_time", "player_a", "player_b", "p_a", "pick", "confidence", "market_p_a"]].round(3).to_string(index=False)
                  if not P.empty else "no upcoming matches")
    elif a.cmd == "backtest":
        feat = pipeline.features_now()
        end = feat[feat["y"].notna()]["start_time"].max()
        bt_df = pipeline.walk_forward(feat, end - pd.Timedelta(days=a.days), end, step_days=a.step,
                                      adaptive=not a.static, progress=_p)
        store.write(bt_df, config.BACKTEST_CSV)
        print(json.dumps(pipeline.summarize(bt_df), indent=2, default=str))
    elif a.cmd == "cycle":
        print(json.dumps(pipeline.run_cycle(_p, scrape=not a.no_scrape, odds=not a.no_odds), indent=2, default=str))


if __name__ == "__main__":
    main()
