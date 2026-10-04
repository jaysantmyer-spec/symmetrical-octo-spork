"""Synthetic end-to-end test: generates realistic TT Elite-style tournaments with hidden player strengths,
then runs features -> train -> backtest -> predict -> ledger -> grade -> relearn -> pricing in a temp dir."""
import os, sys, pathlib, tempfile, random
ROOT = pathlib.Path(__file__).resolve().parent.parent
tmp = tempfile.mkdtemp()
os.environ["TT_ORACLE_ROOT"] = tmp
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
from tt_oracle import config, store, pipeline, model as M, dk
from tt_oracle.features import build_features

rng = random.Random(7)
players = [f"Player {chr(65 + i)}{chr(97 + j)}" for i in range(6) for j in range(8)]  # 48 players
strength = {p: rng.gauss(0, 1) for p in players}
rows, pid = [], 100
start = pd.Timestamp("2026-06-01")
for day in range(90):
    date = start + pd.Timedelta(days=day)
    for si, sess in enumerate(["morning", "afternoon", "evening", "night"]):
        for venue in ("HSC", "OSP"):
            pid += 1
            six = rng.sample(players, 6)
            pairs = [(0, 5), (1, 4), (2, 3), (0, 1), (3, 5), (2, 4), (1, 5), (0, 2), (3, 4), (1, 2), (0, 3), (4, 5), (1, 3), (2, 5), (0, 4)]
            t0 = date + pd.Timedelta(hours=7 + si * 4)
            for order, (i, j) in enumerate(pairs, 1):
                a, b = six[i], six[j]
                future = (day == 89 and si >= 2)
                pa = 1 / (1 + np.exp(-(strength[a] - strength[b]) * 1.2))
                won = rng.random() < pa
                sa, sb = (3, rng.choice([0, 1, 2])) if won else (rng.choice([0, 1, 2]), 3)
                rows.append({"post_id": pid, "date": date, "session": sess, "venue": venue, "order": order,
                             "start_time": t0 + pd.Timedelta(minutes=25 * (order - 1)) + (pd.Timedelta(days=0) if not future else pd.Timedelta(hours=40)),
                             "player_a": a, "player_b": b, "sets_a": None if future else sa, "sets_b": None if future else sb,
                             "match_id": f"{pid}-{order}", "finished": not future, "winner": None if future else ("a" if won else "b")})
m = pd.DataFrame(rows)
# make 'future' rows actually in the future relative to now
now = pd.Timestamp.now()
shift = now - m.loc[~m.finished, "start_time"].min() + pd.Timedelta(hours=2)
m["start_time"] = m["start_time"] + shift
m["date"] = m["start_time"].dt.normalize()
config.ensure_dirs(); store.write(m, config.MATCHES_CSV)
print("matches", len(m), "finished", int(m.finished.sum()))

feat = build_features(m)
assert len(feat) == len(m) and feat["y"].notna().sum() == m.finished.sum()
assert (feat["A_career_n"] >= 0).all()
pred = M.train(feat, ["logreg", "hgb"])
print("trained on", pred.meta["n_train"])
bt = pipeline.walk_forward(feat, feat.start_time.min() + pd.Timedelta(days=45), step_days=7, learners=["logreg", "hgb"])
s = pipeline.summarize(bt)
print(f"backtest: n={s['matches']} acc={s['accuracy']:.3f} ll={s['log_loss']:.3f} elo_acc={s['elo_accuracy']:.3f}")
assert s["accuracy"] > 0.6, "model should beat chance clearly on synthetic data"
pred.save()
# odds: pasted lines for two upcoming matches
up = pipeline.upcoming(feat, horizon_days=5)
assert len(up) > 0
txt = f"{up.iloc[0].player_a} -150 / {up.iloc[0].player_b} +120\n{up.iloc[1].player_b} +200 / {up.iloc[1].player_a} -250"
odds = dk.parse_pasted(txt); assert len(odds) == 2, odds
dk.snapshot_odds(odds)
P = pipeline.predict_upcoming(pred)
assert P["market_p_a"].notna().sum() == 2, P[["player_a", "player_b", "dk_odds_a", "market_p_a"]].head()
pr = dk.price(P.head(3), 1000)
print(pr[["player", "fair", "market", "dk", "edge", "stake", "bet"]].to_string(index=False))
led = store.read(config.LEDGER_CSV); print("ledger rows", len(led))
# simulate results arriving, then grade + relearn
m2 = store.read(config.MATCHES_CSV); fut = ~m2.finished.astype(bool)
m2.loc[fut, "sets_a"], m2.loc[fut, "sets_b"], m2.loc[fut, "finished"], m2.loc[fut, "winner"] = 3, 1, True, "a"
store.write(m2, config.MATCHES_CSV)
g = pipeline.grade_ledger(); print("graded", g)
assert g["graded_now"] == len(led)
print("relearn", pipeline.relearn(min_graded=10)["updated"])
print("OK ->", tmp)
