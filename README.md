# TT Elite Oracle

Predicts TT Elite Series (Poland) table tennis matches and prices them against DraftKings.
Sister project of UFC Oracle; same ensemble design, same honesty rules.

**Data**: every tournament page on tt-series.com (6-player round robins, ~15 matches each, since 2022),
collected by a GitHub Actions workflow — no scraping from the app or your computer.
**Model**: Elo + form + head-to-head + fatigue + session features; logistic regression, histogram
boosting, XGBoost, LightGBM and CatBoost blended with Hedge weights and a temperature calibration;
predictions logged before each match, graded after, and the blend re-learned from the graded ledger.
**Pricing desk**: fair odds vs DraftKings moneylines, edge, EV, fractional-Kelly stakes, with a paste box
for lines when the automatic feed has nothing.
**Track record**: the graded ledger (real, pre-match predictions) and a weekly walk-forward backtest.

See `DEPLOY.md` for setup. Command line: `python -m tt_oracle --help`.

This is a statistical tool, not betting advice.
