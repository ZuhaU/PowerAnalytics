# PowerAnalytics

A data pipeline + Power BI dashboard built around Pakistan's monthly electricity generation
data (published by the State Bank of Pakistan). It ingests the raw data, runs it through a
Bronze → Silver → Gold pipeline, forecasts the next 12 months of generation with a SARIMAX
model, generates a few business-relevant insights from the results, and visualizes all of it
in Power BI.

This started as a fully Azure-based project and later got rebuilt to also run locally after
my Azure subscription ended. Both versions still exist side by side — more on that below.

## The idea

SBP publishes monthly generation figures broken down by source (hydel, coal, gas, RLNG,
nuclear, wind, solar, bagasse, etc). The raw file is a wide, messy table that's not really
usable for analysis on its own. This project cleans it up, restructures it into a proper
analytics-ready format, forecasts where generation is headed, checks how good those forecasts
actually are, and surfaces a handful of insights (growth, trend, forecast bias) instead of
making someone dig through spreadsheets.

## Azure vs. local

I originally built this on Azure: Data Lake Storage Gen2 for the Bronze/Silver/Gold layers,
a Docker image pushed to Azure Container Registry, and the plan was to run it as a scheduled
Azure Container Apps Job with Power BI reading straight from the Data Lake.

My Azure subscription ran out before I got the scheduling part fully working, so I restructured
the project to also run completely locally, using local CSVs instead of the Data Lake. I didn't
rip out the Azure code — `src/azure/` still has the upload scripts, and `pipeline.py` just skips
them unless you set `ENABLE_AZURE_UPLOADS=true` (with `AZURE_STORAGE_ACCOUNT` /
`AZURE_STORAGE_KEY` set). Both Power BI files are still here too — `Power_Analytics_Local.pbix`
reads local CSVs, `Power_Analytics_Azure.pbix` was built against the Data Lake.

## Layout

```
PowerAnalytics/
├── Data/
│   ├── Raw/            original SBP export
│   ├── Silver/         cleaned + reshaped (long format)
│   ├── Gold/            monthly + per-source analytics tables, forecast, forecast evaluation
│   └── Intelligence/   insights.csv (+ executive_summary.txt if AI summary is on)
├── src/
│   ├── ingestion/       SBP scraping (Playwright) + a manual API-key check
│   ├── transformation/  Raw -> Silver -> Gold
│   ├── validation/      data quality checks on Silver
│   ├── forecasting/     SARIMAX forecast + evaluation
│   ├── intelligence/    insight generation + optional Ollama summary
│   ├── azure/           ADLS Gen2 upload scripts (off by default)
│   └── pipeline.py      runs everything above in order
├── powerbi/             both .pbix files
├── tests/               pytest suite against the pipeline outputs
├── requirements.txt
└── Dockerfile
```

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install --with-deps chromium   # only needed for live ingestion

python src/pipeline.py
```

By default `USE_LIVE_INGESTION` and `ENABLE_AZURE_UPLOADS` are both off, so it reuses the Raw
CSV already in the repo and everything stays local — no API key or Azure account needed to get
a full run.

## Forecasting

The forecast model is a SARIMAX(1,1,1)(1,1,1,12) — order and seasonal order tuned for the
monthly seasonality in generation data. Before producing the actual 12-month forecast, it's
trained on everything except the last 12 months and evaluated against that held-out period:

- MAE: **~422 GWh**
- RMSE: **~528 GWh**
- MAPE: **~4.3%**

Given that total monthly generation runs somewhere between 8,000–15,000 GWh depending on the
season, a MAPE around 4% felt like a reasonable place to stop — good enough to trust the
direction of the forecast, not so good that I'd pretend it's more precise than it is.

## The intelligence layer, and why it's not "AI-powered"

`generate_insights.py` computes things like the latest YoY growth (currently **+7.07%**), the
long-term trend across the full history (**+61.48%**), and whether the forecast is running
ahead of or behind actual generation (currently underestimating by about **2.87%**). Every
number here comes straight out of the Gold/forecast tables — there's no model involved in
computing any of it.

Early on, I actually had an LLM (Ollama running phi3:mini) generate the explanation text
directly from the raw numbers. It worked, but not reliably — every so often it would produce
repetitive or slightly malformed sentences, and once something like that ends up on a dashboard
it undermines the whole thing, since you can't tell at a glance whether a number is right or
the model just phrased it weirdly. So I replaced that step with `deterministic_explanation()` —
plain `if/else` logic that fills in pre-written sentence templates with the already-computed
numbers. Same inputs always produce the same sentence, and the text can never disagree with the
number sitting right next to it.

Ollama is still in the project, just moved: it's now an optional layer *on top of* the
deterministic output (`ENABLE_AI_SUMMARY=true`, needs `ollama pull phi3:mini`), where it only
ever reads the already-verified rows in `insights.csv` and writes a short 2-3 sentence executive
summary from them, saved to `executive_summary.txt`. It can misphrase something, but it can't
invent a number that isn't already sitting in the CSV, which is the whole point.

This is also why the Power BI report labels this section **"Automated Insights"** rather than
"AI-powered" — the actual insight computation is deterministic. If I turn the Ollama summary on,
that specific paragraph is genuinely AI-generated and I'll label it as such, but the table of
insights itself isn't, and I'd rather the labeling be accurate than sound more impressive than
it is.

## Tests

There's a small pytest suite in `tests/test_pipeline.py` that checks the shape and sanity of
each output file (no nulls where there shouldn't be, dates in order, renewable share between
0-100%, forecast bounds ordered correctly, etc.), plus direct tests for the two files that
actually have standalone functions to test (`validate_data.py`, `generate_insights.py`).

```bash
pip install pytest
python src/pipeline.py   # generate the outputs first
pytest
```

One thing writing these turned up: `validate_data()` used to throw a raw `KeyError` instead of
a clean error message when a required column was missing entirely, because the null-check ran
before confirming the columns existed. Fixed it with an early return right after the
missing-columns check, and updated the test accordingly.

## What building this actually involved

- Cleaning and reshaping a real, messy government dataset before it was usable for anything —
  wide format, inconsistent naming, mixed units, negative values that shouldn't exist.
- Structuring the pipeline into Bronze/Silver/Gold and actually feeling why that separation
  matters, rather than just following a pattern I'd read about.
- Wiring together Azure Data Lake Storage Gen2, Container Registry, and Container Apps into one
  working flow, and separately dealing with Azure Data Factory for orchestration.
- Getting Playwright/Chromium to run inside a Docker container, which is its own small nightmare
  of missing system libraries and headless-vs-headed issues.
- Losing the Azure environment partway through and having to split the pipeline logic away from
  the cloud infrastructure so the project could still run and be verified without it.
- Debugging the Python → Power BI handoff: after adding the `ai_explanation` column, Power Query
  was still configured with `Columns=10` from the previous CSV schema. Updating the query to
  recognize all 11 columns restored the missing field.
- Data quality validation actually catching something real: the pipeline flagged an invalid
  negative electricity-generation value during Silver-layer validation. That row gets identified
  and dropped before it reaches Gold, so it doesn't quietly propagate into the aggregations and
  the forecast.
- Actually evaluating the forecast against real held-out data (MAE/RMSE/MAPE) instead of just
  generating numbers and assuming they were fine.
- Trying an LLM-generated explanation layer, finding it unreliable enough that I didn't trust it
  on a dashboard, and replacing it with deterministic logic instead — then bringing the LLM back
  later as a narrower, safer add-on rather than the source of truth.
- Making the Power BI report pull from the generated CSVs directly instead of hardcoded figures,
  so it actually updates when new data comes through.

Basically, this ended up being less about any one piece (the forecasting, the cloud setup, the
BI layer) and more about getting all of them talking to each other correctly — Python, Docker,
Azure, CSV schemas, Power Query, and Power BI — and debugging whichever one broke that day.

## Data source

State Bank of Pakistan Easydata API, series group `TS_GP_RLS_ELECGEN_M`, monthly electricity
generation by source since July 2012. An API key is required for live ingestion; the Raw CSV
already in this repo is enough to run everything else without one.
