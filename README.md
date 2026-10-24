# MicroShield

**Understand the CPU. Explore the architecture. Investigate the risks.**

MicroShield is an educational lab for exploring simplified CPU pipeline, branch prediction, cache, and speculative execution models. All values are simulated by documented algorithms; this application does not benchmark physical hardware or reproduce a commercial processor.

## Architecture

```text
React + TypeScript + Vite (frontend)
  ├── Pipeline, branch, and cache educational engines
  ├── Responsive lab UI and browser-local experiment history
  └── Optional REST calls to FastAPI (backend)
         ├── Pydantic request validation
         └── Pure Python simulation cores (backend/app/core)
```

```text
frontend/src/App.tsx                 App shell and lab views
frontend/src/lib/engines/            TypeScript simulation engines
backend/app/main.py                  REST endpoints and validation
backend/app/core/                     Independent Python simulation cores
backend/tests/                        Pytest algorithm tests
frontend/src/lib/engines/*.test.ts    Vitest algorithm tests
fixtures/                              Shared hand-checked input/output cases
```

The standalone UI runs pipeline, branch, and cache calculations in TypeScript for immediate trace rendering. Python endpoints provide the same simulation services for integrations and API exploration. Local history is stored in browser local storage and is not uploaded.

## Run locally (macOS)

Prerequisites: Node.js 20+ and Python 3.10+.

```sh
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

In another terminal:

```sh
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. FastAPI interactive documentation is at <http://localhost:8000/docs>; health check is `GET /api/health`.

## Verify

```sh
cd frontend && npm test && npm run build
cd backend && python -m pytest
```

## API overview

- `GET /api/health`
- `POST /api/pipeline/simulate` — `program`, forwarding, branch policy, memory latency.
- `POST /api/branch-predictor/simulate` — same ordered T/N outcomes for one selected predictor.
- `POST /api/branch-predictor/compare` — run selected predictors on the same sequence.
- `POST /api/cache/simulate` — R/W address sequence, geometry, replacement, hit/miss latency.
- `POST /api/security/cache-timing` — synthetic timing sequences.
- `POST /api/security/speculation` — synthetic speculation timeline and optional barrier.

Example:

```sh
curl -X POST http://localhost:8000/api/branch-predictor/simulate \
  -H 'Content-Type: application/json' \
  -d '{"outcomes":["T","N","T","T"],"predictor":"two_bit"}'
```

`CORS_ALLOW_ORIGINS` accepts comma-separated origins. `CORS_ALLOW_ALL=true` is a development-only escape hatch and should not be enabled in production. Frontend builds with `npm run build`; serve `frontend/dist` from a static host and run the API separately with Uvicorn on a Python host. Configure `CORS_ALLOW_ORIGINS` for the deployed frontend origin. Hosting free tiers and limits can change; check current provider terms when deploying.

## Model assumptions and limitations

- The pipeline is single-issue and in-order with five stages. It models register hazards, forwarding, branches, and blocking memory under the assumptions documented in `backend/app/core/pipeline.py`.
- Branch predictors are textbook educational designs with online updates. They do not represent the size or hybrid behavior of commercial CPU predictors.
- Cache addresses are byte addresses. The cache cold-starts, allocates on writes, and models LRU/FIFO replacement; it does not model dirty writeback traffic or physical cache timing.
- Security examples use synthetic latencies and harmless abstract cache lines. They are not vulnerability demonstrations on the host machine, and the toy mitigation is not a complete defense.
- The interface currently runs whole pipeline simulations, not animated step/play controls. The cache trace is rendered after replay; cache-step interaction is a future improvement.
- Architecture comparison currently focuses on forwarding in the pipeline model. Joint pipeline/branch/cache workload coupling and estimated wall-clock execution time are future work.
- The UI's local history supports save, rename, delete, JSON import/export, and copy. Duplicate, side-by-side history comparison, CSV export, and strict versioned schema migration are future improvements.
- API calls are exposed in FastAPI and backend tests cover the pure Python engines; the UI presently calculates locally.

## Future improvements

Add cycle controls and live pipeline animation, stepping through cache replacement state, full architecture configuration comparison using shared workloads, experiment duplication and comparison, CSV exports, stronger versioned import validation, and visual cache/predictor charts.
