# RoboAlgo High-Speed Scanner Architecture (20,000 symbols < 10s)

## Pipeline
`data_ingestion -> feature_generation -> pattern_detection -> liquidity_engine -> signal_scoring`

## Core Stack
- Redis: event bus, work queue, hot cache, fanout pub/sub
- TimescaleDB: OHLCV/features/scores storage with hypertables and compression
- Worker Pool: CPU-bound multiprocessing workers pinned by core
- Async Pipelines: non-blocking orchestration + batch dispatch

## Topology
- Ingestion Service (async I/O): streams OHLCV deltas to Redis Streams
- Feature Service (vectorized): computes features in columnar batches
- Pattern Service (multiprocessing): scans symbols per-core per-batch
- Liquidity Service (multiprocessing): cluster + trap scoring
- Signal Service (vectorized): weighted scoring + rank + publish
- API Gateway: reads latest snapshots from Redis cache, falls back to TimescaleDB

## Data Model
- Timescale hypertables partitioned by `time` + `symbol_hash`
- Latest snapshot materialized per symbol in Redis hash
- Redis key examples:
  - `scan:batch:{id}:symbols`
  - `scan:stage:{stage}:done`
  - `signal:latest:{symbol}`

## Execution Plan (per scan cycle)
1. Build active universe list (20k symbols) from Redis set.
2. Split into fixed-size batches (e.g., 256 symbols).
3. Push batch jobs to Redis stream per stage.
4. Worker pools consume jobs in parallel.
5. Each stage writes compact outputs to Redis + TimescaleDB async writer.
6. Downstream stage triggers immediately after upstream batch completion.
7. Final signal ranking aggregates all batch outputs.

## Parallelism Strategy
- 1 orchestrator process (async)
- N ingestion coroutines (I/O bound)
- `cpu_count * 2` process workers for pattern/liquidity (CPU bound)
- Batch-level parallelism + stage overlap (pipeline parallelism)
- Zero per-symbol DB round-trips in hot path; bulk reads/writes only

## Performance Constraints to Hit <10s
- Use rolling in-memory windows from Redis (avoid full historical reads each run)
- Use vectorized NumPy/pandas ops in workers
- Keep per-symbol payload tiny (OHLCV window only, e.g., 120-260 bars)
- Bulk write with COPY/execute_values to TimescaleDB
- Cache latest signal payloads in Redis for API reads
- Avoid Python object churn: use arrays and typed dict schemas

## Scheduling
- Continuous micro-batch loop every 250-500ms
- Backpressure via Redis consumer-group lag thresholds
- Drop/merge stale batches when a newer batch for same symbol arrives

## Reliability
- Consumer groups with ack/retry semantics
- Dead-letter stream per stage
- Stage-level idempotency keys: `(scan_id, stage, symbol)`
- Health metrics: throughput, lag, p95 stage latency, error rate

## Pseudocode
```python
async def scan_cycle(symbols):
    batches = chunk(symbols, 256)
    await publish_stage('data_ingestion', batches)

    for stage in ['data_ingestion','feature_generation','pattern_detection',
                  'liquidity_engine','signal_scoring']:
        await run_stage_workers(stage)  # async consumers + process pool
        await wait_stage_complete(stage)

    await publish_latest_signals_to_cache()
```

## Expected Latency Budget (target)
- ingestion: 1.0s
- features: 1.5s
- patterns: 2.5s
- liquidity: 2.0s
- scoring + publish: 1.0s
- overhead: 1.0s
- total: ~9.0s
