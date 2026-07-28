# agent-worker

GEO audit microservice for the Emergence Science platform.

Multi-LLM probe network + brand profiling pipeline.

## Architecture

Part of the Surprisal Protocol three-pod architecture:

```
orchestrator ──(httpx)──▶ agent-worker
  (FastAPI)               (FastAPI, sleep-on-idle)
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check with probe registry status |
| `POST` | `/jobs/geo-audit` | Launch multi-LLM GEO probe → returns `job_id` |
| `POST` | `/jobs/geo-interview` | Process one interview turn → returns next question |
| `GET` | `/jobs/{job_id}/status` | Poll job status with phase/progress/log_entries |
| `GET` | `/jobs/{job_id}/result` | Get full job result (when DONE) |

### Probe Network

| Tier | Providers | Method |
|------|-----------|--------|
| A (V1) | DeepSeek, Kimi (Moonshot), Perplexity | OpenAI-compatible API |
| B (V2) | 豆包, 元宝, 文心一言 | Playwright headless browser |

## Development

```bash
pip install -r requirements.txt
cp .env.example .env  # edit with your API keys
uvicorn main:app --reload --port 8002
```

## Deployment

Deployed on Railway as a sleep-on-idle pod.

Environment variables:

| Variable | Description |
|----------|-------------|
| `PORT` | Railway-assigned port |
| `DATABASE_URL` | PostgreSQL connection string |
| `AGENT_WORKER_TOKEN` | Shared secret for orchestrator auth |
| `DEEPSEEK_API_KEY` | Tier A probe |
| `MOONSHOT_API_KEY` | Tier A probe |
| `PERPLEXITY_API_KEY` | Tier A probe |

## Related

- [Architecture Design Doc](https://github.com/emergencescience/emergence-meta/blob/main/internal/design/2026-07-ARCH-lightweight-orchestrator-pipeline.md)
- [UX Design Spec](https://github.com/emergencescience/emergence-meta/blob/main/internal/design/2026-07-UX-emergence-chat-control-plane.md)
