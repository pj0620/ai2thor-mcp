---
name: restart-ai2thor-mcp
description: Restart the local ai2thor MCP server — stop the process listening on port 8000 (and any orphaned Unity sim), relaunch it with the same arguments, and health-check the endpoint. Use when the server is stale, hung, or running old code after edits.
---

Restart the ai2thor MCP server that runs locally on port 8000.

1. **Find the running server** (it may not be running — then skip to step 4):
   - PID: `lsof -nP -ti tcp:8000 -sTCP:LISTEN`
   - Args to preserve: `ps -o command= -p <PID>` — the command looks like
     `.../python3 python/ai2thor_mcp/main.py --http [--scene <Scene>]`; keep any
     `--scene` (or other) flags for the relaunch.

2. **Stop it.** If this session started it as a background task, use TaskStop on
   that task id. Otherwise `kill <PID>`, wait ~2 s, and only if
   `lsof -nP -ti tcp:8000 -sTCP:LISTEN` still returns it, `kill -9 <PID>`.

3. **Kill orphaned Unity children.** The AI2-THOR Unity process can outlive a
   killed parent. If `pgrep -f thor-OSXIntel64` finds anything,
   `pkill -f thor-OSXIntel64`.

4. **Relaunch** from the repo root as a background task (run_in_background):
   `uv run python/ai2thor_mcp/main.py --http` plus any preserved flags.

5. **Health-check.** The HTTP endpoint is up within a few seconds (Unity itself
   boots lazily on the first tool call, so no window yet is normal):

   ```
   curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/mcp \
     -H "Content-Type: application/json" \
     -H "Accept: application/json, text/event-stream" \
     -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"health","version":"0"}}}'
   ```

   Expect `200`. Retry a few times over ~15 s; on failure, read the background
   task's output file for the actual error before reporting.

6. **Report** concisely: what was stopped (PID + args), the new background task
   id, the health-check result, and the scene in use. Remind the user that
   Claude Desktop reaches this server through the mcp-remote bridge and
   reconnects on next use — but if the tool list changed since Desktop last
   connected, the app needs a full restart (Cmd+Q) to see the new tools.
