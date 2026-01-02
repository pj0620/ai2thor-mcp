import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, cast
import os

import pytest
from fastmcp import Client
from ai2thor_mcp.main import mcp

@pytest.mark.asyncio
async def test_mcp_server_options() -> None:
    async with Client(mcp) as client:
        for _ in range(10):
            _ = await client.call_tool(
                "do_move",
                {"action": "MoveAhead", "moveMagnitude": 1},
            )
            await asyncio.sleep(5)
            


# @pytest.mark.asyncio
# async def test_get_revenue_exercising_at() -> None:
#     option = Option(
#         cfi="OCASPS",
#         contract_type="call",
#         exercise_style="american",
#         expiration_date="2024-02-16",
#         primary_exchange="BATO",
#         shares_per_contract=100,
#         strike_price=120.0,
#         ticker="O:AAPL240216C00120000",
#         underlying_ticker="AAPL",
#     )

#     exercise_at = datetime(2024, 2, 9, 20, 0, tzinfo=timezone.utc)

#     ticker = yf.Ticker(option.underlying_ticker)
#     history = ticker.history(
#         start=exercise_at.date().isoformat(),
#         end=(exercise_at.date() + timedelta(days=2)).isoformat(),
#         interval="1d",
#         auto_adjust=False,
#     )

#     valid_rows = history[history.index <= exercise_at]
#     if valid_rows.empty:
#         valid_rows = history

#     market_price = float(valid_rows["Close"].iloc[-1])

#     async with Client(mcp) as client:
#         revenue_resp = await client.call_tool(
#             "get_revenue_exercising_at",
#             {
#                 "option": option.model_dump(),
#                 "exercise_at": exercise_at.isoformat(),
#             },
#         )

#     revenue = cast(float, revenue_resp.data)
#     expected_revenue = (market_price - option.strike_price) * option.shares_per_contract

#     assert revenue == pytest.approx(expected_revenue)