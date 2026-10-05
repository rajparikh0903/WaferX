from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

API_BASE = os.getenv(
    "YIELDTWIN_API_URL",
    "https://waferx.onrender.com",
).rstrip("/")

mcp = FastMCP("YieldTwin")


async def api_request(
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.request(
            method,
            f"{API_BASE}{path}",
            json=json,
            params=params,
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def analyze_sample(sample_id: int, top_k: int = 5) -> dict[str, Any]:
    """Run full YieldTwin process analysis for a SECOM sample."""
    return await api_request(
        "POST",
        "/analyze",
        json={
            "sample_id": sample_id,
            "top_k": top_k,
        },
    )


@mcp.tool()
async def get_anomaly(sample_id: int) -> dict[str, Any]:
    """Return anomaly detection results for a SECOM sample."""
    return await api_request(
        "POST",
        "/anomaly",
        json={"sample_id": sample_id},
    )


@mcp.tool()
async def get_root_causes(
    sample_id: int,
    top_k: int = 5,
) -> dict[str, Any]:
    """Return ranked root-cause candidates for a SECOM sample."""
    return await api_request(
        "POST",
        "/root-cause",
        json={
            "sample_id": sample_id,
            "top_k": top_k,
        },
    )


@mcp.tool()
async def run_what_if(
    sample_id: int,
    feature: str,
    values: list[float],
) -> dict[str, Any]:
    """Run YieldTwin counterfactual analysis for a sensor."""
    return await api_request(
        "POST",
        "/what-if",
        json={
            "sample_id": sample_id,
            "feature": feature,
            "values": values,
        },
    )


@mcp.tool()
async def get_feature_importance(top_n: int = 20) -> dict[str, Any]:
    """Return the most important process features from YieldTwin."""
    return await api_request(
        "GET",
        "/feature-importance",
        params={"top_n": top_n},
    )


if __name__ == "__main__":
    mcp.run()