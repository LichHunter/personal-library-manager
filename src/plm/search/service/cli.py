"""CLI tool for querying the search service.

Usage:
    plm-query "What is Kubernetes?" --url http://localhost:8000 --k 5
    plm-query "machine learning" --no-rewrite --json
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click
import httpx


def _format_table(results: list[dict[str, Any]]) -> str:
    """Format results as a nice terminal table."""
    if not results:
        return "No results found."

    # Header
    lines = [
        "Rank  Score    Document    Content Preview",
        "----  -----    --------    ---------------",
    ]

    # Rows
    for i, result in enumerate(results, 1):
        rank = str(i)
        score = f"{result['score']:.4f}"
        doc_id = result['doc_id']
        content = result['content']

        # Truncate content to fit nicely (max 50 chars)
        if len(content) > 50:
            content = content[:47] + "..."
        else:
            content = content

        # Format row with fixed widths
        row = f"{rank:<5}{score:<9}{doc_id:<12}{content}"
        lines.append(row)

    return "\n".join(lines)


@click.command()
@click.argument("query")
@click.option(
    "--url",
    default="http://localhost:8000",
    help="Service URL (default: http://localhost:8000)",
)
@click.option(
    "--k",
    default=5,
    type=int,
    help="Number of results (default: 5)",
)
@click.option(
    "--no-rewrite",
    is_flag=True,
    help="Disable Haiku query rewriting",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output raw JSON instead of formatted text",
)
def main(query: str, url: str, k: int, no_rewrite: bool, json_output: bool) -> None:
    """Query the search service.

    QUERY: The search query text.
    """
    try:
        # Build request payload
        payload = {
            "query": query,
            "k": k,
            "use_rewrite": not no_rewrite,
        }

        # Make request
        response = httpx.post(
            f"{url}/query",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        # Parse response
        data = response.json()
        results = data.get("results", [])

        # Output
        if json_output:
            click.echo(json.dumps(data, indent=2))
        else:
            # Format as table
            formatted = _format_table(results)
            click.echo(formatted)

    except httpx.ConnectError:
        click.echo(
            f"Error: Could not connect to service at {url}",
            err=True,
        )
        sys.exit(1)
    except httpx.TimeoutException:
        click.echo(
            f"Error: Request to {url} timed out",
            err=True,
        )
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        click.echo(
            f"Error: Service returned {e.response.status_code}: {e.response.text}",
            err=True,
        )
        sys.exit(1)
    except json.JSONDecodeError:
        click.echo(
            "Error: Invalid JSON response from service",
            err=True,
        )
        sys.exit(1)
    except Exception as e:
        click.echo(
            f"Error: {str(e)}",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
