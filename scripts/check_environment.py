"""Verify that the local development environment can support the build.

Run this before starting any phase. It reports what is present, what is missing,
and which phases each missing item blocks. It never installs or changes anything.

    python scripts/check_environment.py
"""

from __future__ import annotations

import importlib.metadata as md
import os
import shutil
import socket
import sys
from collections.abc import Callable
from dataclasses import dataclass

MIN_PYTHON = (3, 11)

# Package -> phase that first needs it.
REQUIRED_PACKAGES: dict[str, str] = {
    "fastapi": "10",
    "pydantic": "2",
    "sqlalchemy": "2",
    "alembic": "2",
    "psycopg2": "2",
    "redis": "3",
    "numpy": "2",
    "pandas": "2",
    "scipy": "5",
    "scikit-learn": "4",
    "implicit": "7",
    "lightgbm": "9",
    "xgboost": "9",
    "sentence-transformers": "6",
    "pgvector": "6",
    "mlflow": "13",
    "prometheus-client": "16",
    "pytest": "18",
}

SERVICES: list[tuple[str, str, int, str]] = [
    ("PostgreSQL", "POSTGRES_HOST", 5432, "2"),
    ("Redis", "REDIS_HOST", 6379, "3"),
    ("MLflow", "MLFLOW_HOST", 5000, "13"),
]

TOOLS: dict[str, str] = {"git": "1", "node": "12", "npm": "12", "docker": "17"}

OK, WARN, FAIL = "  OK  ", " WARN ", " FAIL "


@dataclass
class Result:
    status: str
    label: str
    detail: str = ""

    def render(self) -> str:
        line = f"[{self.status}] {self.label}"
        return f"{line}\n         {self.detail}" if self.detail else line


def check_python() -> Result:
    v = sys.version_info
    current = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < MIN_PYTHON:
        need = ".".join(map(str, MIN_PYTHON))
        return Result(FAIL, f"Python {current}", f"requires >= {need}")
    return Result(OK, f"Python {current}", sys.executable)


def check_packages() -> list[Result]:
    results: list[Result] = []
    for name, phase in REQUIRED_PACKAGES.items():
        try:
            results.append(Result(OK, f"{name} {md.version(name)}"))
        except md.PackageNotFoundError:
            results.append(
                Result(WARN, name, f"not installed - needed from phase {phase}")
            )
    return results


def check_tools() -> list[Result]:
    results: list[Result] = []
    for tool, phase in TOOLS.items():
        path = shutil.which(tool)
        if path:
            results.append(Result(OK, tool, path))
        else:
            results.append(
                Result(WARN, tool, f"not on PATH - needed from phase {phase}")
            )
    return results


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_services() -> list[Result]:
    results: list[Result] = []
    for label, env_key, default_port, phase in SERVICES:
        host = os.getenv(env_key, "localhost")
        port = int(os.getenv(env_key.replace("_HOST", "_PORT"), default_port))
        if port_open(host, port):
            results.append(Result(OK, label, f"{host}:{port} reachable"))
        else:
            results.append(
                Result(
                    WARN,
                    label,
                    f"{host}:{port} unreachable - needed from phase {phase}",
                )
            )
    return results


def section(title: str, checks: Callable[[], list[Result]]) -> list[Result]:
    print(f"\n{title}\n{'-' * len(title)}")
    results = checks()
    for result in results:
        print(result.render())
    return results


def main() -> int:
    print("Environment check - E-commerce Recommendation Platform")

    print("\nRuntime\n-------")
    python_result = check_python()
    print(python_result.render())

    results = [python_result]
    results += section("Python packages", check_packages)
    results += section("Command-line tools", check_tools)
    results += section("Services", check_services)

    failures = [r for r in results if r.status == FAIL]
    warnings = [r for r in results if r.status == WARN]

    print(f"\nSummary: {len(results) - len(failures) - len(warnings)} ok, "
          f"{len(warnings)} warning, {len(failures)} failing")
    if failures:
        print("Blocking problems must be resolved before continuing.")
        return 1
    if warnings:
        print("Warnings are fine until the phase that needs them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
