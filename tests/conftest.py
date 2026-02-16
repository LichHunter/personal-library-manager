"""Shared test fixtures."""
import pytest


@pytest.fixture
def sample_text():
    return """
    Using React.Component with TypeScript for a NextJS app.
    The CrashLoopBackOff error appears when pods restart.
    Check kubectl logs and docker-compose.yml for issues.
    """


@pytest.fixture
def expected_terms():
    return [
        "React.Component",
        "TypeScript",
        "NextJS",
        "CrashLoopBackOff",
        "kubectl",
        "docker-compose.yml",
    ]
