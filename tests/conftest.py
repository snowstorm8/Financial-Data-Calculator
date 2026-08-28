import pytest

from app import app


@pytest.fixture
def client():
    """Flask test client shared by every test module — makes requests
    in-process against `app` without starting a real HTTP server."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
