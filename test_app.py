import pytest

@pytest.fixture
def app():
    from app import app as flask_app
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_index_route(client):
    """
    Tests the index route to ensure it loads correctly.
    """
    response = client.get('/')
    assert response.status_code == 200
    assert b"Lichess Game Viewer" in response.data

def test_view_game_route_invalid_id(client):
    """
    Tests the game view route with an invalid ID format.
    """
    response = client.get('/game/invalid-game-id-!!!')
    assert response.status_code == 400

# def test_view_game_route_not_found(client):
#     """
#     Tests the game view route with a valid ID format but a game that likely doesn't exist.
#     """
#     response = client.get('/game/nonexist')
#     assert response.status_code == 404
