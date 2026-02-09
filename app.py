import atexit

from runtime import app, close_http_session, socketio

# Route and socket handler registration side-effects.
import routes  # noqa: F401
import analysis_socket  # noqa: F401
import commentary_socket  # noqa: F401
import commentary_batch_socket  # noqa: F401


@atexit.register
def shutdown_http_session():
    close_http_session()


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
