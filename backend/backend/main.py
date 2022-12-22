import os

from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app

from app.app import create_app
from app.extensions import db
# from app.db import setup_db


app = create_app()
app_dispatch = DispatcherMiddleware(app, {
    '/metrics': make_wsgi_app()
})
# with app.app_context():
#     setup_db()

# The following only gets executed when running a debug environment.
if __name__ == "__main__":
  # with app.app_context():
  #   db.create_all()
  app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
