import os

from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from app.extensions import db
from app.factory import create_app

# from app.db import setup_db


instance_of_app = create_app()
app_dispatch = DispatcherMiddleware(instance_of_app, {"/metrics": make_wsgi_app()})
# with app.app_context():
#     setup_db()

# The following only gets executed when running a debug environment.
if __name__ == "__main__":
    # with app.app_context():
    #   db.create_all()
    instance_of_app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
