import os

from flask_admin import Admin, AdminIndexView
from flask_compress import Compress
from flask_migrate import Migrate

# from flask_rq2 import RQ
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.pool import NullPool

admin = Admin(name="Admin View")
db = SQLAlchemy(engine_options={"poolclass": NullPool})
migrate = Migrate()
# rq = RQ()
compress = Compress()
