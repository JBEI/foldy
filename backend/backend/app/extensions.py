from flask_admin import Admin
from flask_migrate import Migrate
from flask_rq2 import RQ
from flask_sqlalchemy import SQLAlchemy
from flask_compress import Compress

from sqlalchemy.pool import NullPool

admin = Admin()
db = SQLAlchemy(engine_options={'poolclass': NullPool})
migrate = Migrate()
rq = RQ()
compress = Compress()
