import os
import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = PACKAGE_ROOT / 'uploads'

class Config:
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SECRET_KEY = 'change-this-not-so-super-secure-key'
    SERVER_PORT = 8000
    UPLOAD_FOLDER = UPLOAD_FOLDER

class ProductionConfig(Config):
    DEBUG = False
    # should this be configured 'more specifically?'
    SERVER_ADDRESS: os.environ.get('SERVER_ADDRESS', '0.0.0.0')
    SERVER_PORT: os.environ.get('SERVER_PORT', '8000')


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True