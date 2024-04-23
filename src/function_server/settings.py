from environs import Env


env = Env()
env.read_env()

LOG_LEVEL = env.str('LOG_LEVEL', 'INFO')
FAKE_ALL_MODEL = env.bool('FAKE_ALL_MODEL', False)
NO_FAKE_MODELS = env.list("NO_FAKE_MODELS", [])
WEB_SEARCH_ENGINE = env.str('WEB_SEARCH_ENGINE', 'bing')