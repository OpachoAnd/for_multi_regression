import pickle

import redis

HOST_REDIS = 'localhost'
PORT_REDIS = 6379

REDIS_CONNECTION = redis.Redis(host=HOST_REDIS, port=PORT_REDIS)
