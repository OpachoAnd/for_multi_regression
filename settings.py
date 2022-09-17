import redis

HOST_REDIS = 'localhost'
PORT_REDIS = 6379

redis_connect = redis.Redis(host=HOST_REDIS, port=PORT_REDIS)

