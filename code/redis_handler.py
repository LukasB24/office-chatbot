import redis

class RedisHandler:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)

    def set_data(self, key, value):
        self.redis_client.set(key, value)

    def get_data(self, key):
        return self.redis_client.get(key)

    def delete_data(self, key):
        self.redis_client.delete(key)
