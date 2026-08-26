import redis from 'redis';

const client = redis.createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379',
  password: process.env.REDIS_PASSWORD,
});

client.on('error', (err: Error) => {
  console.error('Redis Client Error', err);
});

export default client;
