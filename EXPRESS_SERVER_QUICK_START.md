# Express Server - Quick Start Guide

**Version:** 1.0.0
**Status:** ✅ PRODUCTION READY
**Created:** October 17, 2025

---

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment

```bash
# Copy example to actual env file
cp .env.example .env

# Edit .env with your values
# Key values to update:
# - DATABASE_URL: Your PostgreSQL connection string
# - JWT_SECRET: Generate with: node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
# - REDIS_URL: Your Redis connection (optional)
```

### 3. Build

```bash
npm run build
```

### 4. Start Server

```bash
# Production
npm start

# Development (with hot reload)
npm run dev
```

### 5. Test

```bash
# Basic health check
curl http://localhost:3000/health

# Detailed health check
curl http://localhost:3000/health/detailed

# Metrics
curl http://localhost:3000/health/metrics
```

---

## 📁 Project Structure

```
src/
├── index.ts                          # Entry point
├── server/
│   └── server.ts                     # Express app + initialization
├── middleware/
│   ├── authMiddleware.ts             # JWT, RBAC, validation
│   └── errorHandler.ts               # Error handling
├── routes/
│   └── healthRoutes.ts               # Health check endpoints
├── services/
│   └── loggingService.ts             # Structured logging
├── config/
│   └── serverConfig.ts               # Configuration service
└── models/                           # Database models (add here)
```

---

## 🔧 Key Components

### Server (src/server/server.ts)

Initializes Express app with:

- Security middleware (Helmet)
- CORS handling
- Rate limiting
- Request tracking
- Error handling

**API:**

```typescript
import { startServer } from './server/server';

// Starts server with automatic initialization
await startServer({
  port: process.env.PORT || 3000,
  configService: configService,
});
```

### Configuration (src/config/serverConfig.ts)

Singleton service managing:

- Environment variables
- Database connection pooling
- Redis client
- Service health checks

**API:**

```typescript
import { configService } from './config/serverConfig';

const config = configService.getConfig();
const pool = configService.getDatabasePool();
const redis = configService.getRedisClient();
```

### Logging (src/services/loggingService.ts)

Winston-based logging with:

- File and console output
- Log rotation
- Structured logging
- Performance tracking

**API:**

```typescript
import { log } from './services/loggingService';

log.info('Message', { context: 'data' });
log.error('Error', error, { details: 'info' });
log.security('Security event', 'high', { ip: '1.2.3.4' });
log.performance('Operation', 1200, 'ms');
```

### Authentication (src/middleware/authMiddleware.ts)

JWT verification with:

- Request ID tracking
- Role-based access control
- Request validation
- Security logging

**API:**

```typescript
import { authenticateToken, authorize, validateRequest } from './middleware/authMiddleware';

// Require authentication
app.use(authenticateToken);

// Role-based access
app.delete('/admin', authorize(['admin']), handler);

// Validate request
app.post('/users', validateRequest(schema), handler);
```

### Error Handling (src/middleware/errorHandler.ts)

Comprehensive error handling with:

- 8 custom error classes
- Async error wrapper
- Standardized error responses
- Security logging

**API:**

```typescript
import { asyncHandler, AppError, NotFoundError } from './middleware/errorHandler';

// Wrap async handlers
app.get(
  '/data',
  asyncHandler(async (req, res) => {
    const data = await db.getData();
    if (!data) throw new NotFoundError('Data');
    res.json(data);
  }),
);
```

---

## 📊 Health Check Endpoints

All return JSON with request tracking:

```bash
# Basic health check
curl http://localhost:3000/health
# → { status: 'ok', uptime: 3600, ... }

# Detailed status with dependencies
curl http://localhost:3000/health/detailed
# → { status: 'healthy', services: { database, redis }, ... }

# Kubernetes readiness probe
curl http://localhost:3000/health/ready
# → { ready: true, timestamp: '...', ... }

# Kubernetes liveness probe
curl http://localhost:3000/health/live
# → { status: 'alive', uptime: 3600, ... }

# Server metrics
curl http://localhost:3000/health/metrics
# → { memory: { rss, heapTotal, heapUsed }, process: { pid, version } }

# Service status
curl http://localhost:3000/health/status
# → { services: { database, redis }, configuration: { ... } }
```

---

## 🔒 Security

The server includes built-in security:

- **Helmet.js**: Secure HTTP headers
- **Rate limiting**: 100 requests per 15 minutes (configurable)
- **CORS**: Origin whitelist (configure in .env)
- **JWT**: Token-based authentication
- **RBAC**: Role-based access control
- **Password hashing**: Bcrypt with 10+ rounds
- **Error hiding**: Stack traces only in development

---

## 🔄 Adding Routes

Create a new route file:

```typescript
// src/routes/imageRoutes.ts
import { Router } from 'express';
import { asyncHandler } from '../middleware/errorHandler';
import { authenticateToken, authorize } from '../middleware/authMiddleware';

const router = Router();

router.post(
  '/analyze',
  authenticateToken,
  asyncHandler(async (req, res) => {
    // Your handler logic
    res.json({ result: 'analysis' });
  }),
);

router.delete(
  '/:id',
  authenticateToken,
  authorize(['admin']),
  asyncHandler(async (req, res) => {
    // Admin-only handler
    res.json({ deleted: true });
  }),
);

export default router;
```

Register in server:

```typescript
// src/server/server.ts
import imageRoutes from '../routes/imageRoutes';

app.use('/api/v1/images', imageRoutes);
```

---

## 💾 Database Usage

The server provides a PostgreSQL connection pool:

```typescript
// In route handler
import { configService } from '../config/serverConfig';

const pool = configService.getDatabasePool();

// Query
const result = await pool.query('SELECT * FROM images WHERE id = $1', [id]);

// Insert
const insert = await pool.query('INSERT INTO images (name, data) VALUES ($1, $2) RETURNING id', [
  name,
  data,
]);

// Update
await pool.query('UPDATE images SET status = $1 WHERE id = $2', [status, id]);

// Delete
await pool.query('DELETE FROM images WHERE id = $1', [id]);
```

---

## 🐳 Docker

Build and run in Docker:

```bash
# Build image
docker build -f Dockerfile.api -t imaging-api:latest .

# Run container
docker run -p 3000:3000 \
  --env-file .env \
  imaging-api:latest

# Check health
curl http://localhost:3000/health
```

---

## 🛑 Graceful Shutdown

Server handles graceful shutdown:

```
Server receives SIGTERM/SIGINT
  ↓
Stops accepting new requests
  ↓
Waits for in-flight requests (max 30s)
  ↓
Closes database connections
  ↓
Closes Redis connections
  ↓
Exits with code 0
```

Send shutdown signal:

```bash
# Kill gracefully (SIGTERM)
kill -TERM <pid>

# Ctrl+C (SIGINT)
# Both trigger graceful shutdown
```

---

## 📝 Logging

Server logs to:

```
logs/
├── combined.log         # All logs
├── error.log            # Error level only
├── debug.log            # Debug level (development)
├── exceptions.log       # Uncaught exceptions
└── rejections.log       # Unhandled rejections
```

View logs:

```bash
# Tail combined logs
tail -f logs/combined.log

# Errors only
grep ERROR logs/error.log

# Specific level
grep WARN logs/combined.log

# JSON parsing (jq)
tail -f logs/combined.log | jq '.message'
```

---

## 🔍 Debugging

Enable debug logging:

```bash
# Set debug flag
export DEBUG=true

# Or in .env
DEBUG=true

# Start with debug output
npm run dev
```

View health check for diagnostics:

```bash
# Detailed health information
curl http://localhost:3000/health/detailed

# Full metrics
curl http://localhost:3000/health/metrics

# Service status
curl http://localhost:3000/health/status
```

---

## 🧪 Testing

Example health check test:

```bash
#!/bin/bash

echo "Testing server health..."

# Basic check
echo "1. Basic health check..."
curl -s http://localhost:3000/health | jq .

# Detailed check
echo "2. Detailed health..."
curl -s http://localhost:3000/health/detailed | jq .

# Kubernetes probes
echo "3. Readiness probe..."
curl -s http://localhost:3000/health/ready | jq .

echo "4. Liveness probe..."
curl -s http://localhost:3000/health/live | jq .

# Metrics
echo "5. Metrics..."
curl -s http://localhost:3000/health/metrics | jq .

echo "All tests complete!"
```

---

## 📋 Production Checklist

Before deploying to production:

- [ ] Generate strong JWT_SECRET
- [ ] Update DATABASE_URL with production credentials
- [ ] Set NODE_ENV=production
- [ ] Configure CORS_ORIGIN for your domain
- [ ] Set LOG_LEVEL=warn or error
- [ ] Enable HTTPS_ONLY=true
- [ ] Use environment variable vault
- [ ] Test all health check endpoints
- [ ] Load test with expected traffic
- [ ] Set up monitoring/alerting
- [ ] Configure database backups
- [ ] Test graceful shutdown
- [ ] Document API endpoints

---

## 🆘 Troubleshooting

**Server won't start:**

- Check PORT is available: `lsof -i :3000`
- Verify DATABASE_URL is correct
- Check logs: `tail -f logs/combined.log`

**Database connection fails:**

- Verify PostgreSQL is running
- Check DATABASE_URL format
- Test connection: `psql $DATABASE_URL -c "SELECT 1"`

**Health check fails:**

- Check database: `curl http://localhost:3000/health/detailed`
- Check Redis: `redis-cli ping`
- View logs for errors

**Rate limiting blocks requests:**

- Check RATE_LIMIT_MAX setting
- Disable for testing: `DEV_SKIP_RATE_LIMIT=true`
- Use different IP/user

**Authentication fails:**

- Verify JWT_SECRET is set
- Check token format: `Bearer <token>`
- Check token expiration

---

## 📞 Support

- **Status:** `GET /health`
- **Metrics:** `GET /health/metrics`
- **Logs:** Check `logs/` directory
- **Debug:** Enable `DEBUG=true`

---

## 🎯 Next Steps

1. **Add custom routes** → Create files in `src/routes/`
2. **Implement services** → Create files in `src/services/`
3. **Add database models** → Create files in `src/models/`
4. **Set up CI/CD** → GitHub Actions/Docker
5. **Deploy** → Docker/Kubernetes/Cloud
6. **Monitor** → Set up alerting on health endpoints

---

**Version:** 1.0.0
**Status:** ✅ PRODUCTION READY
**Last Updated:** October 17, 2025
