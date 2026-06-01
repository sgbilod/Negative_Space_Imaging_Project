# Development Environment Setup

Complete setup guide for the Negative Space Imaging Project development environment.

## Prerequisites

- Node.js 18+ and npm 9+
- Python 3.10+
- Docker & Docker Compose
- Git
- VS Code (optional but recommended)

## Quick Start

### 1. Install Dependencies

```bash
# Node.js dependencies
npm install

# Python dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Development Environment

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or run services individually
npm run dev          # Node backend
python -m negative_space.api  # Python service
npm run dev:frontend # React frontend
```

### 3. Verify Setup

```bash
# Check Node backend
curl http://localhost:3000/health

# Check Python service
curl http://localhost:5000/health

# View React app
open http://localhost:3000
```

## Environment Variables

Create `.env.local` (ignored by git):

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/negative_space
DATABASE_PASSWORD=secure_password

# Redis
REDIS_URL=redis://localhost:6379

# API Keys
JWT_SECRET=your_jwt_secret_here
API_KEY=your_api_key_here

# Services
NODE_ENV=development
PYTHON_ENV=development
```

## Docker Compose

The project includes a complete Docker setup:

```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after dependency changes
docker-compose build --no-cache
```

## Database Setup

```bash
# Run migrations
npm run migrate

# Seed development data
npm run seed:dev

# View database
npm run db:studio
```

## Testing

```bash
# Frontend tests
npm run test:frontend

# Backend tests
npm run test:backend

# Python tests
pytest tests/

# All tests
npm run test

# With coverage
npm run test:coverage
```

## Code Quality

```bash
# Linting
npm run lint

# Format code
npm run format

# Type checking
npm run type-check

# Security audit
npm audit
```

## Debugging

### Node Backend

```bash
# Enable debug logging
DEBUG=* npm run dev

# VS Code debugger
# Click "Run and Debug" then select "Node Backend"
```

### Python Service

```bash
# Enable debug logging
PYTHONUNBUFFERED=1 python -m negative_space.api

# VS Code debugger
# Create launch configuration in .vscode/launch.json
```

## Troubleshooting

### Port Already in Use

```bash
# Node (3000)
lsof -i :3000
kill -9 <PID>

# Python (5000)
lsof -i :5000
kill -9 <PID>

# PostgreSQL (5432)
docker-compose restart postgres
```

### Database Connection Issues

```bash
# Check container is running
docker ps | grep postgres

# Restart PostgreSQL
docker-compose restart postgres

# Check logs
docker-compose logs postgres
```

### Module Not Found

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Python: Activate venv
source venv/bin/activate
pip install -r requirements.txt
```

## Performance Tips

1. Use Docker for consistent environments
2. Enable hot-reloading in development
3. Use Redis for caching
4. Monitor with provided tools
5. Profile code regularly

## Contributing

See `CONTRIBUTING.md` for guidelines.

## Support

- Check `ARCHITECTURE.md` for system design
- See `DEPLOYMENT.md` for production setup
- Review `README.md` for project overview

---

Last Updated: November 2025
