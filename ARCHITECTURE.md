# System Architecture

**Copyright © 2025 Stephen Bilodeau. All rights reserved.**

## Table of Contents

- [Overview](#overview)
- [High-Level Architecture](#high-level-architecture)
- [Components](#components)
- [Data Flow](#data-flow)
- [Security](#security)
- [Deployment](#deployment)
- [Scalability](#scalability)
- [Monitoring](#monitoring)
- [Development Principles](#development-principles)

---

## Overview

The Negative Space Imaging Project is a full-stack enterprise application designed for analyzing negative space in images using advanced AI/ML techniques.

---

## High-Level Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  ├─ Dashboard                                                │
│  ├─ Image Upload                                             │
│  ├─ Analysis Results Viewer                                  │
│  └─ Settings & Profile Management                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    REST API / WebSocket
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Node Backend │  │Python Service│  │   Cache      │
│  (Express)   │  │  (FastAPI)   │  │   (Redis)    │
│              │  │              │  │              │
│ - Auth       │  │ - Analysis   │  │ - Sessions   │
│ - API Routes │  │ - ML Models  │  │ - Results    │
│ - Validation │  │ - Processing │  │ - Metadata   │
└──────┬───────┘  └──────┬───────┘  └──────────────┘
       │                 │
       └─────────┬───────┘
                 │
        ┌────────▼────────┐
        │   Database      │
        │ (PostgreSQL)    │
        │                 │
        │ - Users         │
        │ - Images        │
        │ - Results       │
        │ - Metadata      │
        └─────────────────┘
```

## Components

### Frontend Layer

**Technology:** React 18, TypeScript, Tailwind CSS

**Key Components:**
- Dashboard: Main interface for users
- ImageUploader: Handles image selection and upload
- AnalysisViewer: Displays processing results
- SettingsPanel: User configuration
- AuthFlow: Login/registration system

**State Management:**
- Redux or Context API for global state
- Local state for component-specific data
- WebSocket connection for real-time updates

### API Layer (Node.js/Express)

**Technology:** Express.js, TypeScript, Joi validation

**Responsibilities:**
- User authentication & authorization
- Image upload handling
- API endpoint routing
- Request validation
- Error handling
- WebSocket management

**Key Endpoints:**
```
POST   /api/auth/register
POST   /api/auth/login
POST   /api/auth/refresh
POST   /api/images/upload
GET    /api/images/:id
POST   /api/analysis/start
GET    /api/analysis/:id
GET    /api/results/:imageId
```

### Python Service (AI/ML Engine)

**Technology:** FastAPI, Python 3.10+

**Responsibilities:**
- Image loading and preprocessing
- Negative space detection algorithms
- ML model inference
- Result generation
- Performance optimization

**Key Endpoints:**
```
POST   /api/analyze
POST   /api/train-model
GET    /api/model-status
POST   /api/batch-process
```

### Data Layer

**Database:** PostgreSQL 14+

**Schema:**
- Users table (authentication, profile)
- Images table (uploaded files, metadata)
- Analysis table (processing results)
- Sessions table (authentication tokens)
- Audit logs (compliance tracking)

**Caching:** Redis for:
- Session storage
- Result caching
- Rate limiting
- Real-time updates

## Data Flow

### Image Analysis Flow

```
1. User uploads image (Frontend)
   ↓
2. File sent to Node backend (API validation)
   ↓
3. Image stored in database + filesystem
   ↓
4. Analysis job queued (Redis/Bull)
   ↓
5. Python service retrieves job
   ↓
6. Preprocessing pipeline runs
   ↓
7. ML model inference
   ↓
8. Results computed
   ↓
9. Results cached (Redis)
   ↓
10. Database updated
   ↓
11. Frontend notified via WebSocket
   ↓
12. User views results
```

## Security

### Authentication
- JWT tokens for API security
- Refresh token rotation
- Secure cookie storage
- Multi-signature support for critical operations

### Authorization
- Role-based access control (RBAC)
- Resource ownership verification
- Rate limiting per user
- IP whitelisting (enterprise feature)

### Data Protection
- Encryption at rest (PostgreSQL)
- HTTPS/TLS for transport
- Secret management via environment variables
- Audit logging for compliance

### Validation
- Input validation (Joi schemas)
- Type checking (TypeScript, Pydantic)
- CORS configuration
- CSRF protection

## Deployment

### Development
- Docker Compose for local development
- Hot-reload enabled
- Debug logging active
- Mock data seeding

### Staging
- Docker containers in Kubernetes
- Real database connection
- Basic monitoring
- Pre-production testing

### Production
- Kubernetes orchestration
- Auto-scaling policies
- Load balancing
- CDN integration
- Comprehensive monitoring
- Automated backups

## Scalability

### Horizontal Scaling
- Stateless Node backend (multiple replicas)
- Python service can scale independently
- Database read replicas
- Cache distributed (Redis Cluster)

### Vertical Scaling
- Resource allocation per pod
- Memory management
- CPU optimization
- Connection pooling

### Optimization
- Database query optimization
- Result caching strategy
- Image compression
- Batch processing

## Monitoring

**Tools:** Prometheus, Grafana, ELK Stack

**Metrics:**
- API response times
- Error rates
- Resource utilization
- Cache hit rates
- Analysis processing time

**Alerting:**
- High error rates (>5%)
- Response time degradation (>2s)
- Resource limits (>80%)
- Service unavailability

## Development Principles

1. **Separation of Concerns:** Each service has single responsibility
2. **API First:** Design APIs before implementation
3. **Type Safety:** TypeScript & Python type hints throughout
4. **Testing:** Unit, integration, and E2E tests
5. **Documentation:** Self-documenting code + API docs
6. **Security:** Security by default
7. **Performance:** Optimize where it matters
8. **Observability:** Logging and monitoring built-in

## Related Documentation

### Architecture Documents

| Document | Description |
|----------|-------------|
| [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) | Detailed system architecture with layer diagrams |
| [SYSTEM_ARCHITECTURE_DETAILED.md](./SYSTEM_ARCHITECTURE_DETAILED.md) | In-depth technical architecture details |
| [PHASE_5_ARCHITECTURE_DIAGRAMS.md](./PHASE_5_ARCHITECTURE_DIAGRAMS.md) | Visual architecture diagrams |
| [EXPRESS_SERVER_ARCHITECTURE.md](./EXPRESS_SERVER_ARCHITECTURE.md) | Express.js backend architecture |

### Other Related Documents

| Document | Description |
|----------|-------------|
| [README.md](./README.md) | Project overview |
| [GETTING_STARTED.md](./GETTING_STARTED.md) | Setup guide |
| [REQUIREMENTS.md](./REQUIREMENTS.md) | Dependencies |
| [README_IMAGE_ACQUISITION.md](./README_IMAGE_ACQUISITION.md) | Image acquisition pipeline |
| [DOCKER_DEPLOYMENT_GUIDE.md](./DOCKER_DEPLOYMENT_GUIDE.md) | Container deployment |
| [API_REFERENCE.md](./API_REFERENCE.md) | API documentation |

---

Last Updated: November 2025
