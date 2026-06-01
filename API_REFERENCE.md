# Negative Space Imaging Project - Complete REST API Reference

## API Overview

**Base URL:** `/api/v1/`
**Total Endpoints:** 23
**Authentication:** JWT (Bearer token)
**Response Format:** JSON

---

## 📋 Table of Contents

1. [Authentication Endpoints](#authentication)
2. [User Management Endpoints](#users)
3. [Image Management Endpoints](#images)
4. [Analysis Results Endpoints](#analysis)
5. [Admin Management Endpoints](#admin)
6. [Error Responses](#errors)
7. [Request/Response Examples](#examples)

---

## Authentication

### Register User

```
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123",
  "firstName": "John",
  "lastName": "Doe"
}

Response: 201 Created
{
  "accessToken": "eyJhbGciOiJIUzI1NiIs...",
  "refreshToken": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "id": "user-123",
    "email": "user@example.com",
    "firstName": "John",
    "lastName": "Doe",
    "roles": ["user"],
    "createdAt": "2025-10-17T10:00:00Z"
  }
}
```

### Login User

```
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123"
}

Response: 200 OK
{
  "accessToken": "eyJhbGciOiJIUzI1NiIs...",
  "refreshToken": "eyJhbGciOiJIUzI1NiIs...",
  "user": { ... }
}
```

### Refresh Token

```
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refreshToken": "eyJhbGciOiJIUzI1NiIs..."
}

Response: 200 OK
{
  "accessToken": "eyJhbGciOiJIUzI1NiIs..."
}
```

### Logout

```
POST /api/v1/auth/logout
Authorization: Bearer {accessToken}

Response: 200 OK
{
  "message": "Logged out successfully"
}
```

### Verify Token

```
GET /api/v1/auth/verify
Authorization: Bearer {accessToken}

Response: 200 OK
{
  "userId": "user-123",
  "email": "user@example.com",
  "roles": ["user"],
  "valid": true,
  "timestamp": "2025-10-17T10:00:00Z"
}
```

---

## Users

### Get Current User Profile

```
GET /api/v1/users/profile
Authorization: Bearer {accessToken}

Response: 200 OK
{
  "id": "user-123",
  "email": "user@example.com",
  "firstName": "John",
  "lastName": "Doe",
  "avatar": "https://...",
  "bio": "Image analysis enthusiast",
  "preferences": {
    "emailNotifications": true,
    "darkMode": false,
    "defaultAnalysisType": "advanced"
  },
  "roles": ["user"],
  "createdAt": "2025-10-17T10:00:00Z"
}
```

### Update User Profile

```
PUT /api/v1/users/profile
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "firstName": "John",
  "lastName": "Doe",
  "avatar": "https://...",
  "bio": "Updated bio",
  "preferences": {
    "emailNotifications": false,
    "darkMode": true,
    "defaultAnalysisType": "ai-powered"
  }
}

Response: 200 OK
{
  "id": "user-123",
  "email": "user@example.com",
  "firstName": "John",
  "lastName": "Doe",
  "avatar": "https://...",
  "bio": "Updated bio",
  "preferences": { ... },
  "roles": ["user"],
  "createdAt": "2025-10-17T10:00:00Z"
}
```

### List All Users (Admin Only)

```
GET /api/v1/users?limit=20&offset=0&sort=created&order=desc
Authorization: Bearer {adminToken}

Response: 200 OK
{
  "data": [
    {
      "id": "user-123",
      "email": "user@example.com",
      "firstName": "John",
      "lastName": "Doe",
      "roles": ["user"],
      "createdAt": "2025-10-17T10:00:00Z"
    },
    ...
  ],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "total": 125,
    "totalPages": 7
  },
  "timestamp": "2025-10-17T10:00:00Z"
}
```

### Get User by ID (Admin Only)

```
GET /api/v1/users/:id
Authorization: Bearer {adminToken}

Response: 200 OK
{
  "id": "user-456",
  "email": "other@example.com",
  "firstName": "Jane",
  "lastName": "Smith",
  "roles": ["user"],
  "createdAt": "2025-10-15T14:20:00Z"
}
```

### Delete User (Admin Only)

```
DELETE /api/v1/users/:id
Authorization: Bearer {adminToken}

Response: 200 OK
{
  "message": "User deleted successfully",
  "userId": "user-456"
}
```

---

## Images

### Upload Image

```
POST /api/v1/images/upload
Authorization: Bearer {accessToken}
Content-Type: multipart/form-data

Form Data:
- file: [binary image data]
- title: "Galaxy Image 01"
- description: "High resolution astronomical image"
- tags: ["astronomy", "deep-space"]
- metadata: { source: "telescope-1" }

Response: 201 Created
{
  "id": "img-123",
  "userId": "user-123",
  "filename": "galaxy-1729160400000-a7b3.fits",
  "originalName": "galaxy.fits",
  "title": "Galaxy Image 01",
  "description": "High resolution astronomical image",
  "tags": ["astronomy", "deep-space"],
  "metadata": { source: "telescope-1" },
  "fileSize": 52428800,
  "mimeType": "image/fits",
  "width": 2048,
  "height": 2048,
  "status": "pending",
  "uploadedAt": "2025-10-17T10:00:00Z"
}
```

### List User's Images

```
GET /api/v1/images?limit=20&offset=0&sort=created&order=desc
Authorization: Bearer {accessToken}

Response: 200 OK
{
  "data": [
    {
      "id": "img-123",
      "title": "Galaxy Image 01",
      "description": "High resolution astronomical image",
      "fileSize": 52428800,
      "mimeType": "image/fits",
      "status": "completed",
      "analysisCount": 2,
      "uploadedAt": "2025-10-17T10:00:00Z"
    },
    ...
  ],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "total": 85,
    "totalPages": 5
  },
  "timestamp": "2025-10-17T10:00:00Z"
}
```

### Get Image Details

```
GET /api/v1/images/:id
Authorization: Bearer {accessToken}

Response: 200 OK
{
  "id": "img-123",
  "userId": "user-123",
  "filename": "galaxy-1729160400000-a7b3.fits",
  "originalName": "galaxy.fits",
  "title": "Galaxy Image 01",
  "description": "High resolution astronomical image",
  "tags": ["astronomy", "deep-space"],
  "metadata": { source: "telescope-1" },
  "fileSize": 52428800,
  "mimeType": "image/fits",
  "width": 2048,
  "height": 2048,
  "status": "completed",
  "analysisCount": 2,
  "uploadedAt": "2025-10-17T10:00:00Z"
}
```

### Delete Image

```
DELETE /api/v1/images/:id
Authorization: Bearer {accessToken}

Response: 200 OK
{
  "message": "Image deleted successfully",
  "imageId": "img-123"
}
```

### Download Image

```
GET /api/v1/images/:id/download
Authorization: Bearer {accessToken}

Response: 200 OK
[Binary file stream]
Headers:
  Content-Type: image/fits
  Content-Disposition: attachment; filename="galaxy.fits"
```

### Analyze Image

```
POST /api/v1/images/:id/analyze
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "algorithmType": "advanced",
  "priority": "high",
  "parameters": {
    "sensitivityThreshold": 0.8,
    "regionSize": 32
  }
}

Response: 202 Accepted
{
  "jobId": "job-123",
  "imageId": "img-123",
  "userId": "user-123",
  "status": "queued",
  "algorithmType": "advanced",
  "priority": "high",
  "position": 3,
  "estimatedProcessingTime": 120000,
  "createdAt": "2025-10-17T10:00:00Z"
}
```

---

## Analysis

### Get Analysis Results

```
GET /api/v1/analysis/:imageId
Authorization: Bearer {accessToken}

Response: 200 OK
[
  {
    "id": "result-1",
    "imageId": "img-123",
    "userId": "user-123",
    "algorithmType": "advanced",
    "status": "completed",
    "startedAt": "2025-10-17T09:45:00Z",
    "completedAt": "2025-10-17T10:00:00Z",
    "duration": 900000,
    "results": {
      "spatialMetrics": {
        "centralBrightness": 0.95,
        "edgeContrast": 0.82,
        "morphology": "elliptical"
      },
      "frequencyAnalysis": {
        "dominantFrequencies": [0.5, 1.2, 2.1],
        "noiseLevel": 0.15
      },
      "statistics": {
        "mean": 128.5,
        "stdDev": 45.2,
        "skewness": 0.25
      },
      "detectionsCount": 42,
      "confidence": 0.95
    },
    "quality": {
      "signalNoiseRatio": 12.5,
      "contrast": 0.8,
      "brightness": 0.7,
      "sharpness": 0.85
    }
  }
]
```

### Export Analysis Results

```
GET /api/v1/analysis/:imageId/export?format=csv&includeMetadata=true
Authorization: Bearer {accessToken}

Response: 200 OK (CSV)
id,algorithm,status,duration,detectionsCount,confidence,...
result-1,advanced,completed,900000,42,0.95,...
```

### Compare Analyses

```
POST /api/v1/analysis/:imageId/compare
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "compareImageIds": ["img-124", "img-125"],
  "metrics": ["contrast", "sharpness", "detectionsCount"]
}

Response: 200 OK
{
  "baseImageId": "img-123",
  "comparisonImageIds": ["img-124", "img-125"],
  "similarityScores": {
    "img-124": 0.92,
    "img-125": 0.78
  },
  "differences": {
    "spatialMetrics": {
      "contrast": { base: 0.8, "img-124": 0.82, "img-125": 0.75 }
    },
    "frequencyAnalysis": { ... }
  },
  "recommendations": [
    "img-123 and img-124 are highly similar in spatial characteristics",
    "img-125 shows different frequency components"
  ],
  "timestamp": "2025-10-17T10:00:00Z"
}
```

### List User's Analyses

```
GET /api/v1/analysis?limit=20&offset=0&sort=created&order=desc&status=completed
Authorization: Bearer {accessToken}

Response: 200 OK
{
  "data": [
    {
      "id": "result-1",
      "imageId": "img-123",
      "algorithmType": "advanced",
      "status": "completed",
      "duration": 900000,
      "completedAt": "2025-10-17T10:00:00Z"
    },
    ...
  ],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "total": 125,
    "totalPages": 7
  },
  "timestamp": "2025-10-17T10:00:00Z"
}
```

---

## Admin

### Get System Statistics

```
GET /api/v1/admin/stats
Authorization: Bearer {adminToken}

Response: 200 OK
{
  "timestamp": "2025-10-17T10:00:00Z",
  "processing": {
    "totalAnalyses": 1250,
    "completedAnalyses": 1100,
    "failedAnalyses": 15,
    "averageProcessingTime": 145000,
    "queueSize": 135
  },
  "storage": {
    "totalImages": 850,
    "totalSize": 536870912000,
    "availableSpace": 2147483648000
  },
  "users": {
    "totalUsers": 42,
    "activeUsers24h": 28,
    "activeUsers7d": 38
  },
  "system": {
    "cpuUsage": 45.2,
    "memoryUsage": 62.8,
    "diskUsage": 25.1,
    "uptime": 604800,
    "errors24h": 3
  }
}
```

### View Processing Queue

```
GET /api/v1/admin/queue?limit=50&offset=0&sort=priority&order=desc
Authorization: Bearer {adminToken}

Response: 200 OK
{
  "data": [
    {
      "jobId": "job-123",
      "imageId": "img-456",
      "userId": "user-789",
      "algorithm": "advanced",
      "priority": "high",
      "status": "processing",
      "createdAt": "2025-10-17T09:30:00Z",
      "estimatedProcessingTime": 120000,
      "position": 1
    },
    ...
  ],
  "pagination": {
    "total": 135,
    "limit": 50,
    "offset": 0,
    "totalPages": 3
  },
  "timestamp": "2025-10-17T10:00:00Z"
}
```

### Update Job Priority

```
POST /api/v1/admin/queue/:jobId/priority
Authorization: Bearer {adminToken}
Content-Type: application/json

{
  "priority": "high",
  "reason": "VIP user request"
}

Response: 200 OK
{
  "jobId": "job-123",
  "priority": "high",
  "previousPriority": "normal",
  "updatedAt": "2025-10-17T10:00:00Z",
  "updatedBy": "admin-1",
  "reason": "VIP user request"
}
```

### Access System Logs

```
GET /api/v1/admin/logs?level=error&limit=100&search=analysis
Authorization: Bearer {adminToken}

Response: 200 OK
{
  "data": [
    {
      "id": "log-123",
      "timestamp": "2025-10-17T09:45:00Z",
      "level": "error",
      "service": "analysis-engine",
      "message": "Failed to process image: Out of memory",
      "metadata": {
        "imageId": "img-123",
        "userId": "user-456",
        "errorCode": "OOM_ERROR"
      }
    },
    ...
  ],
  "pagination": {
    "limit": 100,
    "offset": 0,
    "total": 342,
    "totalPages": 4
  },
  "timestamp": "2025-10-17T10:00:00Z"
}
```

### Update System Configuration

```
POST /api/v1/admin/config
Authorization: Bearer {adminToken}
Content-Type: application/json

{
  "category": "security",
  "settings": {
    "passwordMinLength": 14,
    "sessionTimeout": 3600000,
    "maxFailedLoginAttempts": 5
  },
  "reason": "Enhanced security policy per compliance requirement"
}

Response: 200 OK
{
  "id": "config-update-123",
  "timestamp": "2025-10-17T10:00:00Z",
  "category": "security",
  "previousSettings": {
    "passwordMinLength": 12,
    "sessionTimeout": 7200000,
    "maxFailedLoginAttempts": 3
  },
  "newSettings": {
    "passwordMinLength": 14,
    "sessionTimeout": 3600000,
    "maxFailedLoginAttempts": 5
  },
  "updatedBy": "admin-1",
  "reason": "Enhanced security policy per compliance requirement"
}
```

---

## Error Responses

### Validation Error (400)

```json
{
  "status": 400,
  "message": "Validation failed",
  "errors": [
    {
      "field": "email",
      "message": "Invalid email format"
    },
    {
      "field": "password",
      "message": "Password must be at least 12 characters"
    }
  ],
  "requestId": "req-abc123",
  "timestamp": "2025-10-17T10:00:00Z"
}
```

### Authentication Error (401)

```json
{
  "status": 401,
  "message": "Authentication failed",
  "error": "Invalid credentials",
  "requestId": "req-abc123",
  "timestamp": "2025-10-17T10:00:00Z"
}
```

### Authorization Error (403)

```json
{
  "status": 403,
  "message": "Access denied",
  "error": "Admin role required",
  "requestId": "req-abc123",
  "timestamp": "2025-10-17T10:00:00Z"
}
```

### Not Found Error (404)

```json
{
  "status": 404,
  "message": "Resource not found",
  "resource": "User",
  "requestId": "req-abc123",
  "timestamp": "2025-10-17T10:00:00Z"
}
```

### Conflict Error (409)

```json
{
  "status": 409,
  "message": "Resource conflict",
  "error": "Email already registered",
  "requestId": "req-abc123",
  "timestamp": "2025-10-17T10:00:00Z"
}
```

### Server Error (500)

```json
{
  "status": 500,
  "message": "Internal server error",
  "error": "Failed to process request",
  "requestId": "req-abc123",
  "timestamp": "2025-10-17T10:00:00Z"
}
```

---

## HTTP Status Codes

| Code | Meaning      | Usage                                    |
| ---- | ------------ | ---------------------------------------- |
| 200  | OK           | Successful GET, PUT, DELETE              |
| 201  | Created      | Successful POST                          |
| 202  | Accepted     | Async operation accepted (analysis jobs) |
| 400  | Bad Request  | Validation failed                        |
| 401  | Unauthorized | Missing/invalid authentication           |
| 403  | Forbidden    | Insufficient permissions                 |
| 404  | Not Found    | Resource doesn't exist                   |
| 409  | Conflict     | Resource already exists                  |
| 500  | Server Error | Unhandled server error                   |

---

## Authentication

### Header Format

```
Authorization: Bearer {accessToken}
```

### Token Types

- **Access Token:** Short-lived (15 minutes), used for API requests
- **Refresh Token:** Long-lived (7 days), used to get new access tokens

### Token Refresh Flow

1. Access token expires (401 response)
2. Send refresh token to `POST /api/v1/auth/refresh`
3. Receive new access token
4. Retry original request

---

## Pagination

All list endpoints support pagination:

### Query Parameters

- `limit` (1-100, default: 20) - Items per page
- `offset` (default: 0) - Starting position
- `sort` - Sort field (varies by endpoint)
- `order` (asc/desc, default: desc) - Sort order

### Response Format

```json
{
  "data": [...],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "total": 125,
    "totalPages": 7
  }
}
```

---

## File Upload

### Supported Formats

- JPEG (image/jpeg)
- PNG (image/png)
- TIFF (image/tiff)
- DICOM (application/dicom)
- FITS (image/fits)

### Size Limits

- Maximum file size: 500 MB
- Supported extensions: .jpg, .jpeg, .png, .tiff, .tif, .dcm, .fits

---

## Rate Limiting

- **Window:** 15 minutes
- **Limit:** 100 requests per user (by ID) or IP
- **Exclusions:** Health check endpoints

### Response Headers

```
RateLimit-Limit: 100
RateLimit-Remaining: 95
RateLimit-Reset: 1729160400
```

---

## CORS

**Allowed Origins:** Configurable via environment
**Allowed Methods:** GET, POST, PUT, PATCH, DELETE, OPTIONS
**Allowed Headers:** Content-Type, Authorization, X-Request-ID
**Max Age:** 3600 seconds

---

## Health Checks

### Quick Health Check

```
GET /health
GET /api/v1/health

Response: 200 OK
{
  "status": "ok",
  "timestamp": "2025-10-17T10:00:00Z",
  "api": "v1"
}
```

---

**API Documentation v1.0**
**Last Updated: 2025-10-17**
**Contact: support@negative-space-imaging.dev**
