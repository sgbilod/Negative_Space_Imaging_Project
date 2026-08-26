# REST API Quick Start Guide

## Getting Started with the Negative Space Imaging API

### Base URL

```
http://localhost:3000/api/v1
```

### 1. Register a New User

```bash
curl -X POST http://localhost:3000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePassword123!",
    "firstName": "John",
    "lastName": "Doe"
  }'
```

**Response:**

```json
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

Save the `accessToken` for authenticated requests.

---

### 2. Upload an Image

```bash
curl -X POST http://localhost:3000/api/v1/images/upload \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@/path/to/image.fits" \
  -F "title=Galaxy Survey Image" \
  -F "description=High resolution astronomical image" \
  -F "tags=astronomy" \
  -F "tags=deep-space"
```

**Response:**

```json
{
  "id": "img-123",
  "userId": "user-123",
  "filename": "galaxy-1729160400000-a7b3.fits",
  "title": "Galaxy Survey Image",
  "fileSize": 52428800,
  "mimeType": "image/fits",
  "width": 2048,
  "height": 2048,
  "status": "pending",
  "uploadedAt": "2025-10-17T10:00:00Z"
}
```

Save the `id` for later operations.

---

### 3. List Your Images

```bash
curl -X GET "http://localhost:3000/api/v1/images?limit=10" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response:**

```json
{
  "data": [
    {
      "id": "img-123",
      "title": "Galaxy Survey Image",
      "status": "completed",
      "uploadedAt": "2025-10-17T10:00:00Z"
    }
  ],
  "pagination": {
    "limit": 10,
    "offset": 0,
    "total": 5,
    "totalPages": 1
  }
}
```

---

### 4. Analyze an Image

```bash
curl -X POST http://localhost:3000/api/v1/images/img-123/analyze \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithmType": "advanced",
    "priority": "high",
    "parameters": {
      "sensitivityThreshold": 0.8
    }
  }'
```

**Response:**

```json
{
  "jobId": "job-456",
  "imageId": "img-123",
  "status": "queued",
  "algorithmType": "advanced",
  "priority": "high",
  "position": 3,
  "estimatedProcessingTime": 120000,
  "createdAt": "2025-10-17T10:00:00Z"
}
```

---

### 5. Get Analysis Results

```bash
curl -X GET http://localhost:3000/api/v1/analysis/img-123 \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response:**

```json
[
  {
    "id": "result-1",
    "imageId": "img-123",
    "algorithmType": "advanced",
    "status": "completed",
    "duration": 120000,
    "results": {
      "detectionsCount": 42,
      "confidence": 0.95,
      "spatialMetrics": { ... },
      "frequencyAnalysis": { ... }
    },
    "completedAt": "2025-10-17T10:02:00Z"
  }
]
```

---

### 6. Export Analysis Results

```bash
curl -X GET "http://localhost:3000/api/v1/analysis/img-123/export?format=csv" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -o results.csv
```

---

### 7. Compare Analyses

```bash
curl -X POST http://localhost:3000/api/v1/analysis/img-123/compare \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "compareImageIds": ["img-124", "img-125"]
  }'
```

**Response:**

```json
{
  "baseImageId": "img-123",
  "comparisonImageIds": ["img-124", "img-125"],
  "similarityScores": {
    "img-124": 0.92,
    "img-125": 0.78
  },
  "recommendations": [
    "img-123 and img-124 are highly similar",
    "img-125 shows different characteristics"
  ]
}
```

---

## Admin Operations

### Admin Login

```bash
curl -X POST http://localhost:3000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "password": "AdminPassword123!"
  }'
```

---

### View System Statistics

```bash
curl -X GET http://localhost:3000/api/v1/admin/stats \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"
```

**Response:**

```json
{
  "processing": {
    "totalAnalyses": 1250,
    "completedAnalyses": 1100,
    "failedAnalyses": 15,
    "queueSize": 135
  },
  "storage": {
    "totalImages": 850,
    "totalSize": 536870912000,
    "availableSpace": 2147483648000
  },
  "users": {
    "totalUsers": 42,
    "activeUsers24h": 28
  },
  "system": {
    "cpuUsage": 45.2,
    "memoryUsage": 62.8
  }
}
```

---

### View Processing Queue

```bash
curl -X GET "http://localhost:3000/api/v1/admin/queue?limit=20" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"
```

---

### Adjust Job Priority

```bash
curl -X POST http://localhost:3000/api/v1/admin/queue/job-456/priority \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "priority": "high",
    "reason": "VIP user request"
  }'
```

---

### Access System Logs

```bash
curl -X GET "http://localhost:3000/api/v1/admin/logs?level=error&limit=50" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"
```

---

### Update System Configuration

```bash
curl -X POST http://localhost:3000/api/v1/admin/config \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "category": "security",
    "settings": {
      "passwordMinLength": 14,
      "sessionTimeout": 3600000
    },
    "reason": "Enhanced security policy"
  }'
```

---

## Error Handling

### Validation Error (400)

```json
{
  "status": 400,
  "message": "Validation failed",
  "errors": [
    {
      "field": "email",
      "message": "Invalid email format"
    }
  ]
}
```

### Unauthorized (401)

```json
{
  "status": 401,
  "message": "Authentication failed",
  "error": "Invalid credentials"
}
```

### Forbidden (403)

```json
{
  "status": 403,
  "message": "Access denied",
  "error": "Admin role required"
}
```

### Not Found (404)

```json
{
  "status": 404,
  "message": "Resource not found",
  "resource": "Image"
}
```

---

## Token Management

### Refresh Access Token

When your access token expires (401 error):

```bash
curl -X POST http://localhost:3000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refreshToken": "YOUR_REFRESH_TOKEN"
  }'
```

**Response:**

```json
{
  "accessToken": "eyJhbGciOiJIUzI1NiIs..."
}
```

---

### Logout

```bash
curl -X POST http://localhost:3000/api/v1/auth/logout \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

## Pagination

Most list endpoints support pagination:

```bash
# Get second page with 10 items per page
curl "http://localhost:3000/api/v1/images?limit=10&offset=10" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

Query parameters:

- `limit` - Items per page (1-100, default: 20)
- `offset` - Starting position (default: 0)
- `sort` - Sort field (endpoint-specific)
- `order` - Sort order: asc or desc (default: desc)

---

## File Upload

### Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- DICOM (.dcm)
- FITS (.fits)

### Size Limit

- Maximum: 500 MB

---

## Rate Limiting

- **Limit:** 100 requests per 15 minutes (per user)
- **Headers in Response:**
  - `RateLimit-Limit`
  - `RateLimit-Remaining`
  - `RateLimit-Reset`

---

## JavaScript/TypeScript Example

```typescript
// Using fetch API
const token = 'your_access_token';

// Upload image
const formData = new FormData();
formData.append('file', imageFile);
formData.append('title', 'My Image');
formData.append('description', 'Image description');

const uploadResponse = await fetch('http://localhost:3000/api/v1/images/upload', {
  method: 'POST',
  headers: { Authorization: `Bearer ${token}` },
  body: formData,
});

const image = await uploadResponse.json();
console.log('Image uploaded:', image.id);

// List images
const listResponse = await fetch('http://localhost:3000/api/v1/images?limit=10', {
  headers: { Authorization: `Bearer ${token}` },
});

const images = await listResponse.json();
console.log('Your images:', images.data);

// Analyze image
const analyzeResponse = await fetch(`http://localhost:3000/api/v1/images/${image.id}/analyze`, {
  method: 'POST',
  headers: {
    Authorization: `Bearer ${token}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    algorithmType: 'advanced',
    priority: 'high',
  }),
});

const job = await analyzeResponse.json();
console.log('Analysis job created:', job.jobId);
```

---

## Troubleshooting

### 401 Unauthorized

- Check that your access token is valid and not expired
- Refresh token using `/auth/refresh` endpoint
- Ensure token is in `Authorization: Bearer {token}` format

### 403 Forbidden

- Admin endpoints require admin role
- Check user permissions
- Try logging in as admin user

### 404 Not Found

- Verify the resource ID (image, job, etc.)
- Check the endpoint path spelling
- Ensure the resource hasn't been deleted

### 409 Conflict

- Resource already exists (e.g., email already registered)
- Use different value and retry

### 500 Server Error

- Check server logs
- Try again in a few moments
- Contact support if persists

---

## Support

- **Documentation:** See `API_REFERENCE.md` for complete endpoint documentation
- **Issues:** Check `REST_API_IMPLEMENTATION_SUMMARY.md` for implementation details
- **Contact:** support@negative-space-imaging.dev

---

**API Version:** 1.0
**Last Updated:** 2025-10-17
**Base URL:** http://localhost:3000/api/v1
