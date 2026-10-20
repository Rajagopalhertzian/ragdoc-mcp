# API Reference Guide

## Overview

This document provides a comprehensive reference for the REST API endpoints.

## Authentication

All API requests require authentication via Bearer token:

```
Authorization: Bearer <your-token>
```

## Endpoints

### GET /api/v1/users

Returns a list of users.

**Response:**
```json
{
  "users": [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"}
  ]
}
```

### POST /api/v1/users

Create a new user.

**Request:**
```json
{
  "name": "Charlie",
  "email": "charlie@example.com"
}
```

**Response:**
```json
{
  "id": 3,
  "name": "Charlie",
  "email": "charlie@example.com"
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 500 | Internal Server Error |