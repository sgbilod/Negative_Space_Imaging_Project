# Database Setup for Negative Space Imaging Project

This directory contains SQL scripts for initializing the database schema and adding initial data for the Negative Space Imaging Project.

## Scripts

- **01-init-schema.sql**: Creates the database schema including all tables, indexes, and triggers.
- **02-init-data.sql**: Inserts initial data such as admin user, default projects, and system settings.

## Database Structure

The database schema includes the following tables:

1. **users**: User accounts and authentication data
2. **projects**: Projects that organize image processing work
3. **image_sources**: Sources of astronomical images
4. **images**: Metadata for stored images
5. **processing_jobs**: Image processing job queue and status
6. **processing_results**: Results from processing jobs
7. **api_keys**: API keys for programmatic access
8. **audit_logs**: Security and activity audit logs
9. **project_user_access**: Project sharing and permissions
10. **system_settings**: System-wide configuration settings

## Usage with Docker

These scripts are automatically executed when the database container starts up. The PostgreSQL Docker image executes scripts in the `/docker-entrypoint-initdb.d/` directory in alphabetical order.

## Manual Database Setup

If you need to set up the database manually, you can use the following commands:

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE nsi;

# Connect to the new database
\c nsi

# Run the schema script
\i 01-init-schema.sql

# Run the data script
\i 02-init-data.sql
```

## Default Credentials

The initialization script creates the following default users:

- **Admin User**
  - Username: admin
  - Password: admin123 (change this in production!)
  - Email: admin@negativespaceimagingproject.com

- **Test User**
  - Username: test
  - Password: test123 (change this in production!)
  - Email: test@negativespaceimagingproject.com

**Warning**: These default credentials are for development purposes only. In a production environment, you should change these credentials immediately after setup.

## Backup and Restore

To backup the database:

```bash
pg_dump -U postgres -d nsi > nsi_backup.sql
```

To restore the database:

```bash
psql -U postgres -d nsi < nsi_backup.sql
```
