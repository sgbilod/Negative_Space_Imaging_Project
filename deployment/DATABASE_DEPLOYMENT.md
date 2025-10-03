# Database Deployment System for Negative Space Imaging Project

## Overview

The database deployment system provides a robust framework for managing PostgreSQL databases across development, testing, and production environments for the Negative Space Imaging Project. It handles database creation, schema initialization, migrations, backups, and integration with the HPC environment.

## Components

The system consists of the following components:

1. **Database Connection Utility** (`database_connection.py`): Provides connection pooling and query execution capabilities.
2. **Database Deployment Manager** (`database_deploy.py`): Handles deployment, verification, migrations, and backups.
3. **HPC Database Integration** (`hpc_database_integration.py`): Integrates database operations with HPC computations.
4. **Schema and Data Files**:
   - `01-init-schema.sql`: Database schema with tables, indexes, and triggers.
   - `02-init-data.sql`: Initial data for basic system operation.
   - Migration files in the `migrations` directory.
5. **Test Script** (`test_database_deployment.py`): Verifies database deployment and functionality.

## Configuration

The system is configured using YAML files:

- `deployment/config/database.yaml`: Main configuration file with connection details, pool settings, schema paths, and backup settings.

Example configuration:

```yaml
database:
  # Connection settings
  host: localhost
  port: 5432
  dbname: negative_space_imaging
  user: postgres
  password: postgres
  timeout: 30

  # Pool settings
  min_connections: 1
  max_connections: 10

  # Schema settings
  schema_file: "deployment/database/01-init-schema.sql"
  data_file: "deployment/database/02-init-data.sql"
  migrations_dir: "deployment/database/migrations"

  # Backup settings
  backup_dir: "deployment/database/backups"
  backup_retention_days: 30
```

## Database Schema

The database schema includes tables for:

- **Users and Authentication**: `users`, `api_keys`
- **Projects**: `projects`, `project_members`
- **Images**: `images`, `image_versions`, `image_verifications`
- **Computations**: `computations`, `computation_logs`
- **Deployments**: `deployments`, `deployment_logs`, `system_nodes`
- **Security and Monitoring**: `security_logs`, `system_events`, `settings`

## Usage

### Deploying the Database

To deploy a new database or update an existing one:

```bash
python deployment/database_deploy.py --deploy --config deployment/config/database.yaml
```

### Verifying the Deployment

To verify the database is correctly set up:

```bash
python deployment/database_deploy.py --verify --config deployment/config/database.yaml
```

### Running Migrations

To apply pending database migrations:

```bash
python deployment/database_deploy.py --migrate --config deployment/config/database.yaml
```

### Creating a Backup

To create a database backup:

```bash
python deployment/database_deploy.py --backup --config deployment/config/database.yaml
```

### Restoring from Backup

To restore a database from backup:

```bash
python deployment/database_deploy.py --restore --backup-file deployment/database/backups/negative_space_imaging_backup_20230101_120000.sql --config deployment/config/database.yaml --force
```

### Testing the Deployment

To run a basic deployment test:

```bash
python deployment/test_database_deployment.py --config deployment/config/database.yaml
```

For a more comprehensive test including write operations:

```bash
python deployment/test_database_deployment.py --config deployment/config/database.yaml --full
```

### HPC Integration

The HPC integration component provides utilities for HPC nodes to interact with the database:

```bash
# Register an HPC node in the database
python deployment/hpc_database_integration.py --register-node --config deployment/config/database.yaml

# Register a computation
python deployment/hpc_database_integration.py --register-computation --project-id <project_id> --user-id <user_id> --name "Deep Space Analysis" --type "image_processing" --config deployment/config/database.yaml

# Update computation status
python deployment/hpc_database_integration.py --update-computation --computation-id <computation_id> --status "completed" --config deployment/config/database.yaml

# Log a system event
python deployment/hpc_database_integration.py --log-event --type "hpc.node.status" --message "Node is running at 95% CPU capacity" --severity "warning" --config deployment/config/database.yaml
```

## Connection Pooling

The database connection utility uses connection pooling to efficiently manage database connections. To use it in your code:

```python
from deployment.database_connection import init_db_pool, execute_query

# Initialize the connection pool
init_db_pool("deployment/config/database.yaml")

# Execute a query
result, success = execute_query(
    "SELECT * FROM users WHERE username = %s",
    ("admin",),
    fetch_one=True
)

if success and result:
    print(f"Found user: {result.get('username')}")
```

## Security Considerations

- Database passwords are stored in configuration files and should be properly secured.
- For production environments, consider using environment variables or secrets management tools.
- SSL connections should be enabled for production environments by setting `ssl_mode` to `require` or `verify-full` and providing appropriate certificates.

## Deployment Recommendations

### Development Environment

- Use local PostgreSQL installation with default settings.
- Frequent migrations and schema changes are expected.
- Backups are optional.

### Testing Environment

- Use a dedicated PostgreSQL server or container.
- Regular migrations to match development.
- Daily backups recommended.

### Production Environment

- Use high-availability PostgreSQL setup with replication.
- Carefully managed migrations with rollback plans.
- Frequent backups with off-site storage.
- Monitoring and alerting for database performance and security.

## Dependencies

- Python 3.7+
- PostgreSQL 12+
- Required Python packages:
  - psycopg2
  - pyyaml
  - [Optional] GPUtil (for GPU information in HPC nodes)

## Extending the System

### Adding a New Table

1. Create a migration file in `deployment/database/migrations` with the naming convention `V<version>__<description>.sql`.
2. Run the migration with `python deployment/database_deploy.py --migrate`.

### Adding HPC Integration Features

Modify `hpc_database_integration.py` to add new functions for HPC-specific database operations.

### Customizing Backup Strategy

Modify the backup settings in `database.yaml` and consider adding a scheduled task (cron job) to run backups regularly.

## Troubleshooting

### Connection Issues

- Verify PostgreSQL is running: `pg_isready -h <host> -p <port>`
- Check firewall settings
- Validate username and password

### Migration Failures

- Check migration logs in `database_deployment.log`
- Use `--force` flag to attempt to continue past errors
- Manually inspect database state with `psql`

### Performance Problems

- Adjust pool settings (`min_connections` and `max_connections`)
- Monitor query performance with PostgreSQL tools
- Consider adding indexes for frequently queried columns
