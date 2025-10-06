# Spec Tasks

- [ ] 1. Build libSQL persistence adapter
- [ ] 1.1 Write tests for libSQL persistence adapter covering upsert/delete flows via SQLAlchemy
- [ ] 1.2 Initialize SQLAlchemy engine using Turso database URL + auth token environment variables
- [ ] 1.3 Port write/delete operations to execute SQL/FTS statements through SQLAlchemy connections
  - [ ] 1.4 Verify all tests pass

- [ ] 2. Ship libSQL schema and migrations
  - [ ] 2.1 Write schema/provisioning tests for applying libSQL DDL
  - [ ] 2.2 Create libSQL DDL (tables, DiskANN index, FTS) under provisioning/libsql
  - [ ] 2.3 Retire legacy provisioning assets and update CI workflows to call libSQL setup
  - [ ] 2.4 Verify all tests pass

- [ ] 3. Update configuration and operational docs
  - [ ] 3.1 Write tests for configuration/env var parsing under the libSQL adapter selection
- [ ] 3.2 Replace legacy environment variables with Turso database URL + auth token in CLI + config helpers
  - [ ] 3.3 Refresh docs/runbooks with libSQL provisioning, query, and troubleshooting guidance
  - [ ] 3.4 Verify all tests pass
