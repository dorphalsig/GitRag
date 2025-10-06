# Spec Tasks

- [x] 1. Ship libSQL persistence adapter
  - [x] 1.1 Write tests for libSQL adapter covering connection setup, upsert/delete flows, and vector round-trips
- [x] 1.2 Implement SQLAlchemy-based libSQL engine handling driven by Turso env vars in the persistence layer
- [x] 1.3 Port CLI persistence calls to use the libSQL adapter, including logging for batch sizes and failures
  - [x] 1.4 Verify all tests pass

- [ ] 2. Deliver libSQL schema provisioning
  - [ ] 2.1 Write schema/provisioning tests validating table, ANN index, and FTS creation
  - [ ] 2.2 Add libSQL DDL/migrations with repo, path, repo+path indexes and DiskANN setup
  - [ ] 2.3 Retire legacy persistence assets and update bootstrapping scripts to call libSQL setup
  - [ ] 2.4 Verify all tests pass

- [ ] 3. Update configuration and documentation
  - [ ] 3.1 Write tests for environment parsing and adapter selection covering new libSQL settings
- [ ] 3.2 Replace legacy configuration/env vars in CLI and dev tooling with Turso database URL + auth token envs
  - [ ] 3.3 Refresh docs/runbooks with libSQL provisioning, hybrid query SQL, and troubleshooting steps
  - [ ] 3.4 Verify all tests pass
