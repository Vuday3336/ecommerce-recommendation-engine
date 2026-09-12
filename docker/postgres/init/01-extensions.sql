-- Runs once, on first initialisation of an empty data volume.
--
-- The migration also creates the vector extension (a migration must be
-- self-sufficient so a fresh non-Docker database works too). Doing it here as
-- well means the extension exists before Alembic runs, which keeps the
-- migration's CREATE EXTENSION a no-op rather than the step that needs
-- superuser rights at an awkward moment.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- pg_trgm backs fuzzy product-name matching in the keyword stage of
-- personalised search (FR-08); full-text search alone misses typos.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Session timezone is forced to UTC by the application connection settings;
-- ALTER DATABASE is intentionally not used here because the database name is
-- configurable and psql variable interpolation is not available in this hook.
