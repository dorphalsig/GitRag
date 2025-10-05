# Provisioning Reference

Use these artifacts when preparing Cloudflare resources for GitRag in CI/CD. They mirror the structures enforced by `PersistInVectorize` and can be applied manually or via automation.

## Cloudflare D1

1. Create a D1 database in the Cloudflare dashboard or via API and note the resulting database identifier (`CLOUDFLARE_D1_DATABASE_ID`).
2. Apply the schema:

   ```bash
   # Example using wrangler (must be authenticated):
   wrangler d1 execute <DATABASE_NAME> --file provisioning/d1/schema.sql
   ```

   The schema creates the `chunks` table plus secondary indexes (`repo_path`, `language`, `status`).

## Cloudflare Vectorize

1. Create the Vectorize index:

   ```bash
   curl -X POST \
     "https://api.cloudflare.com/client/v4/accounts/<CLOUDFLARE_ACCOUNT_ID>/vectorize/indexes" \
     -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
     -H "Content-Type: application/json" \
     --data @provisioning/vectorize/create-index.json
   ```

   Replace `<VECTORIZE_INDEX_NAME>` with your desired index name and `<EMBEDDING_DIMENSION>` with the output of `CodeRankCalculator().dimensions` (detected automatically at runtime).

2. Ensure metadata indexes exist for efficient filtering:

   ```bash
   while read -r line; do
     NAME=$(echo "$line" | jq -r '.property_name')
     curl -X POST \
       "https://api.cloudflare.com/client/v4/accounts/<CLOUDFLARE_ACCOUNT_ID>/vectorize/indexes/<VECTORIZE_INDEX_NAME>/metadata-index" \
       -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
       -H "Content-Type: application/json" \
       --data "$line";
   done < provisioning/vectorize/metadata-indexes.jsonl
   ```

## Environment Variables

After provisioning, supply these to the GitRag CLI or GitHub Action:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_VECTORIZE_INDEX`
- `CLOUDFLARE_D1_DATABASE_ID`

Keep this directory up to date if the schema changes.
