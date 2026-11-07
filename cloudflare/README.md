# Cloudflare Python Worker deployment

This is an optional Phase 2 deployment. It uses the Python Workers ASGI
bridge, a D1 binding named `DB`, and a JSON configuration variable named
`LLMROUTER_CONFIG_JSON`. Provider credentials and the HTTP bearer token are
Worker secrets; they are never placed in the configuration JSON or D1.

1. Copy the Wrangler example and replace the D1 database ID:

   ```sh
   cp wrangler.toml.example wrangler.toml
   npx wrangler d1 create llmrouter
   npx wrangler d1 migrations apply llmrouter --remote
   ```

2. Put a validated JSON version of the router configuration in the Worker
   variable `LLMROUTER_CONFIG_JSON`. The repository's YAML example is a useful
   starting point, but every model, price, quota, endpoint, and data policy
   must be verified against the provider's current documentation.

3. Add secrets without committing them:

   ```sh
   npx wrangler secret put LLMROUTER_HTTP_API_KEY
   npx wrangler secret put OPENAI_API_KEY
   ```

4. Deploy with the Python Worker toolchain described by Cloudflare:

   ```sh
   uvx --from workers-py pywrangler dev
   uvx --from workers-py pywrangler deploy
   ```

The Python Workers platform is still an evolving/open-beta deployment target;
keep the `python_workers` compatibility flag and re-check the runtime/package
support before a production rollout.

The public endpoints are `GET /healthz`, authenticated `GET /v1/status`,
authenticated `POST /v1/call`, and authenticated `POST /v1/call/json`.
Use `Authorization: Bearer <LLMROUTER_HTTP_API_KEY>` from Node.js or any other
HTTP client. The health endpoint deliberately exposes no configuration or
database details.
