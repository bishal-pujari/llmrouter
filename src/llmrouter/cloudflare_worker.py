"""Python Worker entrypoint for the optional Cloudflare deployment."""

from workers import asgi

from llmrouter.cloudflare_app import app


Default = asgi.entrypoint(app)
