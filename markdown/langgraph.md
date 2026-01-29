{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/anubis/graph.py:graph"
  },
  "env": ".env.dev",
  "http": {
    "app": "./src/anubis/webapp.py:app"
  }
}
