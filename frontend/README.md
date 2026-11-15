# AskPolicy frontend

React + TypeScript + Vite client for the AskPolicy policy assistant.

## Commands

```powershell
npm install
npm run dev      # http://localhost:5173
npm run build    # type-check and create dist/
npm run lint     # oxlint
npm run preview  # serve the production build locally
```

## Configuration

The API base URL defaults to `http://127.0.0.1:8000`. To point the UI at a
different backend, create `frontend/.env.local`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Only the backend URL is exposed to the browser. Groq and Qdrant credentials
stay in the backend `.env` file and are never sent to the frontend.

See the repository [README](../README.md) for full setup and architecture
documentation.
