import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import StatusBadge from "@/components/StatusBadge";

export default function GraphragServer() {
  return (
    <Layout>
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-3xl font-extrabold tracking-tight">graphrag-server</h1>
        <StatusBadge status="done" />
      </div>
      <p className="text-lg text-zinc-400 mb-10">
        Production REST API with Actix-web 4.9, Apistos (OpenAPI 3.0.3), Qdrant, and Ollama.
      </p>
      <p className="text-sm text-zinc-400 mb-8">
        <a className="text-emerald-400 hover:text-emerald-300" href="https://crates.io/crates/graphrag-server" target="_blank" rel="noreferrer">crates.io</a>
        {" · "}
        <a className="text-emerald-400 hover:text-emerald-300" href="https://github.com/stevedores-org/oxidizedRAG" target="_blank" rel="noreferrer">source</a>
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Endpoints</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden text-[13px] mb-6">
        <table className="w-full"><tbody className="text-zinc-400">
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">POST /documents</td><td className="px-5 py-3">Add documents to the knowledge graph</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">POST /query</td><td className="px-5 py-3">Query the graph with natural language</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">GET /health</td><td className="px-5 py-3">Health check with system status</td></tr>
          <tr><td className="px-5 py-3 text-emerald-400 font-mono font-medium">GET /openapi.json</td><td className="px-5 py-3">Auto-generated OpenAPI 3.0.3 spec</td></tr>
        </tbody></table>
      </div>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Docker Compose</h2>
      <CodeBlock>{`cd graphrag-server
docker-compose up -d

# Starts:
#  - graphrag-server on :8080
#  - Qdrant on :6333
#  - Ollama for embeddings`}</CodeBlock>
    </Layout>
  );
}
