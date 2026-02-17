import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import Callout from "@/components/Callout";

export default function Deployment() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">Deployment</h1>
      <p className="text-lg text-zinc-400 mb-10">Server, WASM, and Hybrid deployment options.</p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Server Deployment</h2>
      <p className="text-zinc-400 text-[15px] mb-4">
        REST API via <code className="text-emerald-300/90 font-mono text-[13px]">graphrag-server</code>. Docker Compose for Qdrant + Ollama.
      </p>
      <CodeBlock>{`# Docker Compose
cd graphrag-server
docker-compose up -d

# Or standalone binary (5.2MB optimized)
cargo build --release --bin graphrag-server --features "qdrant,ollama"
./target/release/graphrag-server`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">WASM Deployment</h2>
      <p className="text-zinc-400 text-[15px] mb-4">
        100% client-side. GPU-accelerated embeddings via ONNX Runtime Web (3-8ms). LLM synthesis via WebLLM (Phi-3-mini, 40-62 tok/s).
      </p>
      <CodeBlock>{`cd graphrag-wasm
trunk build --release
# Deploy dist/ to any static host`}</CodeBlock>

      <Callout icon="🔒">
        <strong className="text-zinc-100">Privacy first.</strong> WASM mode keeps all data in the browser. No documents leave the client. Uses IndexedDB for graphs and Cache API for models.
      </Callout>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Configuration</h2>
      <CodeBlock>{`# config.toml
[general]
input_document_path = "path/to/document.txt"
output_dir = "./output"

[pipeline]
chunk_size = 800
chunk_overlap = 200

[ollama]
enabled = true
host = "http://localhost"
port = 11434
chat_model = "llama3.1:8b"
embedding_model = "nomic-embed-text"

[enhancements]
enabled = true

[enhancements.lightrag]
enabled = true
max_keywords = 20

[enhancements.leiden]
enabled = true
max_cluster_size = 10`}</CodeBlock>
    </Layout>
  );
}
