import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import Callout from "@/components/Callout";

export default function Architecture() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">Architecture</h1>
      <p className="text-lg text-zinc-400 mb-10">How oxidizedRAG is structured across three deployment targets.</p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Dependency Graph</h2>
      <CodeBlock>{`graphrag-server  → graphrag-core
graphrag-wasm    → graphrag-core
graphrag-cli     → graphrag-core
graphrag-leptos  → graphrag-wasm → graphrag-core`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Core Pipeline (7 stages)</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden text-[13px] mb-6">
        <table className="w-full"><tbody className="text-zinc-400">
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">1. Chunk</td><td className="px-5 py-3">Semantic chunking with configurable size and overlap</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">2. Extract</td><td className="px-5 py-3">Entity + relationship extraction via LLM</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">3. Graph</td><td className="px-5 py-3">Build knowledge graph from entities and relationships</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">4. Community</td><td className="px-5 py-3">Leiden community detection (+15% modularity vs Louvain)</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">5. Embed</td><td className="px-5 py-3">Vector embeddings via Ollama, ONNX, or hash fallback</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">6. Retrieve</td><td className="px-5 py-3">LightRAG dual-level retrieval + PageRank + cross-encoder reranking</td></tr>
          <tr><td className="px-5 py-3 text-emerald-400 font-mono font-medium">7. Synthesize</td><td className="px-5 py-3">LLM answer synthesis from retrieved context</td></tr>
        </tbody></table>
      </div>

      <Callout icon="📚">
        Each stage is trait-based and pluggable. Swap embedding providers, vector stores, or LLMs without touching the pipeline.
      </Callout>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Feature Flags</h2>
      <CodeBlock>{`[features]
memory-storage = []          # In-memory (dev)
persistent-storage = [...]   # LanceDB embedded vector DB
redis-storage = [...]        # Redis for distributed caching
parallel-processing = []     # Rayon parallelization
caching = [...]              # LLM response caching
lightrag = []                # Dual-level retrieval
pagerank = []                # Fast-GraphRAG retrieval
ollama = []                  # Ollama local models
metal = [...]                # Apple Silicon GPU
webgpu = [...]               # WebGPU (WASM)`}</CodeBlock>
    </Layout>
  );
}
