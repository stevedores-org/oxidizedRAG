import { Link } from "react-router-dom";
import Layout from "@/components/Layout";
import { Card, CardGrid } from "@/components/Card";
import Callout from "@/components/Callout";

export default function Home() {
  return (
    <Layout>
      <div className="pb-10 mb-10 border-b border-zinc-800/60">
        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight bg-gradient-to-br from-zinc-100 to-zinc-400 bg-clip-text text-transparent leading-tight">oxidizedRAG</h1>
        <p className="text-lg text-zinc-400 mt-3 leading-relaxed max-w-xl">
          High-performance Rust implementation of GraphRAG. Build knowledge graphs from documents and query them with natural language. Three deployment architectures: Server, WASM, and Hybrid.
        </p>
        <div className="flex gap-3 mt-6">
          <Link to="/getting-started" className="bg-emerald-500 hover:bg-emerald-600 text-black font-semibold px-5 py-2.5 rounded-lg transition text-sm">Get Started</Link>
          <a href="https://github.com/stevedores-org/oxidizedRAG" className="border border-zinc-700 hover:border-zinc-500 px-5 py-2.5 rounded-lg transition text-sm text-zinc-300">GitHub</a>
        </div>
      </div>

      <h2 className="text-2xl font-bold tracking-tight mb-3">Deployment Architectures</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden my-4">
        <table className="w-full text-[13px]"><tbody className="text-zinc-400">
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">Server-Only</td><td className="px-5 py-3">REST API + Qdrant + Ollama. Production ready. Best for SaaS / multi-tenant.</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">WASM-Only</td><td className="px-5 py-3">100% client-side with ONNX + WebLLM. Zero infrastructure. Privacy-first.</td></tr>
          <tr><td className="px-5 py-3 text-emerald-400 font-mono font-medium">Hybrid</td><td className="px-5 py-3">WASM client + optional server for heavy workloads. Best UX + scalability.</td></tr>
        </tbody></table>
      </div>

      <Callout icon="🚀">
        <strong className="text-zinc-100">Research-backed.</strong> Implements 5 cutting-edge papers: LightRAG (6000x token reduction), Leiden community detection, cross-encoder reranking, HippoRAG PageRank, semantic chunking. Combined: <strong className="text-zinc-100">+20% accuracy, 99% cost savings.</strong>
      </Callout>

      <h2 className="text-2xl font-bold tracking-tight mt-12 mb-3">Workspace Crates</h2>
      <CardGrid>
        <Card to="/crates/graphrag-core" title="graphrag-core" tag="core" description="Portable core library — LightRAG, PageRank, caching, incremental updates. Native + WASM." />
        <Card to="/crates/graphrag-server" title="graphrag-server" tag="server" description="Production REST API server with Qdrant, Ollama, Docker Compose deployment." />
        <Card to="/crates/graphrag-wasm" title="graphrag-wasm" tag="wasm" description="WASM bindings with ONNX Runtime Web, WebLLM, IndexedDB, and Cache API." />
        <Card to="/crates/graphrag-cli" title="graphrag-cli" tag="cli" description="CLI tools for building knowledge graphs and querying them from the terminal." />
      </CardGrid>
    </Layout>
  );
}
