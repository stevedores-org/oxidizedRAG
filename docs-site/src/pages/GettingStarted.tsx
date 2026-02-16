import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import Callout from "@/components/Callout";

export default function GettingStarted() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">Getting Started</h1>
      <p className="text-lg text-zinc-400 mb-10">Build a knowledge graph and query it in minutes.</p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Quick Start (Server)</h2>
      <CodeBlock>{`git clone https://github.com/stevedores-org/oxidizedRAG.git
cd oxidizedRAG

# Start Qdrant (optional)
cd graphrag-server && docker-compose up -d

# Start Ollama for embeddings
ollama serve &
ollama pull nomic-embed-text

# Start GraphRAG server
export EMBEDDING_BACKEND=ollama
cargo run --release --bin graphrag-server --features "qdrant,ollama"`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Quick Start (WASM)</h2>
      <CodeBlock>{`cargo install trunk wasm-bindgen-cli
cd graphrag-wasm
trunk serve --open`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Programmatic Usage</h2>
      <CodeBlock>{`use graphrag_rs::simple;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let answer = simple::answer("Your document text", "Your question")?;
    println!("Answer: {}", answer);
    Ok(())
}`}</CodeBlock>

      <Callout icon="💡">
        The <code className="text-emerald-300/90 font-mono text-[13px]">simple</code> API is a one-liner. For stateful multi-query sessions, use <code className="text-emerald-300/90 font-mono text-[13px]">SimpleGraphRAG::from_text()</code>.
      </Callout>
    </Layout>
  );
}
