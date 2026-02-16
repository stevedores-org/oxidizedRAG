import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import StatusBadge from "@/components/StatusBadge";

export default function GraphragCli() {
  return (
    <Layout>
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-3xl font-extrabold tracking-tight">graphrag-cli</h1>
        <StatusBadge status="done" />
      </div>
      <p className="text-lg text-zinc-400 mb-10">
        CLI tools for building knowledge graphs and querying them from the terminal.
      </p>
      <p className="text-sm text-zinc-400 mb-8">
        <a className="text-emerald-400 hover:text-emerald-300" href="https://crates.io/crates/graphrag-cli" target="_blank" rel="noreferrer">crates.io</a>
        {" · "}
        <a className="text-emerald-400 hover:text-emerald-300" href="https://github.com/stevedores-org/oxidizedRAG" target="_blank" rel="noreferrer">source</a>
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Smart CLI (Recommended)</h2>
      <CodeBlock>{`# Process document and answer question in one command
cargo run --bin simple_cli config.toml "What are the main themes?"

# Interactive mode
cargo run --bin simple_cli config.toml`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Manual CLI</h2>
      <CodeBlock>{`# Build knowledge graph
./target/release/graphrag-rs config.toml build

# Query the graph
./target/release/graphrag-rs config.toml query "Your question"`}</CodeBlock>
    </Layout>
  );
}
