import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import StatusBadge from "@/components/StatusBadge";

export default function GraphragCore() {
  return (
    <Layout>
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-3xl font-extrabold tracking-tight">graphrag-core</h1>
        <StatusBadge status="done" />
      </div>
      <p className="text-lg text-zinc-400 mb-10">
        Portable core library. LightRAG, PageRank, caching, incremental updates. Works on native and WASM targets.
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">API Surface</h2>
      <CodeBlock>{`// One-line API
let answer = simple::answer("Your document", "Your question")?;

// Stateful API
let mut graph = SimpleGraphRAG::from_text("Your document")?;
let a1 = graph.ask("Question 1")?;
let a2 = graph.ask("Question 2")?;

// Builder API
let mut rag = GraphRAG::builder()
    .with_preset(ConfigPreset::Balanced)
    .auto_detect_llm()
    .build()?;
rag.add_document("Your document")?;
let answer = rag.ask("Your question")?;`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Research Features</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden text-[13px]">
        <table className="w-full"><tbody className="text-zinc-400">
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">LightRAG</td><td className="px-5 py-3">Dual-level retrieval. 6000x token reduction. EMNLP 2025.</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">Leiden</td><td className="px-5 py-3">Community detection. +15% modularity. Sci Reports 2019.</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">Cross-Encoder</td><td className="px-5 py-3">Reranking. +20% accuracy. EMNLP 2019.</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">HippoRAG</td><td className="px-5 py-3">Personalized PageRank. 10-30x cheaper. NeurIPS 2024.</td></tr>
          <tr><td className="px-5 py-3 text-emerald-400 font-mono font-medium">Semantic Chunking</td><td className="px-5 py-3">Better boundaries. LangChain 2024.</td></tr>
        </tbody></table>
      </div>
    </Layout>
  );
}
