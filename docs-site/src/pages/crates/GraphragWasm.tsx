import Layout from "@/components/Layout";
import StatusBadge from "@/components/StatusBadge";

export default function GraphragWasm() {
  return (
    <Layout>
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-3xl font-extrabold tracking-tight">graphrag-wasm</h1>
        <StatusBadge status="wip" />
      </div>
      <p className="text-lg text-zinc-400 mb-10">
        WASM bindings for 100% client-side GraphRAG. GPU-accelerated embeddings via ONNX Runtime Web and LLM synthesis via WebLLM.
      </p>
      <p className="text-sm text-zinc-400 mb-8">
        crates.io: pending publish
        {" · "}
        <a className="text-emerald-400 hover:text-emerald-300" href="https://github.com/stevedores-org/oxidizedRAG" target="_blank" rel="noreferrer">source</a>
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Browser Stack</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden text-[13px]">
        <table className="w-full"><tbody className="text-zinc-400">
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">ONNX Runtime Web</td><td className="px-5 py-3">GPU embeddings in 3-8ms (25-40x speedup)</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">WebLLM</td><td className="px-5 py-3">Phi-3-mini for LLM synthesis (40-62 tok/s on GPU)</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-emerald-400 font-mono font-medium">IndexedDB</td><td className="px-5 py-3">Browser storage for knowledge graphs</td></tr>
          <tr><td className="px-5 py-3 text-emerald-400 font-mono font-medium">Cache API</td><td className="px-5 py-3">Model weight caching between sessions</td></tr>
        </tbody></table>
      </div>
    </Layout>
  );
}
