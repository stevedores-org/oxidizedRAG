import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import GettingStarted from "./pages/GettingStarted";
import Architecture from "./pages/Architecture";
import Deployment from "./pages/Deployment";
import GraphragCore from "./pages/crates/GraphragCore";
import GraphragServer from "./pages/crates/GraphragServer";
import GraphragWasm from "./pages/crates/GraphragWasm";
import GraphragCli from "./pages/crates/GraphragCli";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/getting-started" element={<GettingStarted />} />
      <Route path="/architecture" element={<Architecture />} />
      <Route path="/deployment" element={<Deployment />} />
      <Route path="/crates/graphrag-core" element={<GraphragCore />} />
      <Route path="/crates/graphrag-server" element={<GraphragServer />} />
      <Route path="/crates/graphrag-wasm" element={<GraphragWasm />} />
      <Route path="/crates/graphrag-cli" element={<GraphragCli />} />
    </Routes>
  );
}
