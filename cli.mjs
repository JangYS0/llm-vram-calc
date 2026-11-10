#!/usr/bin/env node
import { calculateVram, SUPPORTED_MODELS, SUPPORTED_QUANTIZATIONS } from "./index.mjs";

const args = process.argv.slice(2);
const getArg = (flag, fallback) => {
  const idx = args.indexOf(flag);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : fallback;
};

if (args.includes("--help") || args.includes("-h")) {
  console.log(`Usage: llm-vram-calc [options]

Options:
  --model <id>     Model ID (${SUPPORTED_MODELS.map((m) => m.id).join(", ")})
  --quant <id>     Quantization (${SUPPORTED_QUANTIZATIONS.map((q) => q.id).join(", ")})
  --context <n>    Context window tokens (default: 8192)
  --batch <n>      Concurrent batch size (default: 1)
  --kv-bytes <n>   KV cache precision in bytes: 2=FP16, 1=FP8, 0.5=INT4
  --json           Output JSON breakdown

Interactive Web UI: https://global-utils.com/en/gpu-vram-calculator`);
  process.exit(0);
}

const kvArg = getArg("--kv-bytes", null);
const res = calculateVram({
  modelId: getArg("--model", "deepseek-r1-v3-671b"),
  quantizationId: getArg("--quant", "fp8"),
  contextLength: Number(getArg("--context", "8192")),
  batchSize: Number(getArg("--batch", "1")),
  ...(kvArg !== null ? { kvPrecisionBytes: Number(kvArg) } : {}),
});

if (args.includes("--json")) {
  console.log(JSON.stringify(res, null, 2));
  process.exit(0);
}

console.log(`\n=== LLM GPU VRAM & Cost Breakdown ===`);
console.log(`Model        : ${res.model.name} (${res.isMla ? "DeepSeek MLA 576-dim" : `GQA ${res.model.qHeads}:${res.model.kvHeads}`})`);
console.log(`Quantization : ${res.quantization.name} (${res.quantization.bytesPerParam} B/param)`);
console.log(`Context      : ${res.contextLength.toLocaleString()} tokens (Batch: ${res.batchSize})`);
console.log(`-------------------------------------`);
console.log(`Model Weights       : ${res.weightsGb.toFixed(2)} GB`);
console.log(`KV Cache            : ${res.kvCacheGb.toFixed(2)} GB (${res.kvBytesPerToken.toLocaleString()} B/token)`);
console.log(`Peak Activations    : ${res.activationGb.toFixed(2)} GB`);
console.log(`CUDA Context        : ${res.cudaOverheadGb.toFixed(2)} GB`);
console.log(`-------------------------------------`);
console.log(`Total Required VRAM : ${res.totalVramGb.toFixed(2)} GB`);
console.log(`Recommended (+15%)  : ${res.recommendedVramGb.toFixed(2)} GB`);
console.log(`Target Hardware     : ${res.recommendedHardware ? res.recommendedHardware.name : "Multi-Node NVLink Cluster"}`);
console.log(`Est. Decode Speed   : ~${res.decodeSpeedTokensPerSec} tok/s ($${res.costPer1MTokensOnDemand.toFixed(3)} / 1M tokens)`);
console.log(`Web UI              : https://global-utils.com/en/gpu-vram-calculator\n`);
