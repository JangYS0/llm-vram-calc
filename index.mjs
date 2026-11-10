export const SUPPORTED_MODELS = [
  {
    id: "llama-4-scout-17b",
    name: "Llama 4 Scout (17B)",
    params: 17.2,
    activeParams: 17.2,
    layers: 48,
    hiddenDim: 5120,
    qHeads: 40,
    kvHeads: 8,
    headDim: 128,
    attentionType: "gqa",
    maxContext: 131072,
    defaultContext: 8192,
  },
  {
    id: "llama-4-maverick-400b",
    name: "Llama 4 Maverick (400B)",
    params: 405.0,
    activeParams: 405.0,
    layers: 126,
    hiddenDim: 16384,
    qHeads: 128,
    kvHeads: 16,
    headDim: 128,
    attentionType: "gqa",
    maxContext: 131072,
    defaultContext: 8192,
  },
  {
    id: "deepseek-r1-v3-671b",
    name: "DeepSeek R1 / V3 (671B MoE)",
    params: 671.0,
    activeParams: 37.0,
    layers: 61,
    hiddenDim: 7168,
    qHeads: 128,
    kvHeads: 128,
    headDim: 128,
    attentionType: "mla",
    mlaDc: 512,
    mlaDr: 64,
    maxContext: 131072,
    defaultContext: 8192,
  },
  {
    id: "qwen-3-72b",
    name: "Qwen 3 (72B)",
    params: 72.7,
    activeParams: 72.7,
    layers: 80,
    hiddenDim: 8192,
    qHeads: 64,
    kvHeads: 8,
    headDim: 128,
    attentionType: "gqa",
    maxContext: 131072,
    defaultContext: 8192,
  },
  {
    id: "qwen-3-32b",
    name: "Qwen 3 (32B)",
    params: 32.5,
    activeParams: 32.5,
    layers: 64,
    hiddenDim: 5120,
    qHeads: 40,
    kvHeads: 8,
    headDim: 128,
    attentionType: "gqa",
    maxContext: 131072,
    defaultContext: 8192,
  },
  {
    id: "gemma-3-27b",
    name: "Gemma 3 (27B)",
    params: 27.2,
    activeParams: 27.2,
    layers: 46,
    hiddenDim: 4608,
    qHeads: 32,
    kvHeads: 16,
    headDim: 128,
    attentionType: "gqa",
    maxContext: 131072,
    defaultContext: 8192,
  },
  {
    id: "mistral-large-123b",
    name: "Mistral Large (123B)",
    params: 123.0,
    activeParams: 123.0,
    layers: 88,
    hiddenDim: 12288,
    qHeads: 96,
    kvHeads: 8,
    headDim: 128,
    attentionType: "gqa",
    maxContext: 131072,
    defaultContext: 8192,
  },
];

export const SUPPORTED_QUANTIZATIONS = [
  { id: "fp16", name: "FP16 / BF16 (16-bit)", bytesPerParam: 2.0, bitsPerParam: 16 },
  { id: "fp8", name: "FP8 (E4M3 / E5M2)", bytesPerParam: 1.0, bitsPerParam: 8 },
  { id: "int8", name: "INT8 (8-bit Quant)", bytesPerParam: 1.0, bitsPerParam: 8 },
  { id: "int4", name: "INT4 / AWQ / GPTQ", bytesPerParam: 0.55, bitsPerParam: 4.4 },
  { id: "q4_k_m", name: "GGUF Q4_K_M (Medium)", bytesPerParam: 0.5625, bitsPerParam: 4.5 },
  { id: "q5_k_m", name: "GGUF Q5_K_M (High)", bytesPerParam: 0.6875, bitsPerParam: 5.5 },
  { id: "q8_0", name: "GGUF Q8_0 (Near-Lossless)", bytesPerParam: 1.0625, bitsPerParam: 8.5 },
];

export const HARDWARE_TIERS = [
  { id: "rtx-5090-1x", name: "1x NVIDIA RTX 5090 (32GB GDDR7)", vramGb: 32, bandwidthGbs: 1792, hourlyOnDemand: 0.99, hourlySpot: 0.69 },
  { id: "rtx-5090-2x", name: "2x NVIDIA RTX 5090 (64GB Total)", vramGb: 64, bandwidthGbs: 3584, hourlyOnDemand: 1.98, hourlySpot: 1.38 },
  { id: "nvidia-h100-1x", name: "1x NVIDIA H100 SXM5 (80GB HBM3)", vramGb: 80, bandwidthGbs: 3350, hourlyOnDemand: 2.85, hourlySpot: 1.99 },
  { id: "mac-studio-192", name: "Mac Studio M4 Ultra (192GB Unified)", vramGb: 192, bandwidthGbs: 800, hourlyOnDemand: 2.20, hourlySpot: 1.80 },
  { id: "nvidia-b200-1x", name: "1x NVIDIA B200 SXM (192GB HBM3e)", vramGb: 192, bandwidthGbs: 8000, hourlyOnDemand: 5.50, hourlySpot: 3.90 },
  { id: "mac-studio-512", name: "Mac Studio M4 Ultra (512GB Unified)", vramGb: 512, bandwidthGbs: 1600, hourlyOnDemand: 4.20, hourlySpot: 3.50 },
  { id: "nvidia-h100-8x", name: "8x NVIDIA H100 SXM5 (640GB NVLink)", vramGb: 640, bandwidthGbs: 26800, hourlyOnDemand: 22.80, hourlySpot: 15.90 },
  { id: "nvidia-b200-8x", name: "8x NVIDIA B200 SXM (1,536GB NVLink)", vramGb: 1536, bandwidthGbs: 64000, hourlyOnDemand: 44.00, hourlySpot: 31.20 },
];

export function calculateVram(input = {}) {
  const model = SUPPORTED_MODELS.find((m) => m.id === input.modelId) || SUPPORTED_MODELS[0];
  const quantization = SUPPORTED_QUANTIZATIONS.find((q) => q.id === input.quantizationId) || SUPPORTED_QUANTIZATIONS[0];

  const contextLength = input.contextLength !== undefined ? Math.max(1, Math.floor(input.contextLength)) : model.defaultContext;
  const batchSize = input.batchSize !== undefined ? Math.max(1, Math.floor(input.batchSize)) : 1;
  const cudaOverheadGb = input.cudaOverheadGb !== undefined ? Math.max(0, input.cudaOverheadGb) : 1.5;
  const kvPrecisionBytes = input.kvPrecisionBytes !== undefined ? input.kvPrecisionBytes : (quantization.bytesPerParam <= 1.0 ? 1 : 2);

  const weightsGb = Number((model.params * quantization.bytesPerParam).toFixed(4));
  const activeParams = model.activeParams ?? model.params;
  const activeWeightsGb = Number((activeParams * quantization.bytesPerParam).toFixed(4));

  const isMla = model.attentionType === "mla";
  const kvBytesPerToken = isMla
    ? model.layers * ((model.mlaDc ?? 512) + (model.mlaDr ?? 64)) * kvPrecisionBytes
    : 2 * model.layers * model.kvHeads * model.headDim * kvPrecisionBytes;

  const kvCacheGb = Number(((batchSize * contextLength * kvBytesPerToken) / 1e9).toFixed(4));
  const isPrefill = input.isPrefill ?? true;
  const activationPrefill = Math.max(1.0, (batchSize * contextLength * model.hiddenDim * 4) / 1e9);
  const activationDecode = Math.max(0.5, (batchSize * model.hiddenDim * model.layers * 2) / 1e9);
  const activationGb = Number((isPrefill ? activationPrefill : activationDecode).toFixed(4));

  const totalVramGb = Number((weightsGb + kvCacheGb + activationGb + cudaOverheadGb).toFixed(4));
  const recommendedVramGb = Number((totalVramGb * 1.15).toFixed(4));
  const allMatchingHardware = HARDWARE_TIERS.filter((tier) => tier.vramGb >= totalVramGb);
  const recommendedHardware = HARDWARE_TIERS.find((tier) => tier.vramGb >= recommendedVramGb) || allMatchingHardware[0] || null;

  const targetHardware = recommendedHardware || HARDWARE_TIERS[HARDWARE_TIERS.length - 1];
  const decodeSpeedTokensPerSec = Number((targetHardware.bandwidthGbs / Math.max(0.1, activeWeightsGb)).toFixed(2));
  const costPer1MTokensOnDemand = Number(((targetHardware.hourlyOnDemand / (3600 * Math.max(1, decodeSpeedTokensPerSec))) * 1_000_000).toFixed(4));

  return {
    model,
    quantization,
    contextLength,
    batchSize,
    weightsGb,
    activeWeightsGb,
    kvCacheGb,
    kvBytesPerToken,
    activationGb,
    cudaOverheadGb,
    totalVramGb,
    recommendedVramGb,
    isMla,
    recommendedHardware,
    decodeSpeedTokensPerSec,
    costPer1MTokensOnDemand,
  };
}
