let ortSession = null;
let appConfig = {
  model_name: "densenet121",
  img_size: 256,
  threshold: 0.12,
  top_k: 128,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
};

let selectedFiles = [];
let triageResults = [];
let currentFilter = "ALL";
let activeModalItem = null;

// DOM Elements
const statusDot = document.getElementById("model-status-dot");
const statusText = document.getElementById("model-status-text");
const configBadge = document.getElementById("config-badge");
const btnSelectFiles = document.getElementById("btn-select-files");
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const selectedBox = document.getElementById("selected-preview-box");
const selectedCountLabel = document.getElementById("selected-count-label");
const thumbnailStrip = document.getElementById("thumbnail-strip");
const btnClearFiles = document.getElementById("btn-clear-files");
const btnRunTriage = document.getElementById("btn-run-triage");
const progressContainer = document.getElementById("progress-container");
const progressLabel = document.getElementById("progress-label");
const progressPercent = document.getElementById("progress-percent");
const progressFill = document.getElementById("progress-fill");

const viewUpload = document.getElementById("view-upload");
const viewQueue = document.getElementById("view-queue");
const btnBackUpload = document.getElementById("btn-back-upload");
const queueTbody = document.getElementById("queue-tbody");

// Modal Elements
const inspectModal = document.getElementById("inspect-modal");
const modalTitle = document.getElementById("modal-title");
const modalSubtitle = document.getElementById("modal-subtitle");
const canvasOriginal = document.getElementById("canvas-original");
const canvasOverlay = document.getElementById("canvas-overlay");
const toggleMask = document.getElementById("toggle-mask");

// ==================== 1. INITIALIZE ONNX MODEL ====================
async function initModel() {
  try {
    const cfgRes = await fetch("assets/config.json");
    if (cfgRes.ok) {
      appConfig = { ...appConfig, ...(await cfgRes.json()) };
    }

    configBadge.textContent = `Threshold: ${appConfig.threshold} | Top-K: ${appConfig.top_k}`;
    statusText.textContent = `กำลังโหลดน้ำหนักโมเดล (${appConfig.model_name})...`;

    // ตั้งค่า WASM path จาก CDN เพื่อความเข้ากันได้ 100% บน GitHub Pages
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

    const modelPath = `assets/${appConfig.model_name}_triage.onnx`;
    ortSession = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["wasm"],
    });

    statusDot.classList.add("ready");
    statusText.textContent = `พร้อมใช้งาน (${appConfig.model_name} • Client-Side ONNX)`;
    btnSelectFiles.disabled = false;
  } catch (err) {
    console.error("Failed to load ONNX model:", err);
    statusDot.classList.add("error");
    statusText.textContent = "ไม่พบไฟล์โมเดล assets/*.onnx (กรุณารัน export_onnx.py ก่อน)";
  }
}

// ==================== 2. FILE SELECTION & DRAG-DROP ====================
function handleFiles(fileList) {
  const validFiles = Array.from(fileList).filter((f) =>
    ["image/png", "image/jpeg"].includes(f.type)
  );
  if (validFiles.length === 0) return;

  selectedFiles = [...selectedFiles, ...validFiles];
  renderThumbnails();
}

function renderThumbnails() {
  if (selectedFiles.length === 0) {
    selectedBox.classList.add("hidden");
    return;
  }

  selectedBox.classList.remove("hidden");
  selectedCountLabel.textContent = `เลือกแล้ว ${selectedFiles.length} ภาพ (พร้อมจัดคิวคัดกรอง)`;
  thumbnailStrip.innerHTML = "";

  selectedFiles.forEach((file) => {
    const url = URL.createObjectURL(file);
    const div = document.createElement("div");
    div.className = "thumb-item";
    div.innerHTML = `<img src="${url}" alt="${file.name}" /><span>${file.name}</span>`;
    thumbnailStrip.appendChild(div);
  });
}

dropZone.addEventListener("click", () => {
  if (!btnSelectFiles.disabled) fileInput.click();
});

fileInput.addEventListener("change", (e) => {
  handleFiles(e.target.files);
  fileInput.value = "";
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (!btnSelectFiles.disabled) handleFiles(e.dataTransfer.files);
});

btnClearFiles.addEventListener("click", () => {
  selectedFiles = [];
  renderThumbnails();
});

// ==================== 3. IMAGE PREPROCESSING & ONNX INFERENCE ====================
function loadImageElement(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve({ img, url });
    img.onerror = reject;
    img.src = url;
  });
}

function preprocessImageToTensor(img, size) {
  const offCanvas = document.createElement("canvas");
  offCanvas.width = size;
  offCanvas.height = size;
  const ctx = offCanvas.getContext("2d");
  ctx.drawImage(img, 0, 0, size, size);

  const { data } = ctx.getImageData(0, 0, size, size);
  const floatData = new Float32Array(3 * size * size);
  const area = size * size;
  const [meanR, meanG, meanB] = appConfig.mean;
  const [stdR, stdG, stdB] = appConfig.std;

  // แปลง [H, W, RGBA] -> [1, 3, H, W] (CHW) พร้อม ImageNet Normalization
  for (let i = 0; i < area; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    floatData[i] = (r - meanR) / stdR;
    floatData[area + i] = (g - meanG) / stdG;
    floatData[2 * area + i] = (b - meanB) / stdB;
  }

  return new ort.Tensor("float32", floatData, [1, 3, size, size]);
}

function sigmoid(x) {
  return 1.0 / (1.0 + Math.exp(-x));
}

async function runBatchInference() {
  if (!ortSession || selectedFiles.length === 0) return;

  btnRunTriage.disabled = true;
  btnClearFiles.disabled = true;
  progressContainer.classList.remove("hidden");

  const results = [];
  const size = appConfig.img_size;
  const topK = appConfig.top_k;
  const threshold = appConfig.threshold;
  let totalInferenceMs = 0;

  for (let idx = 0; idx < selectedFiles.length; idx++) {
    const file = selectedFiles[idx];
    const pct = Math.round(((idx + 1) / selectedFiles.length) * 100);
    progressLabel.textContent = `กำลังวิเคราะห์ภาพ (${idx + 1}/${selectedFiles.length}): ${file.name}`;
    progressPercent.textContent = `${pct}%`;
    progressFill.style.width = `${pct}%`;

    // เปิดโอกาสให้ UI อัปเดต Progress Bar ระหว่างวนลูป
    await new Promise((r) => setTimeout(r, 15));

    const { img, url } = await loadImageElement(file);
    const inputTensor = preprocessImageToTensor(img, size);

    const t0 = performance.now();
    const outputs = await ortSession.run({ input_image: inputTensor });
    const dt = performance.now() - t0;
    totalInferenceMs += dt;

    const segLogits = outputs.seg_logits.data; // Float32Array [size * size]
    const probMap = new Float32Array(segLogits.length);
    let positivePixels = 0;

    for (let i = 0; i < segLogits.length; i++) {
      const p = sigmoid(segLogits[i]);
      probMap[i] = p;
      if (p > 0.5) positivePixels++;
    }

    // คำนวณ Top-K Pixel Mean เหมือนกับใน PyTorch (logits_to_image_prob)
    const sortedProbs = Float32Array.from(probMap).sort((a, b) => b - a);
    const k = Math.min(topK, sortedProbs.length);
    let topKSum = 0;
    for (let i = 0; i < k; i++) topKSum += sortedProbs[i];
    const riskScore = topKSum / k;

    const isUrgent = riskScore >= threshold;
    const areaPct = ((positivePixels / segLogits.length) * 100).toFixed(2);

    results.push({
      id: `CXR-${String(idx + 1).padStart(3, "0")}`,
      fileName: file.name,
      imageUrl: url,
      imgElement: img,
      probMap,
      size,
      riskScore,
      isUrgent,
      areaPct,
      latencyMs: Math.round(dt),
    });
  }

  // เรียงลำดับคิวจากความเสี่ยงสูงสุด -> ต่ำสุด
  results.sort((a, b) => b.riskScore - a.riskScore);
  triageResults = results;

  // อัปเดต Summary Cards
  const urgentCount = results.filter((r) => r.isUrgent).length;
  const normalCount = results.length - urgentCount;
  const avgLatency = Math.round(totalInferenceMs / results.length);

  document.getElementById("stat-total").textContent = results.length;
  document.getElementById("stat-urgent").textContent = urgentCount;
  document.getElementById("stat-normal").textContent = normalCount;
  document.getElementById("stat-latency").textContent = `${avgLatency} ms`;

  // สลับไปหน้า Queue Dashboard
  progressContainer.classList.add("hidden");
  btnRunTriage.disabled = false;
  btnClearFiles.disabled = false;

  viewUpload.classList.add("hidden");
  viewQueue.classList.remove("hidden");
  btnBackUpload.classList.remove("hidden");

  renderQueueTable();
}

btnRunTriage.addEventListener("click", runBatchInference);

btnBackUpload.addEventListener("click", () => {
  viewQueue.classList.add("hidden");
  btnBackUpload.classList.add("hidden");
  viewUpload.classList.remove("hidden");
});

// ==================== 4. RENDER QUEUE TABLE & FILTERS ====================
function renderQueueTable() {
  queueTbody.innerHTML = "";

  const filtered = triageResults.filter((item) => {
    if (currentFilter === "URGENT") return item.isUrgent;
    if (currentFilter === "NORMAL") return !item.isUrgent;
    return true;
  });

  filtered.forEach((item) => {
    const globalRank = triageResults.indexOf(item) + 1;
    const pct = (item.riskScore * 100).toFixed(1);
    const barColor = item.isUrgent ? "#dc2626" : "#16a34a";

    const tr = document.createElement("tr");
    if (item.isUrgent) tr.classList.add("row-urgent");

    tr.innerHTML = `
      <td><span class="rank-badge">#${globalRank}</span></td>
      <td><img src="${item.imageUrl}" class="table-thumb" alt="scan" /></td>
      <td>
        <strong>${item.id}</strong><br />
        <span style="color:#64748b; font-size:0.8rem;">${item.fileName}</span>
      </td>
      <td>
        ${
          item.isUrgent
            ? `<span class="status-pill status-urgent">🚨 URGENT (พบความเสี่ยง)</span>`
            : `<span class="status-pill status-normal">✓ NORMAL (ปกติ)</span>`
        }
      </td>
      <td>
        <div class="risk-bar-wrap">
          <span class="risk-score-text" style="color:${barColor}">${pct}%</span>
          <div class="risk-bar">
            <div class="risk-bar-fill" style="width:${Math.min(pct, 100)}%; background:${barColor}"></div>
          </div>
        </div>
      </td>
      <td>${item.areaPct}% ของภาพ</td>
      <td>
        <button class="btn btn-primary btn-sm btn-inspect" data-id="${item.id}">
          🔍 ดูผล Segmentation
        </button>
      </td>
    `;

    tr.querySelector(".btn-inspect").addEventListener("click", () => openInspectModal(item, globalRank));
    queueTbody.appendChild(tr);
  });
}

document.querySelectorAll(".pill").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".pill").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.dataset.filter;
    renderQueueTable();
  });
});

// ==================== 5. SEGMENTATION OVERLAY MODAL ====================
function drawOverlayCanvas(item, showMask) {
  const size = item.size;
  canvasOriginal.width = size;
  canvasOriginal.height = size;
  canvasOverlay.width = size;
  canvasOverlay.height = size;

  const ctxOrig = canvasOriginal.getContext("2d");
  const ctxOver = canvasOverlay.getContext("2d");

  ctxOrig.drawImage(item.imgElement, 0, 0, size, size);
  ctxOver.drawImage(item.imgElement, 0, 0, size, size);

  if (!showMask) return;

  const imgData = ctxOver.getImageData(0, 0, size, size);
  const pixels = imgData.data;
  const thresh = appConfig.threshold;

  const lowThresh = 0.25;

  for (let i = 0; i < item.probMap.length; i++) {
    const p = item.probMap[i];
    if (p > 0.5) {
      // ความมั่นใจสูง > 0.50 -> ไฮไลต์สีแดงสว่าง
      pixels[i * 4] = Math.min(255, pixels[i * 4] * 0.4 + 239 * 0.6);
      pixels[i * 4 + 1] = pixels[i * 4 + 1] * 0.4 + 68 * 0.6;
      pixels[i * 4 + 2] = pixels[i * 4 + 2] * 0.4 + 68 * 0.6;
    } else if (p >= lowThresh) {
      // จุดเฝ้าระวัง (>= 0.25) -> ไฮไลต์สีฟ้า Cyan
      pixels[i * 4] = pixels[i * 4] * 0.5 + 56 * 0.5;
      pixels[i * 4 + 1] = Math.min(255, pixels[i * 4 + 1] * 0.5 + 189 * 0.5);
      pixels[i * 4 + 2] = Math.min(255, pixels[i * 4 + 2] * 0.5 + 248 * 0.5);
    }
  }

  ctxOver.putImageData(imgData, 0, 0);
}

function openInspectModal(item, rank) {
  activeModalItem = item;
  const pct = (item.riskScore * 100).toFixed(2);
  modalTitle.textContent = `คิวที่ #${rank} — ${item.fileName}`;
  modalSubtitle.textContent = `สถานะ: ${item.isUrgent ? "URGENT (เร่งด่วน)" : "NORMAL"} | Risk Score: ${pct}% (Threshold: ${appConfig.threshold}) | Mask Area: ${item.areaPct}%`;

  toggleMask.checked = true;
  drawOverlayCanvas(item, true);
  inspectModal.classList.remove("hidden");
}

toggleMask.addEventListener("change", () => {
  if (activeModalItem) {
    drawOverlayCanvas(activeModalItem, toggleMask.checked);
  }
});

document.getElementById("modal-close-btn").addEventListener("click", () => {
  inspectModal.classList.add("hidden");
});
document.getElementById("modal-close-bg").addEventListener("click", () => {
  inspectModal.classList.add("hidden");
});

// เริ่มโหลดโมเดลทันทีที่เปิดหน้าเว็บ
initModel();