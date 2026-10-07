let currentView = 'lilt';
let zoomLevel = 1.1; // Default 110% zoom as requested ("Also lilt bit zoom in the page")
let pipelineData = null;
let selectedFile = null;

// DOM Elements
const modelSelect = document.getElementById('modelSelect');
const fileInput = document.getElementById('fileInput');
const fileNameDisplay = document.getElementById('fileNameDisplay');
const runBtn = document.getElementById('runBtn');

const btnOcr = document.getElementById('btnOcr');
const btnLilt = document.getElementById('btnLilt');

const imagePaneTitle = document.getElementById('imagePaneTitle');
const displayImage = document.getElementById('displayImage');
const imageContainer = document.getElementById('imageContainer');
const originalDocImage = document.getElementById('originalDocImage');
const originalImageContainer = document.getElementById('originalImageContainer');
const loadingOverlay = document.getElementById('loadingOverlay');

const zoomInBtn = document.getElementById('zoomInBtn');
const zoomOutBtn = document.getElementById('zoomOutBtn');
const zoomResetBtn = document.getElementById('zoomResetBtn');
const zoomLevelDisplay = document.getElementById('zoomLevelDisplay');

const ocrContent = document.getElementById('ocrContent');
const liltContent = document.getElementById('liltContent');

const ocrCountBadge = document.getElementById('ocrCountBadge');
const ocrSummaryLine = document.getElementById('ocrSummaryLine');
const ocrTableBody = document.getElementById('ocrTableBody');

const liltCountBadge = document.getElementById('liltCountBadge');
const serCount = document.getElementById('serCount');
const serList = document.getElementById('serList');
const reCount = document.getElementById('reCount');
const reList = document.getElementById('reList');

// Initialize
function init() {
  setupEventListeners();
  applyZoom();
}

function setupEventListeners() {
  // View Switcher Buttons (Core User Requirement)
  btnOcr.addEventListener('click', () => switchView('ocr'));
  btnLilt.addEventListener('click', () => switchView('lilt'));

  // Model Selection
  if (modelSelect) {
    modelSelect.addEventListener('change', () => {
      if (selectedFile) {
        runPipeline();
      }
    });
  }

  // Upload Custom File
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      selectedFile = e.target.files[0];
      fileNameDisplay.textContent = selectedFile.name;
      runPipeline();
    }
  });

  // Run Pipeline Button
  runBtn.addEventListener('click', () => {
    if (!selectedFile) {
      fileInput.click();
    } else {
      runPipeline();
    }
  });

  // Zoom Controls
  zoomInBtn.addEventListener('click', () => {
    if (zoomLevel < 2.2) {
      zoomLevel += 0.15;
      applyZoom();
    }
  });

  zoomOutBtn.addEventListener('click', () => {
    if (zoomLevel > 0.6) {
      zoomLevel -= 0.15;
      applyZoom();
    }
  });

  zoomResetBtn.addEventListener('click', () => {
    zoomLevel = 1.1; // Default 110%
    applyZoom();
  });
}

function applyZoom() {
  imageContainer.style.transform = `scale(${zoomLevel.toFixed(2)})`;
  if (originalImageContainer) {
    originalImageContainer.style.transform = `scale(${zoomLevel.toFixed(2)})`;
  }
  zoomLevelDisplay.textContent = `${Math.round(zoomLevel * 100)}%`;
}

// Switch between OCR Results and LiLT Results
function switchView(view) {
  currentView = view;
  btnOcr.classList.toggle('active', view === 'ocr');
  btnLilt.classList.toggle('active', view === 'lilt');

  if (!pipelineData) return;

  if (view === 'ocr') {
    // 1. LEFT SIDE: Image with bounding boxes which OCR fetched for us
    imagePaneTitle.textContent = 'OCR Detected Document (Word Boxes & Coordinates)';
    displayImage.src = pipelineData.visuals.ocr_overlay;

    // 2. RIGHT SIDE: OCR text and coordinates
    ocrContent.style.display = 'block';
    liltContent.style.display = 'none';

  } else {
    // 1. LEFT SIDE: Different image processed by LiLT Transformer (SER + RE)
    imagePaneTitle.textContent = 'LiLT Processed Document (SER Boxes & RE Relations)';
    displayImage.src = pipelineData.visuals.lilt_overlay;

    // 2. RIGHT SIDE: SEM (Semantic Entity Recognition) + RE (Relation Extraction) with scores
    ocrContent.style.display = 'none';
    liltContent.style.display = 'block';
  }
}

async function runPipeline() {
  if (!selectedFile) {
    fileInput.click();
    return;
  }

  setLoading(true);
  try {
    const formData = new FormData();
    formData.append('file', selectedFile);
    if (modelSelect) {
      formData.append('layout_model', modelSelect.value);
    }

    const res = await fetch('/api/process', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Pipeline execution failed.');
    }

    pipelineData = await res.json();

    // Set actual / original raw image below
    if (pipelineData.visuals && pipelineData.visuals.original && originalDocImage) {
      originalDocImage.src = pipelineData.visuals.original;
    }

    // 1. Render OCR data on the right
    renderOcrData(pipelineData.ocr_result, pipelineData.image_size);

    // 2. Render LiLT data on the right (SER + RE with scores)
    renderLiltData(pipelineData.transformer_result);

    // 3. Update view on left and right according to current toggle
    switchView(currentView);

  } catch (err) {
    alert('Error running pipeline: ' + err.message);
  } finally {
    setLoading(false);
  }
}

function renderOcrData(ocrResult, imgSize) {
  const words = ocrResult.words || [];
  const [imgW, imgH] = imgSize || [800, 1000];

  ocrCountBadge.textContent = `${words.length} words`;
  ocrSummaryLine.textContent = `Extracted ${words.length} words in ${ocrResult.processing_time_ms.toFixed(0)}ms:`;

  if (words.length === 0) {
    ocrTableBody.innerHTML = `<tr><td colspan="5" class="empty-state">No words detected by OCR.</td></tr>`;
    return;
  }

  ocrTableBody.innerHTML = words.map(w => {
    const confPct = Math.round(w.confidence * 100);
    const confClass = confPct >= 80 ? 'conf-high' : (confPct >= 50 ? 'conf-med' : 'conf-low');

    const normBox = [
      Math.round(1000 * w.bbox_pixels[0] / imgW),
      Math.round(1000 * w.bbox_pixels[1] / imgH),
      Math.round(1000 * w.bbox_pixels[2] / imgW),
      Math.round(1000 * w.bbox_pixels[3] / imgH)
    ];

    return `
      <tr>
        <td style="color:#64748b;">${w.id}</td>
        <td><strong>${escapeHtml(w.text)}</strong></td>
        <td><span class="conf-tag ${confClass}">${confPct}%</span></td>
        <td style="color:#334155;">[${w.bbox_pixels.join(', ')}]</td>
        <td style="color:#64748b;">[${normBox.join(', ')}]</td>
      </tr>
    `;
  }).join('');
}

function renderLiltData(transformerResult) {
  const entities = transformerResult.entities || [];
  const relations = transformerResult.relations || [];

  liltCountBadge.textContent = `${entities.length} entities | ${relations.length} relations`;

  // --- Render SEM / SER (Semantic Entity Recognition) ---
  serCount.textContent = `${entities.length} recognized`;

  if (entities.length === 0) {
    serList.innerHTML = `<div class="empty-state">No semantic entities identified.</div>`;
  } else {
    serList.innerHTML = entities.map(e => {
      const type = e.label_type;
      const score = e.ser_score !== undefined ? e.ser_score : (e.confidence * 100).toFixed(1);
      const pillClass = `pill-${type}` in { 'pill-HEADER':1, 'pill-QUESTION':1, 'pill-ANSWER':1, 'pill-NAME':1 } ? `pill-${type}` : 'pill-OTHER';

      return `
        <div class="ser-card">
          <div class="ser-top">
            <span class="ser-type-pill ${pillClass}">${type}</span>
            <span class="ser-score">LiLT SER Score: ${score}%</span>
          </div>
          <div class="ser-text">${escapeHtml(e.text)}</div>
          <div class="ser-coords">BBox: [${e.bbox_pixels.join(', ')}] | Norm: [${e.bbox_norm.join(', ')}]</div>
        </div>
      `;
    }).join('');
  }

  // --- Render RE (Relation Extraction) ---
  reCount.textContent = `${relations.length} linked`;

  if (relations.length === 0) {
    reList.innerHTML = `<div class="empty-state">No relations extracted for this document.</div>`;
  } else {
    reList.innerHTML = relations.map(r => {
      const score = r.re_score !== undefined ? r.re_score : (r.confidence * 100).toFixed(1);

      return `
        <div class="re-card">
          <div class="re-top">
            <span class="re-link-badge">LINK: ${r.relation_type}</span>
            <span class="re-score">LiLT RE Score: ${score}%</span>
          </div>
          <div class="re-flow">
            <span class="re-head">${escapeHtml(r.head_text)}</span>
            <span class="re-arrow">&rarr;</span>
            <span class="re-tail">${escapeHtml(r.tail_text)}</span>
          </div>
          <div class="re-coords">Head: [${r.head_bbox.join(', ')}] &rarr; Tail: [${r.tail_bbox.join(', ')}]</div>
        </div>
      `;
    }).join('');
  }
}

function setLoading(isLoading) {
  loadingOverlay.style.display = isLoading ? 'flex' : 'none';
  runBtn.disabled = isLoading;
}

function escapeHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

document.addEventListener('DOMContentLoaded', init);
