// ──────── Configuration ────────
const API_BASE = "http://localhost:8000";

// ──────── DOM references ────────
const dropZone        = document.getElementById("drop-zone");
const fileInput       = document.getElementById("file-input");
const fileList        = document.getElementById("file-list");
const btnUpload       = document.getElementById("btn-upload");
const uploadStatus    = document.getElementById("upload-status");
const chatMessages    = document.getElementById("chat-messages");
const queryForm       = document.getElementById("query-form");
const queryInput      = document.getElementById("query-input");
const btnReset        = document.getElementById("btn-reset");
const healthDot       = document.getElementById("health-indicator");
const healthLabel     = document.getElementById("health-label");
const modalOverlay    = document.getElementById("modal-overlay");
const modalCancel     = document.getElementById("modal-cancel");
const modalConfirm    = document.getElementById("modal-confirm");

// ──────── State ────────
let selectedFiles = [];
let hasDocuments  = false;          // tracks if anything has been ingested

// ──────── Health check ────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error();
    healthDot.className  = "health-dot ok";
    healthLabel.textContent = "API online";
  } catch {
    healthDot.className  = "health-dot fail";
    healthLabel.textContent = "API offline";
  }
}

checkHealth();
setInterval(checkHealth, 30_000);

// ──────── File selection ────────
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  addFiles(e.dataTransfer.files);
});

fileInput.addEventListener("change", () => {
  addFiles(fileInput.files);
  fileInput.value = "";            // allow re-selecting same files
});

function addFiles(fileListObj) {
  const ALLOWED = new Set([".pdf", ".txt", ".md"]);
  for (const file of fileListObj) {
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED.has(ext)) {
      setStatus(`Skipped "${file.name}" — unsupported type.`, "error");
      continue;
    }
    // avoid duplicate names
    if (selectedFiles.some((f) => f.name === file.name && f.size === file.size)) continue;
    selectedFiles.push(file);
  }
  renderFileList();
}

function renderFileList() {
  fileList.innerHTML = "";
  selectedFiles.forEach((file, idx) => {
    const item = document.createElement("div");
    item.className = "file-item";
    item.innerHTML = `
      <span class="file-item-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
      <span class="file-item-size">${formatBytes(file.size)}</span>
      <button class="file-item-remove" data-idx="${idx}" title="Remove">&times;</button>
    `;
    fileList.appendChild(item);
  });

  // remove-button listeners
  fileList.querySelectorAll(".file-item-remove").forEach((btn) => {
    btn.addEventListener("click", () => {
      selectedFiles.splice(Number(btn.dataset.idx), 1);
      renderFileList();
    });
  });

  btnUpload.disabled = selectedFiles.length === 0;
}

// ──────── Upload / Ingest ────────
btnUpload.addEventListener("click", async () => {
  if (selectedFiles.length === 0) return;

  btnUpload.classList.add("loading");
  btnUpload.disabled = true;
  setStatus("Uploading and ingesting…", "loading");

  const form = new FormData();
  selectedFiles.forEach((f) => form.append("files", f));

  try {
    const res = await fetch(`${API_BASE}/ingest`, { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Ingestion failed.");
    }

    setStatus(data.message, "success");
    selectedFiles = [];
    renderFileList();
    hasDocuments = true;
    clearEmptyState();
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    btnUpload.classList.remove("loading");
    btnUpload.disabled = selectedFiles.length === 0;
  }
});

// ──────── Query / Chat ────────
queryForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = queryInput.value.trim();
  if (!question) return;

  clearEmptyState();
  appendBubble(question, "user");
  queryInput.value = "";

  const typingEl = showTyping();

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k: 4 }),
    });

    const data = await res.json();
    removeTyping(typingEl);

    if (!res.ok) {
      throw new Error(data.detail || "Query failed.");
    }

    appendBubble(data.answer, "assistant");
  } catch (err) {
    removeTyping(typingEl);
    appendBubble(err.message, "error");
  }
});

// ──────── Reset store ────────
btnReset.addEventListener("click", () => {
  modalOverlay.hidden = false;
});

modalCancel.addEventListener("click", () => {
  modalOverlay.hidden = true;
});

modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) modalOverlay.hidden = true;
});

modalConfirm.addEventListener("click", async () => {
  modalOverlay.hidden = true;
  setStatus("Resetting vector store…", "loading");

  try {
    const res = await fetch(`${API_BASE}/reset`, { method: "DELETE" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Reset failed.");

    setStatus(data.message, "success");
    hasDocuments = false;
    chatMessages.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">💬</span>
        <p>Upload a document and start asking questions</p>
      </div>`;
  } catch (err) {
    setStatus(err.message, "error");
  }
});

// ──────── Helpers ────────
function setStatus(msg, type) {
  uploadStatus.textContent = msg;
  uploadStatus.className = `status-message ${type}`;
}

function appendBubble(text, type) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${type}`;
  bubble.textContent = text;
  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTyping() {
  const el = document.createElement("div");
  el.className = "typing-indicator";
  el.innerHTML = "<span></span><span></span><span></span>";
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return el;
}

function removeTyping(el) {
  if (el && el.parentNode) el.parentNode.removeChild(el);
}

function clearEmptyState() {
  const empty = chatMessages.querySelector(".empty-state");
  if (empty) empty.remove();
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeHtml(str) {
  const el = document.createElement("span");
  el.textContent = str;
  return el.innerHTML;
}
