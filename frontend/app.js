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

const langSelect     = document.getElementById("lang-select");

// ──────── State & i18n ────────
let selectedFiles = [];
let hasDocuments  = false;
let currentLang   = "en";

const translations = {
  en: {
    logo: "BookChat",
    connecting: "Connecting…",
    online: "API online",
    offline: "API offline",
    btn_reset: "Reset Store",
    documents_title: "Documents",
    drop_files: "Drop files here or",
    browse: "browse",
    drop_hint: "PDF, TXT, MD — up to 500 MB each",
    btn_upload: "Upload & Ingest",
    chat_title: "Ask a Question",
    empty_chat: "Upload a document and start asking questions",
    input_placeholder: "Ask something about your documents…",
    modal_title: "Reset Vector Store",
    modal_desc: "This will permanently delete all ingested documents. Are you sure?",
    modal_cancel: "Cancel",
    modal_confirm: "Reset",
    uploading: "Uploading and ingesting…",
    resetting: "Resetting vector store…",
  },
  kn: {
    logo: "ಬುಕ್‍ಚಾಟ್",
    connecting: "ಸಂಪರ್ಕಿಸಲಾಗುತ್ತಿದೆ…",
    online: "API ಸಕ್ರಿಯವಾಗಿದೆ",
    offline: "API ನಿಷ್ಕ್ರಿಯವಾಗಿದೆ",
    btn_reset: "ಮರುಹೊಂದಿಸಿ",
    documents_title: "ದಾಖಲೆಗಳು",
    drop_files: "ಫೈಲ್‌ಗಳನ್ನು ಇಲ್ಲಿ ಹಾಕಿ ಅಥವಾ",
    browse: "ಹುಡುಕಿ",
    drop_hint: "PDF, TXT, MD — ತಲಾ 500 MB ವರೆಗೆ",
    btn_upload: "ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
    chat_title: "ಪ್ರಶ್ನೆ ಕೇಳಿ",
    empty_chat: "ದಾಖಲೆಯನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಮತ್ತು ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಿ",
    input_placeholder: "ನಿಮ್ಮ ದಾಖಲೆಗಳ ಬಗ್ಗೆ ಏನನ್ನಾದರೂ ಕೇಳಿ…",
    modal_title: "ವೆಕ್ಟರ್ ಸ್ಟೋರ್ ಮರುಹೊಂದಿಸಿ",
    modal_desc: "ಇದು ಎಲ್ಲಾ ದಾಖಲೆಗಳನ್ನು ಕಾಯಂ ಆಗಿ ಅಳಿಸುತ್ತದೆ. ನೀವು ಖಚಿತವಾಗಿದ್ದೀರಾ?",
    modal_cancel: "ರದ್ದುಮಾಡಿ",
    modal_confirm: "ಮರುಹೊಂದಿಸಿ",
    uploading: "ಅಪ್‌ಲೋಡ್ ಮಾಡಲಾಗುತ್ತಿದೆ…",
    resetting: "ಮರುಹೊಂದಿಸಲಾಗುತ್ತಿದೆ…",
  },
  pa: {
    logo: "ਬੁੱਕਚੈਟ",
    connecting: "ਕਨੈਕਟ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ…",
    online: "API ਔਨਲਾਈਨ ਹੈ",
    offline: "API ਔਫਲਾਈਨ ਹੈ",
    btn_reset: "ਸਟੋਰ ਰੀਸੈਟ ਕਰੋ",
    documents_title: "ਦਸਤਾਵੇਜ਼",
    drop_files: "ਇੱਥੇ ਫ਼ਾਈਲਾਂ ਸੁੱਟੋ ਜਾਂ",
    browse: "ਬ੍ਰਾਊਜ਼ ਕਰੋ",
    drop_hint: "PDF, TXT, MD — ਹਰੇਕ 500 MB ਤੱਕ",
    btn_upload: "ਅੱਪਲੋਡ ਅਤੇ ਇੰਜੈਸਟ ਕਰੋ",
    chat_title: "ਸਵਾਲ ਪੁੱਛੋ",
    empty_chat: "ਇੱਕ ਦਸਤਾਵੇਜ਼ ਅੱਪਲੋਡ ਕਰੋ ਅਤੇ ਸਵਾਲ ਪੁੱਛਣਾ ਸ਼ੁਰੂ ਕਰੋ",
    input_placeholder: "ਆਪਣੇ ਦਸਤਾਵੇਜ਼ਾਂ ਬਾਰੇ ਕੁਝ ਪੁੱਛੋ…",
    modal_title: "ਵੈਕਟਰ ਸਟੋਰ ਰੀਸੈਟ ਕਰੋ",
    modal_desc: "ਇਹ ਸਾਰੇ ਇੰਜੈਸਟ ਕੀਤੇ ਦਸਤਾਵੇਜ਼ਾਂ ਨੂੰ ਪੱਕੇ ਤੌਰ 'ਤੇ ਮਿਟਾ ਦੇਵੇਗਾ। ਕੀ ਤੁਸੀਂ ਯਕੀਨੀ ਹੋ?",
    modal_cancel: "ਰੱਦ ਕਰੋ",
    modal_confirm: "ਰੀਸੈਟ ਕਰੋ",
    uploading: "ਅੱਪਲੋਡ ਅਤੇ ਇੰਜੈਸਟ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ…",
    resetting: "ਵੈਕਟਰ ਸਟੋਰ ਰੀਸੈਟ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ…",
  }
};

function setLanguage(lang) {
  currentLang = lang;
  const t = translations[lang] || translations.en;

  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (t[key]) {
      el.textContent = t[key];
    }
  });

  queryInput.placeholder = t.input_placeholder;
}

langSelect.addEventListener("change", (e) => {
  setLanguage(e.target.value);
});

// ──────── Health check ────────
async function checkHealth() {
  const t = translations[currentLang] || translations.en;
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error();
    healthDot.className  = "health-dot ok";
    healthLabel.textContent = t.online;
  } catch {
    healthDot.className  = "health-dot fail";
    healthLabel.textContent = t.offline;
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
  const t = translations[currentLang] || translations.en;
  setStatus(t.uploading, "loading");

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
      body: JSON.stringify({ question, k: 4, lang: currentLang }),
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
  const t = translations[currentLang] || translations.en;
  setStatus(t.resetting, "loading");

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
