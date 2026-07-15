const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const saveBtn = document.getElementById("saveBtn");
const paintBtn = document.getElementById("paintBtn");
const eraseBtn = document.getElementById("eraseBtn");
const polygonBtn = document.getElementById("polygonBtn");
const applyPolygonBtn = document.getElementById("applyPolygonBtn");
const clearPinsBtn = document.getElementById("clearPinsBtn");
const brushSize = document.getElementById("brushSize");
const brushValue = document.getElementById("brushValue");
const fileInfo = document.getElementById("fileInfo");

let imageList = [];
let currentIndex = 0;
let currentName = "";
let mode = "paint";
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let polygonMode = false;
let polygonPoints = [];
let hoverPoint = null;

let baseImage = new Image();
let maskCanvas = document.createElement("canvas");
let maskCtx = maskCanvas.getContext("2d");
let overlayCanvas = document.createElement("canvas");
let overlayCtx = overlayCanvas.getContext("2d");

function setButtonsState() {
  paintBtn.classList.toggle("active", mode === "paint");
  eraseBtn.classList.toggle("active", mode === "erase");
  polygonBtn.classList.toggle("active", polygonMode);
}

function updateInfo() {
  fileInfo.textContent = `${currentIndex + 1}/${imageList.length} - ${currentName}`;
  brushValue.textContent = brushSize.value;
}

function drawComposite() {
  if (!baseImage.width || !baseImage.height) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(baseImage, 0, 0);

  // Render strict red overlay using mask alpha
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  overlayCtx.fillStyle = "rgb(255, 0, 0)";
  overlayCtx.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  overlayCtx.globalCompositeOperation = "destination-in";
  overlayCtx.drawImage(maskCanvas, 0, 0);
  overlayCtx.globalCompositeOperation = "source-over";

  ctx.save();
  ctx.globalAlpha = 0.55;
  ctx.drawImage(overlayCanvas, 0, 0);
  ctx.restore();

  drawPolygonGuide();
}

function toDataUrlFromBase64Png(base64png) {
  return `data:image/png;base64,${base64png}`;
}

async function loadImagesMeta() {
  const res = await fetch("/api/images");
  const data = await res.json();
  imageList = data.images || [];
  currentIndex = data.last_index || 0;
}

async function loadImageAt(index) {
  if (imageList.length === 0) {
    fileInfo.textContent = "No images found in this folder.";
    return;
  }

  currentIndex = Math.max(0, Math.min(index, imageList.length - 1));
  const res = await fetch(`/api/image/${currentIndex}`);
  const data = await res.json();
  if (data.error) {
    alert(data.error);
    return;
  }

  currentName = data.name;
  const baseDataUrl = toDataUrlFromBase64Png(data.image_data);
  const maskDataUrl = toDataUrlFromBase64Png(data.mask_data);

  await new Promise((resolve) => {
    baseImage.onload = resolve;
    baseImage.src = baseDataUrl;
  });

  canvas.width = baseImage.width;
  canvas.height = baseImage.height;
  maskCanvas.width = baseImage.width;
  maskCanvas.height = baseImage.height;
  overlayCanvas.width = baseImage.width;
  overlayCanvas.height = baseImage.height;

  await new Promise((resolve) => {
    const maskImg = new Image();
    maskImg.onload = () => {
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      // Convert loaded grayscale mask into alpha-only binary mask
      const temp = document.createElement("canvas");
      temp.width = maskCanvas.width;
      temp.height = maskCanvas.height;
      const tctx = temp.getContext("2d");
      tctx.drawImage(maskImg, 0, 0);
      const imgData = tctx.getImageData(0, 0, temp.width, temp.height);
      const src = imgData.data;
      for (let i = 0; i < src.length; i += 4) {
        const v = src[i];
        const on = v > 20;
        src[i] = on ? 255 : 0;
        src[i + 1] = on ? 255 : 0;
        src[i + 2] = on ? 255 : 0;
        src[i + 3] = on ? 255 : 0;
      }
      tctx.putImageData(imgData, 0, 0);
      maskCtx.drawImage(temp, 0, 0);
      polygonPoints = [];
      hoverPoint = null;
      resolve();
    };
    maskImg.src = maskDataUrl;
  });

  updateInfo();
  drawComposite();
}

function canvasPoint(evt) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (evt.clientX - rect.left) * scaleX,
    y: (evt.clientY - rect.top) * scaleY,
  };
}

function drawLine(x1, y1, x2, y2) {
  maskCtx.save();
  maskCtx.lineWidth = Number(brushSize.value);
  maskCtx.lineCap = "round";
  maskCtx.lineJoin = "round";
  if (mode === "paint") {
    maskCtx.globalCompositeOperation = "source-over";
    maskCtx.strokeStyle = "rgba(255, 255, 255, 1)";
  } else {
    maskCtx.globalCompositeOperation = "destination-out";
    maskCtx.strokeStyle = "rgba(255, 255, 255, 1)";
  }
  maskCtx.beginPath();
  maskCtx.moveTo(x1, y1);
  maskCtx.lineTo(x2, y2);
  maskCtx.stroke();
  maskCtx.restore();
}

function fillPolygon(points) {
  if (points.length < 3) return;
  maskCtx.save();
  if (mode === "paint") {
    maskCtx.globalCompositeOperation = "source-over";
    maskCtx.fillStyle = "rgba(255,255,255,1)";
  } else {
    maskCtx.globalCompositeOperation = "destination-out";
    maskCtx.fillStyle = "rgba(255,255,255,1)";
  }
  maskCtx.beginPath();
  maskCtx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) {
    maskCtx.lineTo(points[i].x, points[i].y);
  }
  maskCtx.closePath();
  maskCtx.fill();
  maskCtx.restore();
}

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function drawPolygonGuide() {
  if (!polygonMode || polygonPoints.length === 0) return;

  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#5ec0ff";
  ctx.fillStyle = "#ffe600";

  ctx.beginPath();
  ctx.moveTo(polygonPoints[0].x, polygonPoints[0].y);
  for (let i = 1; i < polygonPoints.length; i += 1) {
    ctx.lineTo(polygonPoints[i].x, polygonPoints[i].y);
  }
  if (hoverPoint) {
    ctx.lineTo(hoverPoint.x, hoverPoint.y);
  }
  ctx.stroke();

  for (const p of polygonPoints) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // highlight first pin for easy closing
  const first = polygonPoints[0];
  ctx.beginPath();
  ctx.strokeStyle = "#00ff88";
  ctx.arc(first.x, first.y, 8, 0, Math.PI * 2);
  ctx.stroke();

  ctx.restore();
}

canvas.addEventListener("mousedown", (evt) => {
  const p = canvasPoint(evt);
  if (polygonMode) {
    if (polygonPoints.length >= 3 && dist(p, polygonPoints[0]) <= 10) {
      // close and fill polygon
      fillPolygon(polygonPoints);
      polygonPoints = [];
      hoverPoint = null;
    } else {
      polygonPoints.push(p);
    }
    drawComposite();
    return;
  }

  isDrawing = true;
  lastX = p.x;
  lastY = p.y;
  drawLine(lastX, lastY, lastX, lastY);
  drawComposite();
});

canvas.addEventListener("mousemove", (evt) => {
  const p = canvasPoint(evt);
  hoverPoint = p;
  if (!polygonMode) {
    if (!isDrawing) return;
    drawLine(lastX, lastY, p.x, p.y);
    lastX = p.x;
    lastY = p.y;
  }
  drawComposite();
});

["mouseup", "mouseleave"].forEach((eventName) => {
  canvas.addEventListener(eventName, () => {
    isDrawing = false;
  });
});

async function saveCurrent() {
  const temp = document.createElement("canvas");
  temp.width = maskCanvas.width;
  temp.height = maskCanvas.height;
  const tctx = temp.getContext("2d");
  tctx.clearRect(0, 0, temp.width, temp.height);
  tctx.drawImage(maskCanvas, 0, 0);

  // Force strict binary mask for backend save
  const imgData = tctx.getImageData(0, 0, temp.width, temp.height);
  const arr = imgData.data;
  for (let i = 0; i < arr.length; i += 4) {
    // Use only alpha channel from working mask canvas to avoid false full-image masks.
    const on = arr[i + 3] > 10;
    arr[i] = on ? 255 : 0;
    arr[i + 1] = on ? 255 : 0;
    arr[i + 2] = on ? 255 : 0;
    arr[i + 3] = 255;
  }
  tctx.putImageData(imgData, 0, 0);

  const maskDataUrl = temp.toDataURL("image/png");
  const maskBase64 = maskDataUrl.split(",")[1];

  const res = await fetch(`/api/save/${currentIndex}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mask_data: maskBase64 }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "Save failed");
    return;
  }
}

async function setState(index) {
  await fetch("/api/state", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ last_index: index }),
  });
}

async function moveTo(index) {
  await setState(index);
  await loadImageAt(index);
}

prevBtn.addEventListener("click", async () => {
  if (currentIndex > 0) {
    await moveTo(currentIndex - 1);
  }
});

nextBtn.addEventListener("click", async () => {
  if (currentIndex < imageList.length - 1) {
    await moveTo(currentIndex + 1);
  }
});

saveBtn.addEventListener("click", saveCurrent);
paintBtn.addEventListener("click", () => {
  mode = "paint";
  setButtonsState();
});
eraseBtn.addEventListener("click", () => {
  mode = "erase";
  setButtonsState();
});
polygonBtn.addEventListener("click", () => {
  polygonMode = !polygonMode;
  if (!polygonMode) {
    polygonPoints = [];
    hoverPoint = null;
  }
  setButtonsState();
  drawComposite();
});
applyPolygonBtn.addEventListener("click", () => {
  fillPolygon(polygonPoints);
  polygonPoints = [];
  hoverPoint = null;
  drawComposite();
});
clearPinsBtn.addEventListener("click", () => {
  polygonPoints = [];
  hoverPoint = null;
  drawComposite();
});

brushSize.addEventListener("input", () => {
  brushValue.textContent = brushSize.value;
});

document.addEventListener("keydown", async (evt) => {
  const key = evt.key.toLowerCase();
  if (key === "r") {
    mode = "paint";
    setButtonsState();
  } else if (key === "e") {
    mode = "erase";
    setButtonsState();
  } else if (key === "s") {
    evt.preventDefault();
    await saveCurrent();
  } else if (key === "p") {
    polygonMode = !polygonMode;
    if (!polygonMode) {
      polygonPoints = [];
      hoverPoint = null;
    }
    setButtonsState();
    drawComposite();
  } else if (key === "f") {
    fillPolygon(polygonPoints);
    polygonPoints = [];
    hoverPoint = null;
    drawComposite();
  } else if (key === "x") {
    polygonPoints = [];
    hoverPoint = null;
    drawComposite();
  } else if (key === "a") {
    if (currentIndex > 0) await moveTo(currentIndex - 1);
  } else if (key === "d") {
    if (currentIndex < imageList.length - 1) await moveTo(currentIndex + 1);
  } else if (key === "[") {
    brushSize.value = Math.max(2, Number(brushSize.value) - 2);
    brushValue.textContent = brushSize.value;
  } else if (key === "]") {
    brushSize.value = Math.min(120, Number(brushSize.value) + 2);
    brushValue.textContent = brushSize.value;
  }
});

async function init() {
  setButtonsState();
  await loadImagesMeta();
  await loadImageAt(currentIndex);
}

init();
