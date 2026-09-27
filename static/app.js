const input = document.getElementById('image-input');
const analyzeBtn = document.getElementById('analyze');
const errorBox = document.getElementById('error');
const preview = document.getElementById('preview');
const resultBody = document.getElementById('result-body');
const emptyState = document.getElementById('empty-state');
const fileName = document.getElementById('file-name');
const heatmap = document.getElementById('heatmap');
const priorityTag = document.getElementById('priority-tag');
const cameraBtn = document.getElementById('camera-btn');
const uploadAction = document.getElementById('upload-action');
const modeInstruction = document.getElementById('mode-instruction');
const cameraStage = document.getElementById('camera-stage');
const cameraVideo = document.getElementById('camera-video');
const captureBtn = document.getElementById('capture-btn');
const stopCameraBtn = document.getElementById('stop-camera-btn');
const dropzone = document.getElementById('dropzone');

let selectedFile = null;
let selectedMode = 'fundus';
let cameraStream = null;

function setCaptureModeFromSelection() {
  const mode = document.querySelector('input[name="capture_mode"]:checked')?.value || 'fundus';
  selectedMode = mode;
}

function updateModeInterface() {
  const selfie = selectedMode === 'selfie';
  uploadAction.classList.toggle('hidden', selfie);
  cameraBtn.classList.toggle('hidden', !selfie);
  modeInstruction.textContent = selfie
    ? 'Open the camera and centre one eye inside the frame before capturing.'
    : 'Upload a clear retinal photograph to begin the quality gate.';
  dropzone.classList.toggle('selfie-mode', selfie);
  dropzone.classList.toggle('hidden', selfie && !selectedFile);
}

document.querySelectorAll('input[name="capture_mode"]').forEach((radio) => {
  radio.addEventListener('change', () => {
    setCaptureModeFromSelection();
    selectedFile = null;
    preview.classList.remove('visible');
    fileName.textContent = 'No image selected';
    analyzeBtn.disabled = true;
    errorBox.textContent = '';
    updateModeInterface();
    if (cameraStream) {
      stopCamera();
      startCamera();
    }
  });
});

updateModeInterface();

function updatePriorityTag(flag, action) {
  if (flag) {
    priorityTag.textContent = 'Review required';
    priorityTag.className = 'priority-tag review';
    return;
  }

  if (action && action.toLowerCase().includes('urgent')) {
    priorityTag.textContent = 'Urgent review';
    priorityTag.className = 'priority-tag alert';
    return;
  }

  priorityTag.textContent = 'Stable screening';
  priorityTag.className = 'priority-tag ok';
}

function renderProbabilities(probabilities) {
  const container = document.getElementById('probabilities');
  container.innerHTML = probabilities
    .map((item) => {
      const pct = Math.round(item.value * 100);
      return `
        <div class="probability-row">
          <span>${item.label}</span>
          <div class="bar"><span style="width:${pct}%"></span></div>
          <strong>${pct}%</strong>
        </div>
      `;
    })
    .join('');
}

function renderExplanation(textList) {
  const list = document.getElementById('explanation-list');
  list.innerHTML = textList.map((item) => `<li>${item}</li>`).join('');
}

function updateHumanReading(result) {
  const grade = result.grade || 'the selected grade';
  const quality = result.quality?.label || 'unknown quality';
  const confidencePct = Math.round((result.confidence || 0) * 100);
  document.getElementById('human-summary').textContent = `${result.category} at ${confidencePct}% confidence.`;
  document.getElementById('human-detail').textContent = `The model selected ${grade}. The image was judged as ${quality.toLowerCase()}, so this result should be read as screening support rather than a final diagnosis.`;
  document.getElementById('confidence-copy').textContent = `The model preferred ${grade} over the other grades with ${confidencePct}% confidence. A lower value means the case deserves closer human review.`;
  document.getElementById('quality-copy').textContent = `${quality} means the image has limited or acceptable visual detail for this screening estimate. If quality is poor, capture another image before relying on the result.`;
  document.getElementById('next-step-copy').textContent = `${result.action} A clinician should make the final decision, especially when the case is flagged for review.`;
}

function renderResult(result) {
  emptyState.style.display = 'none';
  resultBody.classList.remove('hidden');

  const confidencePct = Math.round(result.confidence * 100);
  const qualityPct = Math.round((result.quality.score || 0) * 100);

  document.getElementById('category').textContent = result.category;
  document.getElementById('grade-box').textContent = result.grade;
  document.getElementById('confidence').textContent = `${confidencePct}%`;
  document.getElementById('confidence-meter').style.width = `${confidencePct}%`;
  document.getElementById('quality-label').textContent = result.quality.label;
  document.getElementById('quality-meter').style.width = `${qualityPct}%`;
  document.getElementById('action-box').textContent = result.action;
  renderProbabilities(result.probabilities);
  renderExplanation(result.explanation);
  updateHumanReading(result);
  updatePriorityTag(result.review_required, result.action);

  if (result.heatmap) {
    heatmap.src = result.heatmap;
    heatmap.classList.add('visible');
  }
}

function setSelectedFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    errorBox.textContent = 'Please select a valid image file.';
    return;
  }

  selectedFile = file;
  dropzone.classList.remove('hidden');
  fileName.textContent = file.name;
  preview.src = URL.createObjectURL(file);
  preview.classList.add('visible');
  analyzeBtn.disabled = false;
  errorBox.textContent = '';
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    errorBox.textContent = 'This browser does not support camera capture. Please upload an image instead.';
    return;
  }

  try {
    setCaptureModeFromSelection();
    cameraStage.classList.remove('hidden');
    const constraints = {
      video: {
        facingMode: selectedMode === 'selfie' ? 'user' : 'environment',
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };
    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    cameraVideo.srcObject = cameraStream;
    cameraVideo.play();
    errorBox.textContent = '';
  } catch (error) {
    cameraStage.classList.add('hidden');
    errorBox.textContent = 'Camera access was blocked or unavailable. Please choose an image from your device instead.';
  }
}

function stopCamera() {
  cameraStage.classList.add('hidden');
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  if (cameraVideo) {
    cameraVideo.srcObject = null;
  }
}

function captureFrame() {
  const canvas = document.getElementById('camera-canvas');
  const video = cameraVideo;
  const width = video.videoWidth || 1280;
  const height = video.videoHeight || 720;

  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, width, height);

  canvas.toBlob((blob) => {
    if (!blob) {
      errorBox.textContent = 'Unable to capture image from camera. Please try again.';
      return;
    }

    const fileName = selectedMode === 'selfie' ? 'selfie-capture.png' : 'fundus-capture.png';
    const file = new File([blob], fileName, { type: 'image/png' });
    setSelectedFile(file);
    stopCamera();
  }, 'image/png', 0.95);
}

input.addEventListener('change', function () {
  setSelectedFile(input.files[0]);
});

cameraBtn.addEventListener('click', startCamera);
stopCameraBtn.addEventListener('click', stopCamera);
captureBtn.addEventListener('click', captureFrame);

['dragenter', 'dragover'].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', (event) => {
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  setSelectedFile(file);
});

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    errorBox.textContent = 'Choose an image first.';
    return;
  }

  setCaptureModeFromSelection();
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing...';
  errorBox.textContent = '';

  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('capture_mode', selectedMode);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }

    renderResult(data);
  } catch (error) {
    errorBox.textContent = error.message || 'Something went wrong while processing the image.';
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze image';
  }
});

document.querySelectorAll('[data-explain-tab]').forEach((tab) => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.explainTab;
    document.querySelectorAll('[data-explain-tab]').forEach((item) => item.classList.toggle('active', item === tab));
    document.querySelectorAll('[data-explain-view]').forEach((view) => view.classList.toggle('active', view.dataset.explainView === target));
  });
});
