// TrafficVision - Frontend Controller
let selectedFile = null;

const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('fileInput');
const previewCard = document.getElementById('previewCard');
const imagePreview = document.getElementById('imagePreview');
const previewFilename = document.getElementById('previewFilename');
const btnRemovePreview = document.getElementById('btnRemovePreview');
const btnAnalyze = document.getElementById('btnAnalyze');

const processingSection = document.getElementById('processingSection');
const processingStatusText = document.getElementById('processingStatusText');
const errorToast = document.getElementById('errorToast');
const errorMessage = document.getElementById('errorMessage');

const emptyState = document.getElementById('emptyState');
const resultsDashboard = document.getElementById('resultsDashboard');
const latencyBadge = document.getElementById('latencyBadge');
const backendBadge = document.getElementById('backendBadge');

// Result fields
const densityBanner = document.getElementById('densityBanner');
const densityCategory = document.getElementById('densityCategory');
const densityExplanation = document.getElementById('densityExplanation');

const kpiTotal = document.getElementById('kpiTotal');
const kpiVehicles = document.getElementById('kpiVehicles');
const kpiPedestrians = document.getElementById('kpiPedestrians');

const countCar = document.getElementById('countCar');
const countBus = document.getElementById('countBus');
const countMotorcycle = document.getElementById('countMotorcycle');
const countPerson = document.getElementById('countPerson');

const barCar = document.getElementById('barCar');
const barBus = document.getElementById('barBus');
const barMoto = document.getElementById('barMoto');
const barPerson = document.getElementById('barPerson');

const barCarNum = document.getElementById('barCarNum');
const barBusNum = document.getElementById('barBusNum');
const barMotoNum = document.getElementById('barMotoNum');
const barPersonNum = document.getElementById('barPersonNum');

const annotatedImageView = document.getElementById('annotatedImageView');
const originalImageView = document.getElementById('originalImageView');

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    initDragAndDrop();
});

// Check API Health
async function checkHealth() {
    try {
        const res = await fetch('/health');
        if (res.ok) {
            const data = await res.json();
            backendBadge.className = 'status-pill status-ready';
            backendBadge.innerText = data.model_loaded ? 'Model Ready' : 'Degraded';
        } else {
            backendBadge.className = 'status-pill status-loading';
            backendBadge.innerText = 'Server Error';
        }
    } catch {
        backendBadge.className = 'status-pill status-loading';
        backendBadge.innerText = 'Offline';
    }
}

// Drag & Drop
function initDragAndDrop() {
    ['dragenter', 'dragover'].forEach(name => {
        dropArea.addEventListener(name, (e) => {
            e.preventDefault();
            dropArea.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropArea.addEventListener(name, (e) => {
            e.preventDefault();
            dropArea.classList.remove('dragover');
        });
    });

    dropArea.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFileSelection(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length > 0) handleFileSelection(fileInput.files[0]);
    });

    btnRemovePreview.addEventListener('click', clearFileSelection);
    btnAnalyze.addEventListener('click', analyzeTraffic);
}

function handleFileSelection(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file (JPG, PNG, WebP).');
        return;
    }

    selectedFile = file;
    hideError();

    previewFilename.innerText = file.name;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        originalImageView.src = e.target.result;
        previewCard.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function clearFileSelection() {
    selectedFile = null;
    fileInput.value = '';
    previewCard.style.display = 'none';
    hideError();
}

async function analyzeTraffic() {
    if (!selectedFile) return;

    hideError();
    processingSection.style.display = 'block';
    btnAnalyze.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Inference failure.');
        }

        const data = await response.json();
        renderResults(data);
    } catch (err) {
        showError(err.message);
    } finally {
        processingSection.style.display = 'none';
        btnAnalyze.disabled = false;
    }
}

function renderResults(data) {
    emptyState.style.display = 'none';
    resultsDashboard.style.display = 'block';

    // Latency badge
    latencyBadge.style.display = 'inline-block';
    latencyBadge.innerText = `${data.inference_ms}ms (inference) | ${data.total_processing_ms}ms (total)`;

    // Density banner
    const density = data.traffic_density.toLowerCase();
    densityBanner.className = `density-banner density-${density}`;
    densityCategory.innerText = data.traffic_density;
    densityExplanation.innerText = data.classification_details.explanation;

    // KPI Counters
    kpiTotal.innerText = data.total_objects;
    kpiVehicles.innerText = data.vehicle_count;
    kpiPedestrians.innerText = data.pedestrian_count;

    // Breakdown
    const c = data.counts;
    countCar.innerText = c.car;
    countBus.innerText = c.bus;
    countMotorcycle.innerText = c.motorcycle;
    countPerson.innerText = c.person;

    // Visual Bars
    const maxVal = Math.max(c.car, c.bus, c.motorcycle, c.person, 1);
    barCar.style.width = `${(c.car / maxVal) * 100}%`;
    barBus.style.width = `${(c.bus / maxVal) * 100}%`;
    barMoto.style.width = `${(c.motorcycle / maxVal) * 100}%`;
    barPerson.style.width = `${(c.person / maxVal) * 100}%`;

    barCarNum.innerText = c.car;
    barBusNum.innerText = c.bus;
    barMotoNum.innerText = c.motorcycle;
    barPersonNum.innerText = c.person;

    // Annotated Image View
    annotatedImageView.src = data.annotated_image_url + '?t=' + new Date().getTime();
}

function showError(msg) {
    errorMessage.innerText = msg;
    errorToast.style.display = 'flex';
}

function hideError() {
    errorToast.style.display = 'none';
}
