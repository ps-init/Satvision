const uploadInput = document.getElementById("imageUpload");
const thermalPreview = document.getElementById("thermalPreview");
const rgbPreview = document.getElementById("rgbPreview");
const compareBefore = document.getElementById("compareBefore");
const compareAfter = document.getElementById("compareAfter");
const slider = document.getElementById("slider");
const overlay = document.getElementById("overlay");

// ==================== API CONFIG ====================
const API_BASE_URL = "http://localhost:8000";
let processingStartTime = 0;

// ==================== UPLOAD AND PROCESS ====================
uploadInput.addEventListener("change", async function(){
    if(this.files.length > 0){
        const file = this.files[0];
        
        // Update UI with filename
        document.querySelector(".upload-label h3").innerHTML = file.name;
        
        // Show thermal preview from local file
        const reader = new FileReader();
        reader.onload = function(e){
            thermalPreview.src = e.target.result;
            compareBefore.src = e.target.result;
        }
        reader.readAsDataURL(file);
        
        // Show loading state
        showLoadingState();
        
        // Send to API
        processingStartTime = Date.now();
        await processImage(file);
    }
});

// ==================== PROCESS IMAGE WITH API ====================
async function processImage(file) {
    try {
        const formData = new FormData();
        formData.append("file", file);
        
        updateStatus("🚀 Uploading thermal image...");
        
        const response = await fetch(`${API_BASE_URL}/detect`, {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        const processingTime = ((Date.now() - processingStartTime) / 1000).toFixed(2);
        
        updateStatus("✨ Processing complete!");
        displayResults(result, processingTime);
        hideLoadingState();
        
    } catch (error) {
        console.error("Processing error:", error);
        updateStatus(`❌ Error: ${error.message}`);
        hideLoadingState();
        alert(`Error processing image: ${error.message}\n\nMake sure the API server is running on ${API_BASE_URL}`);
    }
}

// ==================== DISPLAY RESULTS ====================
function displayResults(result, processingTime) {
    // Display RGB image with detections
    if (result.annotated_image_base64) {
        const rgbImage = `data:image/png;base64,${result.annotated_image_base64}`;
        rgbPreview.src = rgbImage;
        compareAfter.src = rgbImage;
    }
    
    // Update detection counts
    const objectCount = result.object_count || {};
    
    // Count vehicles (cars, trucks, buses, motorcycles, bicycles)
    const vehicleClasses = ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"];
    let vehicleCount = 0;
    for (const [cls, count] of Object.entries(objectCount)) {
        if (vehicleClasses.includes(cls.toLowerCase())) {
            vehicleCount += count;
        }
    }
    document.getElementById("vehicleCount").textContent = vehicleCount;
    
    // Count persons
    const personCount = objectCount["person"] || objectCount["Person"] || 0;
    document.getElementById("personCount").textContent = personCount;
    
    // Count traffic lights
    const trafficLightCount = objectCount["traffic light"] || objectCount["Traffic Light"] || 0;
    document.getElementById("trafficLightCount").textContent = trafficLightCount;
    
    // Calculate average confidence
    let avgConfidence = 0;
    if (result.detections && result.detections.length > 0) {
        avgConfidence = (result.detections.reduce((sum, det) => sum + det.confidence, 0) / result.detections.length * 100).toFixed(1);
    }
    document.getElementById("confidenceValue").textContent = avgConfidence + "%";
    
    // Update metrics
    document.getElementById("totalObjectsMetric").textContent = result.total_objects || 0;
    document.getElementById("processingTime").textContent = processingTime + "s";
    
    // Log full detections for debugging
    console.log("Full detection results:", result);
    console.log("Processing time:", processingTime + "s");
}

// ==================== UI HELPERS ====================
function updateStatus(message) {
    const statusContainer = document.getElementById("statusContainer");
    const statusText = document.getElementById("statusText");
    
    if (statusContainer && statusText) {
        statusContainer.style.display = "block";
        statusText.innerHTML = message;
    }
}

function showLoadingState() {
    const uploadLabel = document.querySelector(".upload-label");
    if (uploadLabel) {
        uploadLabel.style.opacity = "0.6";
        uploadLabel.style.pointerEvents = "none";
    }
}

function hideLoadingState() {
    const uploadLabel = document.querySelector(".upload-label");
    if (uploadLabel) {
        uploadLabel.style.opacity = "1";
        uploadLabel.style.pointerEvents = "auto";
    }
}

// ==================== IMAGE COMPARISON SLIDER ====================
slider.addEventListener("input", function(){
    overlay.style.width = this.value + "%";
});

// ==================== DRAG AND DROP ====================
const uploadBox = document.querySelector(".upload-box");

if (uploadBox) {
    uploadBox.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadBox.style.backgroundColor = "rgba(100, 200, 255, 0.1)";
        uploadBox.style.borderColor = "#4CAF50";
    });
    
    uploadBox.addEventListener("dragleave", () => {
        uploadBox.style.backgroundColor = "transparent";
        uploadBox.style.borderColor = "#ddd";
    });
    
    uploadBox.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadBox.style.backgroundColor = "transparent";
        uploadBox.style.borderColor = "#ddd";
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadInput.files = files;
            const event = new Event("change", { bubbles: true });
            uploadInput.dispatchEvent(event);
        }
    });
}

// ==================== PRIMARY BUTTON ACTION ====================
const primaryButton = document.querySelector(".buttons .primary");
if (primaryButton) {
    primaryButton.addEventListener("click", () => {
        uploadInput.click();
    });
}

// Alternative: Hero button
const uploadBtnHero = document.getElementById("uploadBtnHero");
if (uploadBtnHero) {
    uploadBtnHero.addEventListener("click", () => {
        uploadInput.click();
    });
}

// ==================== SECONDARY BUTTON (LEARN MORE) ====================
const secondaryButton = document.querySelector(".buttons .secondary");
if (secondaryButton) {
    secondaryButton.addEventListener("click", () => {
        const pipelineSection = document.querySelector(".pipeline-section");
        if (pipelineSection) {
            pipelineSection.scrollIntoView({ behavior: "smooth" });
        }
    });
}

const learnMoreBtn = document.getElementById("learnMoreBtn");
if (learnMoreBtn) {
    learnMoreBtn.addEventListener("click", () => {
        const pipelineSection = document.querySelector(".pipeline-section");
        if (pipelineSection) {
            pipelineSection.scrollIntoView({ behavior: "smooth" });
        }
    });
}

// ==================== CHECK API HEALTH ====================
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === "healthy" && data.models_loaded) {
            console.log("✅ API is ready and models are loaded!");
            updateAPIStatus(true, "Ready");
            return true;
        } else {
            console.warn("⚠️ API is running but models may not be loaded");
            updateAPIStatus(false, "Models Loading...");
            return false;
        }
    } catch (error) {
        console.warn(`⚠️ API not available at ${API_BASE_URL}. Is it running?`);
        updateAPIStatus(false, "Offline");
        return false;
    }
}

function updateAPIStatus(isHealthy, statusText) {
    const apiStatusEl = document.getElementById("apiStatus");
    const apiStatusTextEl = document.getElementById("apiStatusText");
    
    if (apiStatusEl) {
        apiStatusEl.textContent = isHealthy ? "✅" : "⚠️";
        apiStatusEl.style.color = isHealthy ? "#4CAF50" : "#FF9800";
    }
    
    if (apiStatusTextEl) {
        apiStatusTextEl.textContent = statusText;
    }
}

// Check API on page load
window.addEventListener("load", () => {
    checkAPIHealth();
    
    // Re-check every 10 seconds
    setInterval(checkAPIHealth, 10000);
});

// ==================== SMOOTH SCROLLING FOR NAV LINKS ====================
document.querySelectorAll("nav a").forEach(link => {
    link.addEventListener("click", (e) => {
        const href = link.getAttribute("href");
        if (href && href !== "#" && !href.startsWith("http")) {
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({ behavior: "smooth" });
            }
        }
    });
});
