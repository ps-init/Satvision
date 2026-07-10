const uploadInput = document.getElementById("imageUpload");
const thermalPreview = document.getElementById("thermalPreview");
const rgbPreview = document.getElementById("rgbPreview");
const compareBefore = document.getElementById("compareBefore");
const compareAfter = document.getElementById("compareAfter");
const slider = document.getElementById("slider");
const overlay = document.getElementById("overlay");

// ==================== API CONFIG ====================
const API_BASE_URL = "http://localhost:8000";

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
            body: formData,
            headers: {
                // Don't set Content-Type, browser will set it with boundary
            }
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        
        updateStatus("✨ Processing complete!");
        displayResults(result);
        
    } catch (error) {
        console.error("Processing error:", error);
        updateStatus(`❌ Error: ${error.message}`);
        alert(`Error processing image: ${error.message}\n\nMake sure the API server is running on ${API_BASE_URL}`);
    }
}

// ==================== DISPLAY RESULTS ====================
function displayResults(result) {
    // Display RGB image
    if (result.annotated_image_base64) {
        const rgbImage = `data:image/png;base64,${result.annotated_image_base64}`;
        rgbPreview.src = rgbImage;
        compareAfter.src = rgbImage;
    }
    
    // Update detection counts
    const objectCount = result.object_count || {};
    
    // Update counts for each detected class
    document.getElementById("vehicleCount").textContent = objectCount["car"] || 
                                                         objectCount["vehicle"] || 
                                                         objectCount["Car"] || 0;
    
    document.getElementById("buildingCount").textContent = objectCount["building"] || 
                                                          objectCount["Building"] || 0;
    
    document.getElementById("roadCount").textContent = objectCount["road"] || 
                                                       objectCount["Road"] || 0;
    
    // Calculate average confidence
    if (result.detections && result.detections.length > 0) {
        const avgConfidence = (result.detections.reduce((sum, det) => sum + det.confidence, 0) / result.detections.length * 100).toFixed(1);
        document.getElementById("confidenceValue").textContent = avgConfidence + "%";
    }
    
    // Total objects
    document.getElementById("totalObjects") = result.total_objects || 0;
    
    // Log full detections for debugging
    console.log("Full detection results:", result);
}

// ==================== UI HELPERS ====================
function updateStatus(message) {
    const statusText = document.getElementById("statusText");
    if (statusText) {
        statusText.innerHTML = message;
    }
}

function showLoadingState() {
    // Disable upload button
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
    });
    
    uploadBox.addEventListener("dragleave", () => {
        uploadBox.style.backgroundColor = "transparent";
    });
    
    uploadBox.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadBox.style.backgroundColor = "transparent";
        
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

// ==================== SECONDARY BUTTON (LEARN MORE) ====================
const secondaryButton = document.querySelector(".buttons .secondary");
if (secondaryButton) {
    secondaryButton.addEventListener("click", () => {
        document.querySelector(".pipeline-section").scrollIntoView({ behavior: "smooth" });
    });
}

// ==================== CHECK API HEALTH ====================
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log("API Health:", data);
        
        if (data.status === "healthy" && data.models_loaded) {
            console.log("✅ API is ready!");
            return true;
        } else {
            console.warn("⚠️ API is running but models may not be loaded");
            return false;
        }
    } catch (error) {
        console.warn(`⚠️ API not available at ${API_BASE_URL}. Is it running?`);
        return false;
    }
}

// Check API on page load
window.addEventListener("load", () => {
    checkAPIHealth();
});
