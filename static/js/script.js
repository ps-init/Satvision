const uploadInput = document.getElementById("imageUpload");

const thermalPreview = document.getElementById("thermalPreview");

const rgbPreview = document.getElementById("rgbPreview");

const compareBefore = document.getElementById("compareBefore");

const compareAfter = document.getElementById("compareAfter");

const slider = document.getElementById("slider");

const overlay = document.getElementById("overlay");

// Upload Preview

uploadInput.addEventListener("change", function(){

    if(this.files.length > 0){

        document.querySelector(".upload-label h3").innerHTML =
            this.files[0].name;

        const reader = new FileReader();

        reader.onload = function(e){

            thermalPreview.src = e.target.result;

            compareBefore.src = e.target.result;

        }

        reader.readAsDataURL(this.files[0]);

    }

});

// Slider

slider.addEventListener("input", function(){

    overlay.style.width = this.value + "%";

});


// Dummy values for frontend testing

document.getElementById("vehicleCount").textContent = 12;

document.getElementById("buildingCount").textContent = 8;

document.getElementById("roadCount").textContent = 5;

document.getElementById("treeCount").textContent = 21;

document.getElementById("timeMetric").textContent = "0.82 s";

document.getElementById("ssimMetric").textContent = "0.87";

document.getElementById("psnrMetric").textContent = "24.5 dB";


const statusText = document.getElementById("statusText");

uploadInput.addEventListener("change", function(){

    statusText.innerHTML = "Uploading Image...";

    setTimeout(function(){

        statusText.innerHTML =
        "Colorizing Thermal Image...";

    },1200);

    setTimeout(function(){

        statusText.innerHTML =
        "Running YOLO Detection...";

    },2500);

    setTimeout(function(){

        statusText.innerHTML =
        "Calculating Performance Metrics...";

    },3800);

    setTimeout(function(){

        statusText.innerHTML =
        "Analysis Complete ✅";

    },5000);

});