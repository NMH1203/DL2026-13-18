/**
 * Project 18: Interactive Web Demo Application Logic (app.js)
 * Manages reactive UI, API communication, Weight Inspection, and Inference Pipeline.
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
    let selectedSamplePath = null;
    let uploadedFile = null;
    let weightsData = null;
    let chartMap = null;
    let chartLoss = null;

    // Elements
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const samplesContainer = document.getElementById("samplesContainer");
    const confSlider = document.getElementById("confSlider");
    const confVal = document.getElementById("confVal");
    const iouSlider = document.getElementById("iouSlider");
    const iouVal = document.getElementById("iouVal");
    const viewMode = document.getElementById("viewMode");
    const btnRunPipeline = document.getElementById("btnRunPipeline");

    // Output Elements
    const emptyState = document.getElementById("emptyState");
    const grid4Way = document.getElementById("grid4Way");
    const grid2Way = document.getElementById("grid2Way");
    const detectionsCard = document.getElementById("detectionsCard");
    const detectionsTbody = document.getElementById("detectionsTbody");

    // Stats Elements
    const statDceTime = document.getElementById("statDceTime");
    const statYoloTime = document.getElementById("statYoloTime");
    const statTotalTime = document.getElementById("statTotalTime");
    const statFps = document.getElementById("statFps");
    const statObjectsCount = document.getElementById("statObjectsCount");

    // Image Elements
    const imgRawDark = document.getElementById("imgRawDark");
    const imgClahe = document.getElementById("imgClahe");
    const imgZeroDCE = document.getElementById("imgZeroDCE");
    const imgYoloRetrained = document.getElementById("imgYoloRetrained");
    const imgDetDark = document.getElementById("imgDetDark");
    const imgDetRetrained2 = document.getElementById("imgDetRetrained2");

    // IQA Badges
    const iqaRaw = document.getElementById("iqaRaw");
    const iqaClahe = document.getElementById("iqaClahe");
    const iqaDce = document.getElementById("iqaDce");
    const yoloObjCount = document.getElementById("yoloObjCount");
    const countDarkDet = document.getElementById("countDarkDet");
    const countRetrainedDet = document.getElementById("countRetrainedDet");

    // =========================================================================
    // 1. TAB NAVIGATION
    // =========================================================================
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add("active");

            // Re-render charts if benchmark tab selected
            if (targetId === "tab-benchmark" && weightsData) {
                renderCharts(weightsData);
            }
        });
    });

    // =========================================================================
    // 2. SLIDERS & CONTROLS
    // =========================================================================
    confSlider.addEventListener("input", (e) => {
        confVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    iouSlider.addEventListener("input", (e) => {
        iouVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // =========================================================================
    // 3. DROPZONE & SAMPLE IMAGES
    // =========================================================================
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    function handleFileUpload(file) {
        uploadedFile = file;
        selectedSamplePath = null;
        document.querySelectorAll(".sample-thumbnail").forEach(t => t.classList.remove("active"));
        
        dropzone.querySelector(".dropzone-text").innerHTML = `Đã chọn file: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB)`;
        
        // Auto trigger pipeline
        runInference();
    }

    // Fetch and render sample images
    async function loadSampleImages() {
        try {
            const res = await fetch("/api/sample_images");
            const samples = await res.json();
            samplesContainer.innerHTML = "";

            samples.forEach((sample, idx) => {
                const img = document.createElement("img");
                img.src = `/api/image_raw?path=${encodeURIComponent(sample.rel_path)}`;
                img.className = "sample-thumbnail";
                img.title = `${sample.filename} (${sample.size_kb} KB)`;
                
                img.addEventListener("click", () => {
                    document.querySelectorAll(".sample-thumbnail").forEach(t => t.classList.remove("active"));
                    img.classList.add("active");
                    selectedSamplePath = sample.rel_path;
                    uploadedFile = null;
                    dropzone.querySelector(".dropzone-text").innerHTML = `Đã chọn mẫu: <strong>${sample.filename}</strong>`;
                    runInference();
                });

                samplesContainer.appendChild(img);

                // Auto-select first sample initially
                if (idx === 0) {
                    img.click();
                }
            });
        } catch (err) {
            console.error("Lỗi khi tải ảnh mẫu:", err);
            samplesContainer.innerHTML = "<div class='text-muted'>Không thể nạp ảnh mẫu</div>";
        }
    }

    // =========================================================================
    // 4. INFERENCE PIPELINE EXECUTION
    // =========================================================================
    btnRunPipeline.addEventListener("click", runInference);

    async function runInference() {
        if (!selectedSamplePath && !uploadedFile) {
            alert("Vui lòng chọn 1 ảnh mẫu hoặc kéo thả tải ảnh lên!");
            return;
        }

        btnRunPipeline.disabled = true;
        btnRunPipeline.innerHTML = '<span class="btn-icon">⏳</span> ĐANG XỬ LÝ PIPELINE...';

        const formData = new FormData();
        formData.append("conf", confSlider.value);
        formData.append("iou", iouSlider.value);
        formData.append("model_choice", "both");

        if (uploadedFile) {
            formData.append("file", uploadedFile);
        } else if (selectedSamplePath) {
            formData.append("sample_path", selectedSamplePath);
        }

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            if (!res.ok) throw new Error("Inference API error");

            const data = await res.json();
            renderPredictionResults(data);
        } catch (err) {
            console.error("Lỗi suy luận:", err);
            alert("Đã xảy ra lỗi khi thực thi mô hình!");
        } finally {
            btnRunPipeline.disabled = false;
            btnRunPipeline.innerHTML = '<span class="btn-icon">⚡</span> CHẠY TOÀN BỘ PIPELINE';
        }
    }

    function renderPredictionResults(data) {
        emptyState.classList.add("d-none");
        
        // Update stats
        statDceTime.textContent = `${data.timing.zerodce_ms} ms`;
        statYoloTime.textContent = `${data.timing.yolo_retrained_ms} ms`;
        statTotalTime.textContent = `${data.timing.total_ms} ms`;
        statFps.textContent = `${data.timing.fps} FPS`;
        statObjectsCount.textContent = data.detections_count.scenario4_retrained;

        // Update image sources
        imgRawDark.src = data.images_base64.raw_dark;
        imgClahe.src = data.images_base64.clahe;
        imgZeroDCE.src = data.images_base64.zerodce;
        imgYoloRetrained.src = data.images_base64.scenario4_det;

        imgDetDark.src = data.images_base64.scenario1_det;
        imgDetRetrained2.src = data.images_base64.scenario4_det;

        // Quality badges
        iqaRaw.textContent = `NIQE: ${data.quality_metrics.raw.niqe} | BRISQUE: ${data.quality_metrics.raw.brisque}`;
        iqaClahe.textContent = `NIQE: ${data.quality_metrics.clahe.niqe} | BRISQUE: ${data.quality_metrics.clahe.brisque}`;
        iqaDce.textContent = `NIQE: ${data.quality_metrics.zerodce.niqe} | BRISQUE: ${data.quality_metrics.zerodce.brisque}`;
        yoloObjCount.textContent = `${data.detections_count.scenario4_retrained} vật thể`;

        countDarkDet.textContent = `${data.detections_count.scenario1_dark} vật thể`;
        countRetrainedDet.textContent = `${data.detections_count.scenario4_retrained} vật thể`;

        // View Mode toggle
        const mode = viewMode.value;
        if (mode === "four_way") {
            grid4Way.classList.remove("d-none");
            grid2Way.classList.add("d-none");
        } else if (mode === "compare_dark_vs_zerodce") {
            grid4Way.classList.add("d-none");
            grid2Way.classList.remove("d-none");
        } else {
            grid4Way.classList.remove("d-none");
            grid2Way.classList.add("d-none");
        }

        // Render Detections Table
        renderDetectionsTable(data.detections_s4);
    }

    viewMode.addEventListener("change", () => {
        const mode = viewMode.value;
        if (mode === "compare_dark_vs_zerodce") {
            grid4Way.classList.add("d-none");
            grid2Way.classList.remove("d-none");
        } else {
            grid4Way.classList.remove("d-none");
            grid2Way.classList.add("d-none");
        }
    });

    function renderDetectionsTable(detections) {
        if (!detections || detections.length === 0) {
            detectionsCard.classList.remove("d-none");
            detectionsTbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">Không phát hiện vật thể nào với ngưỡng tin cậy hiện tại.</td></tr>`;
            return;
        }

        detectionsCard.classList.remove("d-none");
        detectionsTbody.innerHTML = detections.map((det, i) => `
            <tr>
                <td><strong>#${i + 1}</strong></td>
                <td><span class="class-tag">${det.class}</span></td>
                <td>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-weight: 700; color: #34D399;">${(det.confidence * 100).toFixed(1)}%</span>
                        <div style="width: 80px; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
                            <div style="width: ${det.confidence * 100}%; height: 100%; background: #10B981;"></div>
                        </div>
                    </div>
                </td>
                <td><code>[${det.bbox.join(", ")}]</code></td>
            </tr>
        `).join("");
    }

    // =========================================================================
    // 5. WEIGHTS INSPECTOR & METADATA LOADER
    // =========================================================================
    async function loadWeightsMetadata() {
        try {
            const res = await fetch("/api/weights_info");
            weightsData = await res.json();

            // Populate Device
            const deviceBadge = document.getElementById("deviceBadge");
            if (deviceBadge) deviceBadge.textContent = `💻 Thiết Bị: ${weightsData.device}`;

            // Populate Zero-DCE
            if (weightsData.models.zerodce) {
                const dce = weightsData.models.zerodce;
                document.getElementById("dceParamCount").textContent = dce.total_params.toLocaleString();
                document.getElementById("dceFileSize").textContent = `${dce.size_kb} KB`;
                
                const dceLayersTbody = document.getElementById("dceLayersTbody");
                if (dceLayersTbody && dce.layers) {
                    dceLayersTbody.innerHTML = dce.layers.map(l => `
                        <tr>
                            <td><code>${l.layer}</code></td>
                            <td><code>${l.shape}</code></td>
                            <td>${l.params.toLocaleString()}</td>
                        </tr>
                    `).join("");
                }
            }

            // Populate YOLO Dark
            if (weightsData.models.yolo_dark) {
                const yd = weightsData.models.yolo_dark;
                document.getElementById("darkMap50").textContent = `${(yd.metrics.map50 * 100).toFixed(1)}%`;
                document.getElementById("darkPrecision").textContent = `${(yd.metrics.precision * 100).toFixed(1)}%`;
                document.getElementById("darkRecall").textContent = `${(yd.metrics.recall * 100).toFixed(1)}%`;
                document.getElementById("darkEpochs").textContent = `${yd.total_epochs} Epochs`;

                const darkCloud = document.getElementById("darkClassesCloud");
                if (darkCloud && yd.classes) {
                    darkCloud.innerHTML = Object.values(yd.classes).map(c => `<span class="class-tag">${c}</span>`).join("");
                }
            }

            // Populate YOLO Retrained
            if (weightsData.models.yolo_zerodce) {
                const yr = weightsData.models.yolo_zerodce;
                document.getElementById("retrainedPrecision").textContent = `${(yr.metrics.precision * 100).toFixed(1)}%`;
                document.getElementById("retrainedMap50").textContent = `${(yr.metrics.map50 * 100).toFixed(1)}%`;
                document.getElementById("retrainedRecall").textContent = `${(yr.metrics.recall * 100).toFixed(1)}%`;
                document.getElementById("retrainedEpochs").textContent = `${yr.total_epochs} Epochs`;
            }

            // Initial render of charts
            renderCharts(weightsData);
        } catch (err) {
            console.error("Lỗi tải thông tin trọng số:", err);
        }
    }

    // =========================================================================
    // 6. SCIENTIFIC CHARTS (Chart.js)
    // =========================================================================
    function renderCharts(data) {
        // Chart 1: mAP Comparison across 4 Scenarios
        const ctxMap = document.getElementById("chartMapComparison");
        if (ctxMap) {
            if (chartMap) chartMap.destroy();

            const labels = ["1. Raw Dark", "2. CLAHE", "3. Zero-DCE (Domain Shift)", "4. Zero-DCE Retrained"];
            const map50Vals = [0.6235, 0.6747, 0.2291, 0.5922];
            const map50_95Vals = [0.2909, 0.3204, 0.0970, 0.2737];

            chartMap = new Chart(ctxMap, {
                type: "bar",
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: "mAP@0.5 (Chỉ số cốt lõi)",
                            data: map50Vals,
                            backgroundColor: ["#38BDF8", "#34D399", "#F87171", "#818CF8"],
                            borderRadius: 6
                        },
                        {
                            label: "mAP@0.5:0.95 (Khớp biên)",
                            data: map50_95Vals,
                            backgroundColor: ["#0284C7", "#059669", "#DC2626", "#4F46E5"],
                            borderRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 0.8,
                            grid: { color: "rgba(255,255,255,0.06)" },
                            ticks: { color: "#94A3B8" }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: "#CBD5E1", font: { weight: "600" } }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: "#E2E8F0" } }
                    }
                }
            });
        }

        // Chart 2: 40 Epochs Training Loss / mAP Convergence Curve
        const ctxLoss = document.getElementById("chartTrainingLoss");
        if (ctxLoss && data.models.yolo_zerodce && data.models.yolo_zerodce.history) {
            if (chartLoss) chartLoss.destroy();

            const hist = data.models.yolo_zerodce.history;
            const epochs = hist.epochs.length > 0 ? hist.epochs : Array.from({length: 40}, (_, i) => i + 1);
            const lossData = hist.box_loss.length > 0 ? hist.box_loss : [];
            const mapData = hist.map50.length > 0 ? hist.map50 : [];

            chartLoss = new Chart(ctxLoss, {
                type: "line",
                data: {
                    labels: epochs,
                    datasets: [
                        {
                            label: "Box Loss (Hội tụ giảm)",
                            data: lossData,
                            borderColor: "#F43F5E",
                            backgroundColor: "rgba(244, 63, 94, 0.1)",
                            tension: 0.3,
                            yAxisID: "y"
                        },
                        {
                            label: "mAP@0.5 (Tăng trưởng)",
                            data: mapData,
                            borderColor: "#10B981",
                            backgroundColor: "rgba(16, 185, 129, 0.1)",
                            tension: 0.3,
                            yAxisID: "y1"
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: "index", intersect: false },
                    scales: {
                        x: {
                            title: { display: true, text: "Epoch (1 - 40)", color: "#94A3B8" },
                            grid: { color: "rgba(255,255,255,0.06)" },
                            ticks: { color: "#94A3B8" }
                        },
                        y: {
                            type: "linear",
                            display: true,
                            position: "left",
                            title: { display: true, text: "Loss", color: "#F43F5E" },
                            grid: { color: "rgba(255,255,255,0.06)" },
                            ticks: { color: "#F43F5E" }
                        },
                        y1: {
                            type: "linear",
                            display: true,
                            position: "right",
                            title: { display: true, text: "mAP@50", color: "#10B981" },
                            grid: { drawOnChartArea: false },
                            ticks: { color: "#10B981" }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: "#E2E8F0" } }
                    }
                }
            });
        }
    }

    // Initialize
    loadWeightsMetadata();
    loadSampleImages();
});
