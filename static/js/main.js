document.addEventListener("DOMContentLoaded", () => {
    // Force light theme settings
    const htmlElement = document.documentElement;
    htmlElement.setAttribute("data-theme", "light");
    let activeCharts = [];

    // ==========================================
    // 2. Loading Animation Overlay
    // ==========================================
    const mainForm = document.getElementById("predict-form");
    const batchForm = document.getElementById("batch-form");
    const loadingOverlay = document.getElementById("loading-overlay");

    const showLoading = () => {
        if (loadingOverlay) {
            loadingOverlay.classList.add("show");
        }
    };

    if (mainForm) {
        mainForm.addEventListener("submit", showLoading);
    }
    if (batchForm) {
        batchForm.addEventListener("submit", showLoading);
    }

    // ==========================================
    // 3. Risk Gauge Progress Meter Animation
    // ==========================================
    const gaugeFill = document.getElementById("gauge-fill");
    if (gaugeFill) {
        const targetWidth = parseFloat(gaugeFill.getAttribute("data-width")) || 0;
        setTimeout(() => {
            gaugeFill.style.width = targetWidth + "%";
        }, 300);
    }

    // ==========================================
    // 4. Interactive Dashboard Charts (Chart.js)
    // ==========================================
    const riskPieCanvas = document.getElementById("riskPieChart");
    const riverLineCanvas = document.getElementById("riverLineChart");

    if (riskPieCanvas && riverLineCanvas) {
        const warningsCount = parseInt(document.getElementById("db-warnings-count").textContent) || 0;
        const safeCount = parseInt(document.getElementById("db-safe-count").textContent) || 0;
        const rawChartData = document.getElementById("db-chart-data").textContent;
        
        let chartData = [];
        try {
            chartData = JSON.parse(rawChartData);
        } catch (e) {
            console.error("Error reading chart records: ", e);
        }

        const getThemeColors = (theme) => {
            const isDark = theme === "dark";
            return {
                text: isDark ? "#8aa8c0" : "#5a7a8a",
                grid: isDark ? "rgba(30, 50, 70, 0.2)" : "rgba(212, 226, 237, 0.5)",
                warnings: isDark ? "#ff6b5a" : "#e85d4a",
                safe: isDark ? "#4ade80" : "#2ecc71",
                water: isDark ? "#60a5fa" : "#3498db"
            };
        };

        let currentThemeColors = getThemeColors(htmlElement.getAttribute("data-theme"));

        // Render Pie Chart
        const renderPieChart = () => {
            return new Chart(riskPieCanvas, {
                type: 'doughnut',
                data: {
                    labels: ['Safe Scans', 'Flood Warnings'],
                    datasets: [{
                        data: [safeCount, warningsCount],
                        backgroundColor: [currentThemeColors.safe, currentThemeColors.warnings],
                        borderColor: htmlElement.getAttribute("data-theme") === "dark" ? "#141e2d" : "#ffffff",
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: currentThemeColors.text,
                                font: { family: 'Inter', weight: 500 }
                            }
                        }
                    }
                }
            });
        };

        // Render Line Chart
        const renderLineChart = () => {
            const labels = chartData.map((_, index) => `Audit ${index + 1}`);
            const riverLevels = chartData.map(item => item.river_level);
            const rainLevels = chartData.map(item => item.seasonal_rainfall);

            return new Chart(riverLineCanvas, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'River Level (m)',
                            data: riverLevels,
                            borderColor: currentThemeColors.warnings,
                            backgroundColor: 'rgba(232, 93, 74, 0.1)',
                            borderWidth: 3,
                            tension: 0.3,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Seasonal Rain (mm)',
                            data: rainLevels,
                            borderColor: currentThemeColors.water,
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            borderWidth: 3,
                            tension: 0.3,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: { color: currentThemeColors.text }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: 'River Level (m)', color: currentThemeColors.text },
                            grid: { color: currentThemeColors.grid },
                            ticks: { color: currentThemeColors.text }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'Rainfall (mm)', color: currentThemeColors.text },
                            grid: { drawOnChartArea: false },
                            ticks: { color: currentThemeColors.text }
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                color: currentThemeColors.text,
                                font: { family: 'Inter', weight: 500 }
                            }
                        }
                    }
                }
            });
        };

        let pieChart = renderPieChart();
        let lineChart = renderLineChart();
        activeCharts = [pieChart, lineChart];

        // Function to rebuild charts on theme toggle
        function updateChartTheme(theme) {
            currentThemeColors = getThemeColors(theme);
            activeCharts.forEach(chart => chart.destroy());
            pieChart = renderPieChart();
            lineChart = renderLineChart();
            activeCharts = [pieChart, lineChart];
        }
    }

    // ==========================================
    // 5. Interactive Log Audit Search & Page List
    // ==========================================
    const historyTableBody = document.getElementById("historyTableBody");
    const logSearchInput = document.getElementById("logSearch");
    const filterButtons = document.querySelectorAll(".filter-btn");
    const historyPagination = document.getElementById("historyPagination");

    if (historyTableBody && historyPagination) {
        const rows = Array.from(historyTableBody.querySelectorAll(".history-row"));
        let activeFilter = "all";
        let activeSearchQuery = "";
        let currentPage = 1;
        const rowsPerPage = 10;

        // Perform row matching
        const filterAndSearchRows = () => {
            return rows.filter(row => {
                const outcome = row.getAttribute("data-outcome");
                const searchTarget = row.querySelector(".search-target").textContent.toLowerCase();
                const refCode = row.querySelector("td").textContent.toLowerCase();
                
                const matchesFilter = activeFilter === "all" || outcome === activeFilter;
                const matchesSearch = searchTarget.includes(activeSearchQuery) || refCode.includes(activeSearchQuery);
                
                return matchesFilter && matchesSearch;
            });
        };

        // Render current visible table page
        const updateTableDisplay = () => {
            const filteredRows = filterAndSearchRows();
            const totalRows = filteredRows.length;
            const totalPages = Math.ceil(totalRows / rowsPerPage) || 1;
            
            // Boundary checks
            if (currentPage > totalPages) currentPage = totalPages;
            if (currentPage < 1) currentPage = 1;

            const startIndex = (currentPage - 1) * rowsPerPage;
            const endIndex = startIndex + rowsPerPage;

            // Toggle show/hide of elements
            rows.forEach(row => row.style.display = "none");
            filteredRows.slice(startIndex, endIndex).forEach(row => row.style.display = "");

            // Build Pagination Buttons
            historyPagination.innerHTML = "";

            if (totalPages > 1) {
                // Previous button
                const prevLi = document.createElement("li");
                prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
                prevLi.innerHTML = `<button class="page-link" type="button"><i class="bi bi-chevron-left"></i></button>`;
                prevLi.addEventListener("click", () => {
                    if (currentPage > 1) {
                        currentPage--;
                        updateTableDisplay();
                    }
                });
                historyPagination.appendChild(prevLi);

                // Numbers
                for (let i = 1; i <= totalPages; i++) {
                    const numLi = document.createElement("li");
                    numLi.className = `page-item ${currentPage === i ? 'active' : ''}`;
                    numLi.innerHTML = `<button class="page-link" type="button">${i}</button>`;
                    numLi.addEventListener("click", () => {
                        currentPage = i;
                        updateTableDisplay();
                    });
                    historyPagination.appendChild(numLi);
                }

                // Next button
                const nextLi = document.createElement("li");
                nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
                nextLi.innerHTML = `<button class="page-link" type="button"><i class="bi bi-chevron-right"></i></button>`;
                nextLi.addEventListener("click", () => {
                    if (currentPage < totalPages) {
                        currentPage++;
                        updateTableDisplay();
                    }
                });
                historyPagination.appendChild(nextLi);
            }
        };

        // Bind filter outcomes
        filterButtons.forEach(btn => {
            btn.addEventListener("click", (e) => {
                filterButtons.forEach(b => b.classList.remove("active"));
                e.target.classList.add("active");
                activeFilter = e.target.getAttribute("data-filter");
                currentPage = 1;
                updateTableDisplay();
            });
        });

        // Bind search entry
        if (logSearchInput) {
            logSearchInput.addEventListener("input", (e) => {
                activeSearchQuery = e.target.value.toLowerCase().trim();
                currentPage = 1;
                updateTableDisplay();
            });
        }

        // Run initial setup
        updateTableDisplay();
    }
});
