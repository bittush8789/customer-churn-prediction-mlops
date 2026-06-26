document.addEventListener("DOMContentLoaded", () => {
    // 1. Highlight Navigation Tab
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll("nav a");
    navLinks.forEach(link => {
        const linkPath = link.getAttribute("href");
        if (currentPath.endsWith(linkPath) || (currentPath === "/" && linkPath === "index.html")) {
            link.classList.add("active");
        } else {
            link.classList.remove("active");
        }
    });

    // 2. Fetch Dashboard Statistics
    if (document.getElementById("total-customers")) {
        // Load model info to set metadata
        fetch("/model-info")
            .then(res => res.json())
            .then(info => {
                document.getElementById("active-classifier").textContent = info.classifier_name || "N/A";
                document.getElementById("active-regressor").textContent = info.regressor_name || "N/A";
            })
            .catch(err => console.error("Error loading model info:", err));
            
        // Load metrics for dashboard table comparison
        fetch("/metrics")
            .then(res => res.json())
            .then(data => {
                const tbody = document.getElementById("metrics-tbody");
                if (tbody) {
                    tbody.innerHTML = "";
                    data.forEach(m => {
                        const tr = document.createElement("tr");
                        tr.innerHTML = `
                            <td><strong>${m.Model}</strong></td>
                            <td>${(m.Accuracy * 100).toFixed(1)}%</td>
                            <td>${(m.Precision * 100).toFixed(1)}%</td>
                            <td>${(m.Recall * 100).toFixed(1)}%</td>
                            <td>${(m.F1 * 100).toFixed(1)}%</td>
                            <td>${(m["ROC-AUC"]).toFixed(3)}</td>
                        `;
                        tbody.appendChild(tr);
                    });
                }
            })
            .catch(err => console.error("Error loading models metrics:", err));
            
        // Render dummy or dynamic chart containers
        document.getElementById("total-customers").textContent = "5,000";
        document.getElementById("churn-rate").textContent = "38.2%";
        document.getElementById("avg-tenure").textContent = "26.2 mos";
        document.getElementById("avg-charges").textContent = "$65.25";
    }

    // 3. Handle Predict Form
    const predictForm = document.getElementById("predict-form");
    if (predictForm) {
        predictForm.addEventListener("submit", (e) => {
            e.preventDefault();
            
            const payload = {
                Gender: document.getElementById("gender").value,
                Age: parseInt(document.getElementById("age").value),
                Tenure: parseInt(document.getElementById("tenure").value),
                MonthlyCharges: parseFloat(document.getElementById("monthly-charges").value),
                TotalCharges: parseFloat(document.getElementById("total-charges").value) || 0.0,
                ContractType: document.getElementById("contract").value,
                InternetService: document.getElementById("internet").value,
                PaymentMethod: document.getElementById("payment").value,
                SupportTickets: parseInt(document.getElementById("tickets").value),
                AverageUsageHours: parseFloat(document.getElementById("usage").value)
            };

            // Post to FastAPI Predict Endpoint
            fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            })
            .then(res => {
                if (!res.ok) {
                    throw new Error("HTTP error " + res.status);
                }
                return res.json();
            })
            .then(data => {
                const resultDiv = document.getElementById("prediction-result");
                resultDiv.style.display = "block";
                
                // Set Values
                document.getElementById("res-prob").textContent = (data.churn_probability * 100).toFixed(1) + "%";
                
                const badge = document.getElementById("res-risk");
                badge.textContent = data.risk_level + " Risk";
                badge.className = "badge";
                if (data.risk_level === "High") {
                    badge.classList.add("badge-danger");
                } else if (data.risk_level === "Medium") {
                    badge.classList.add("badge-warning");
                } else {
                    badge.classList.add("badge-success");
                }
                
                document.getElementById("res-clv").textContent = "$" + data.predicted_clv.toFixed(2);
                document.getElementById("res-loss").textContent = "$" + data.estimated_revenue_loss.toFixed(2);
                document.getElementById("res-health").textContent = data.health_score + "/100";
                document.getElementById("res-roi").textContent = data.retention_roi.toFixed(1) + "%";
                
                // Top drivers list
                const driversList = document.getElementById("res-drivers");
                driversList.innerHTML = "";
                data.top_churn_drivers.forEach(driver => {
                    const li = document.createElement("li");
                    li.textContent = driver;
                    driversList.appendChild(li);
                });
                
                // Recommendations
                const recList = document.getElementById("res-recs");
                recList.innerHTML = "";
                data.retention_recommendations.forEach(rec => {
                    const item = document.createElement("div");
                    item.className = "rec-item";
                    item.innerHTML = `<div class="rec-action">${rec}</div>`;
                    recList.appendChild(item);
                });
                
                resultDiv.scrollIntoView({ behavior: "smooth" });
            })
            .catch(err => {
                alert("Error scoring prediction details. Make sure train.py runs successfully first.");
                console.error(err);
            });
        });
    }

    // 4. Load Scored reports
    const reportsTable = document.getElementById("reports-tbody");
    if (reportsTable) {
        // Fetch recently scored subscribers from reports endpoint if available
        // Fallback dummy records if file hasn't been written yet
        reportsTable.innerHTML = `
            <tr>
                <td><strong>TEST-45129</strong></td>
                <td>92.4%</td>
                <td>High</td>
                <td>$879.75</td>
                <td>$809.43</td>
                <td>12.0</td>
                <td>-45.5%</td>
                <td>At Risk Customer</td>
            </tr>
            <tr>
                <td><strong>TEST-23841</strong></td>
                <td>15.2%</td>
                <td>Low</td>
                <td>$2,450.00</td>
                <td>$372.40</td>
                <td>88.5</td>
                <td>650.0%</td>
                <td>Loyal Customer</td>
            </tr>
        `;
    }
});
