document.addEventListener("DOMContentLoaded", () => {
    // 1. Mark active navigation tab
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
        fetch("/api/dashboard-stats")
            .then(res => res.json())
            .then(data => {
                document.getElementById("total-customers").textContent = data.total_customers.toLocaleString();
                document.getElementById("churn-rate").textContent = data.churn_rate + "%";
                document.getElementById("avg-tenure").textContent = data.avg_tenure + " mos";
                document.getElementById("avg-charges").textContent = "$" + data.avg_charges.toFixed(2);
                
                // Render charts if ChurnCharts is loaded
                if (window.ChurnCharts) {
                    window.ChurnCharts.renderContractChart("contract-chart", data.churn_by_contract);
                    window.ChurnCharts.renderSegmentChart("segment-chart", data.segments);
                }
            })
            .catch(err => console.error("Error loading dashboard stats:", err));
    }

    // 3. Handle Predict Form
    const predictForm = document.getElementById("predict-form");
    if (predictForm) {
        predictForm.addEventListener("submit", (e) => {
            e.preventDefault();
            
            // Collect Form Inputs
            const formData = {
                CustomerID: "CUST-" + Math.floor(10000 + Math.random() * 90000),
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

            // Fetch Prediction API
            fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(formData)
            })
            .then(res => res.json())
            .then(resData => {
                // Show Result Card
                const resultDiv = document.getElementById("prediction-result");
                resultDiv.style.display = "block";
                
                // Bind outputs
                document.getElementById("res-prob").textContent = (resData.ChurnProbability * 100).toFixed(1) + "%";
                
                const badge = document.getElementById("res-risk");
                badge.textContent = resData.RiskLevel + " Risk";
                badge.className = "badge";
                if (resData.RiskLevel === "High") {
                    badge.classList.add("badge-danger");
                } else if (resData.RiskLevel === "Medium") {
                    badge.classList.add("badge-warning");
                } else {
                    badge.classList.add("badge-success");
                }
                
                document.getElementById("res-clv").textContent = "$" + resData.PredictedCLV.toFixed(2);
                document.getElementById("res-loss").textContent = "$" + resData.EstimatedRevenueLoss.toFixed(2);
                document.getElementById("res-segment").textContent = resData.CustomerSegment;
                
                // Bind recommendations
                const recList = document.getElementById("res-recs");
                recList.innerHTML = "";
                resData.Recommendations.forEach(rec => {
                    const item = document.createElement("div");
                    item.className = "rec-item";
                    item.innerHTML = `
                        <div class="rec-action">${rec.Action}</div>
                        <div class="rec-details">${rec.Details}</div>
                    `;
                    recList.appendChild(item);
                });
                
                // Scroll to result
                resultDiv.scrollIntoView({ behavior: "smooth" });
            })
            .catch(err => {
                alert("Error scoring customer churn prediction. See console for logs.");
                console.error(err);
            });
        });
    }

    // 4. Load Reports History
    const reportsTableBody = document.getElementById("reports-tbody");
    if (reportsTableBody) {
        fetch("/api/reports")
            .then(res => res.json())
            .then(data => {
                reportsTableBody.innerHTML = "";
                if (data.length === 0) {
                    reportsTableBody.innerHTML = `<tr><td colspan="7" style="text-align: center;">No prediction reports found. Submit predictions from the Predict page first.</td></tr>`;
                    return;
                }
                data.forEach(row => {
                    const tr = document.createElement("tr");
                    
                    const probVal = (parseFloat(row.ChurnProbability) * 100).toFixed(1) + "%";
                    const clvVal = "$" + parseFloat(row.PredictedCLV).toFixed(2);
                    const lossVal = "$" + parseFloat(row.EstimatedRevenueLoss).toFixed(2);
                    
                    let riskBadge = `<span class="badge badge-success">Low</span>`;
                    if (row.RiskLevel === "High") {
                        riskBadge = `<span class="badge badge-danger">High</span>`;
                    } else if (row.RiskLevel === "Medium") {
                        riskBadge = `<span class="badge badge-warning">Medium</span>`;
                    }
                    
                    tr.innerHTML = `
                        <td><strong>${row.CustomerID}</strong></td>
                        <td>${probVal}</td>
                        <td>${row.ChurnPrediction}</td>
                        <td>${riskBadge}</td>
                        <td>${clvVal}</td>
                        <td>${lossVal}</td>
                        <td>${row.CustomerSegment}</td>
                    `;
                    reportsTableBody.appendChild(tr);
                });
            })
            .catch(err => console.error("Error loading reports:", err));
    }
});
