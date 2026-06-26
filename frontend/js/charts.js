// Chart rendering utilities
window.ChurnCharts = {
    renderContractChart: function(canvasId, contractData) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        const labels = Object.keys(contractData);
        const values = Object.values(contractData);
        
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Churn Rate (%)',
                    data: values,
                    backgroundColor: [
                        'rgba(99, 102, 241, 0.65)',
                        'rgba(59, 130, 246, 0.65)',
                        'rgba(139, 92, 246, 0.65)'
                    ],
                    borderColor: [
                        '#6366f1',
                        '#3b82f6',
                        '#8b5cf6'
                    ],
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: '#2b354f'
                        },
                        ticks: {
                            color: '#9ca3af'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#9ca3af'
                        }
                    }
                }
            }
        });
    },

    renderSegmentChart: function(canvasId, segmentData) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        const labels = Object.keys(segmentData);
        const values = Object.values(segmentData);
        
        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: [
                        '#ef4444', // At Risk
                        '#3b82f6', // High Value
                        '#10b981', // Loyal
                        '#a855f7'  // New
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            color: '#9ca3af',
                            font: {
                                size: 12
                            }
                        }
                    }
                },
                cutout: '70%'
            }
        });
    }
};
