document.addEventListener('DOMContentLoaded', () => {
    // Register Chart.js DataLabels plugin
    Chart.register(ChartDataLabels);

    const form = document.getElementById('assessmentForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');
    const resultsSection = document.getElementById('resultsSection');
    
    let chartInstance = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI Loading State
        submitBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';

        // Gather form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Server error');
            }

            const result = await response.json();
            
            // Update UI with results
            renderResults(result);
            
            // Show results section and scroll
            resultsSection.classList.remove('hidden');
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            // Restore UI
            submitBtn.disabled = false;
            btnText.style.display = 'inline-block';
            btnLoader.style.display = 'none';
        }
    });

    function renderResults(result) {
        // 1. Update Metrics
        document.getElementById('accuracyChip').textContent = `Model Accuracy: ${result.model_accuracy}%`;
        document.getElementById('aucChip').textContent = `AUC: ${result.model_auc}`;

        // 2. Update Gauge
        const prob = result.probability;
        document.getElementById('probabilityValue').textContent = `${prob}%`;
        
        const dasharray = 125.6; // Circumference of semicircle (r=40 -> pi*40 = 125.6)
        const offset = dasharray * (1 - (prob / 100));
        const gaugePath = document.getElementById('gaugePath');
        
        // Color mapping based on risk band
        let color = 'var(--risk-mod)';
        if (result.risk_band === 'Low') color = 'var(--risk-low)';
        if (result.risk_band === 'High') color = 'var(--risk-high)';
        
        setTimeout(() => {
            gaugePath.style.strokeDashoffset = offset;
            gaugePath.style.stroke = color;
        }, 100);

        // 3. Update Risk Badge
        const badge = document.getElementById('riskBadge');
        badge.textContent = `${result.risk_band} Risk`;
        badge.style.backgroundColor = color;
        
        const clsText = result.predicted_class === 1 ? 'Positive prediction' : 'Negative prediction';
        document.getElementById('predictedClass').textContent = clsText;

        // 4. Render Chart
        renderChart(result.contributions);

        // 5. Render Factors List
        renderFactorsList(result.contributions);

        // 6. Render Tips
        renderTips(result.contributions);
    }

    function renderChart(contributions) {
        const ctx = document.getElementById('impactChart').getContext('2d');
        
        if (chartInstance) {
            chartInstance.destroy();
        }

        const labels = contributions.map(c => c.name);
        const data = contributions.map(c => c.points);
        
        const colors = data.map(v => v > 0 ? '#ef4444' : '#10b981'); // Red for + risk, Green for - risk

        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderRadius: 4,
                    barThickness: 'flex',
                    maxBarThickness: 30
                }]
            },
            options: {
                indexAxis: 'y', // Horizontal
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false },
                    datalabels: {
                        anchor: (context) => context.dataset.data[context.dataIndex] >= 0 ? 'end' : 'start',
                        align: (context) => context.dataset.data[context.dataIndex] >= 0 ? 'right' : 'left',
                        formatter: (value) => {
                            const sign = value > 0 ? '+' : '';
                            return `${sign}${value.toFixed(1)} pts`;
                        },
                        font: { weight: 'bold', size: 11 },
                        color: (context) => context.dataset.data[context.dataIndex] >= 0 ? '#ef4444' : '#10b981',
                        padding: { left: 4, right: 4 }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false, drawBorder: false },
                        ticks: { display: false },
                        suggestedMin: Math.min(...data) * 1.2 - 5,
                        suggestedMax: Math.max(...data) * 1.2 + 5
                    },
                    y: {
                        grid: { display: false, drawBorder: false },
                        ticks: { font: { family: 'Inter', size: 12 }, color: '#1e293b' }
                    }
                },
                layout: { padding: { right: 50, left: 20 } },
                animation: { duration: 1000, easing: 'easeOutQuart' }
            }
        });
    }

    function renderFactorsList(contributions) {
        const container = document.getElementById('factorsList');
        container.innerHTML = '';

        contributions.forEach(c => {
            const isPos = c.points > 0;
            const sign = isPos ? '+' : '';
            const cls = isPos ? 'pos' : 'neg';
            
            const el = document.createElement('div');
            el.className = 'factor-item';
            el.innerHTML = `
                <div class="factor-header">
                    <span>${c.name}</span>
                    <span class="factor-points ${cls}">${sign}${c.points.toFixed(1)} pts</span>
                </div>
                <div class="factor-desc">${c.reason}</div>
                <div class="factor-precaution">${c.precaution}</div>
            `;
            container.appendChild(el);
        });
    }

    function renderTips(contributions) {
        const container = document.getElementById('tipsList');
        container.innerHTML = '';
        
        // Take top 3 factors that INCREASE risk
        const topRisks = contributions.filter(c => c.points > 0).slice(0, 3);
        
        if (topRisks.length === 0) {
            container.innerHTML = `<div class="tip-card">
                <h4>Keep it up!</h4>
                <p>You have a great profile. Continue maintaining your healthy lifestyle choices.</p>
            </div>`;
            return;
        }

        topRisks.forEach(c => {
            const el = document.createElement('div');
            el.className = 'tip-card';
            el.innerHTML = `
                <h4>Focus on ${c.name}</h4>
                <p>${c.precaution}</p>
            `;
            container.appendChild(el);
        });
    }
});
