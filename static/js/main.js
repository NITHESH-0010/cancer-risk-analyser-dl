document.addEventListener('DOMContentLoaded', () => {
    // Load metrics
    fetch('/api/model-info')
        .then(res => res.json())
        .then(data => {
            if (data.test_metrics) {
                const buildTable = (dataArr, tableId) => {
                    const table = document.getElementById(tableId);
                    if (!dataArr.length) return;
                    const keys = Object.keys(dataArr[0]);
                    let html = '<thead><tr>' + keys.map(k => `<th>${k}</th>`).join('') + '</tr></thead><tbody>';
                    dataArr.forEach(row => {
                        html += '<tr>' + keys.map(k => {
                            let val = row[k];
                            if (typeof val === 'number') val = val.toFixed(3);
                            return `<td>${val}</td>`;
                        }).join('') + '</tr>';
                    });
                    html += '</tbody>';
                    table.innerHTML = html;
                };
                buildTable(data.test_metrics, 'testTable');
                buildTable(data.cv_metrics, 'cvTable');
            }
        });

    let waterfallChart = null;

    document.getElementById('predictForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        document.getElementById('globalError').style.display = 'none';
        
        const fd = new FormData(e.target);
        const data = { model: document.getElementById('modelSelect').value };
        let hasError = false;
        
        for (let [key, val] of fd.entries()) {
            const num = parseFloat(val);
            data[key] = num;
            const input = document.getElementById(key);
            const err = document.getElementById(`err-${key}`);
            if (input.min && input.max) {
                if (num < parseFloat(input.min) || num > parseFloat(input.max)) {
                    err.style.display = 'block';
                    hasError = true;
                } else {
                    err.style.display = 'none';
                }
            }
        }
        
        if (hasError) return;

        const btn = document.getElementById('submitBtn');
        btn.disabled = true;
        btn.textContent = 'Calculating...';
        
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await res.json();
            
            if (!res.ok) {
                document.getElementById('globalError').textContent = result.error;
                document.getElementById('globalError').style.display = 'block';
                return;
            }
            
            // Show result
            document.getElementById('resultCard').classList.remove('hidden');
            
            // Gauge
            document.getElementById('riskScore').textContent = `${(result.probability * 100).toFixed(1)}%`;
            const bandEl = document.getElementById('riskBand');
            bandEl.textContent = result.risk_band + ' Risk';
            
            const gauge = document.getElementById('riskGauge');
            gauge.className = 'risk-gauge';
            if (result.risk_color === 'green') gauge.classList.add('risk-low');
            else if (result.risk_color === 'amber') gauge.classList.add('risk-mod');
            else if (result.risk_color === 'orange') gauge.classList.add('risk-high');
            else gauge.classList.add('risk-vhigh');

            // Waterfall Chart
            renderWaterfall(result.base_value_pts, result.contributions_pts, result.final_prob_pts);
            
            // Top drivers
            const contribs = Object.entries(result.contributions_pts).sort((a,b) => b[1] - a[1]);
            const top = contribs.filter(c => c[1] > 1.0);
            if (top.length > 0) {
                document.getElementById('topDriversText').innerHTML = `<strong>${top[0][0]}</strong> is the biggest factor raising this score.`;
            } else {
                document.getElementById('topDriversText').textContent = 'No single factor is significantly raising the score above average.';
            }

            // What if
            const wi = result.what_if;
            let wiHtml = '';
            
            const renderWiCard = (factor, tip, key) => {
                if (wi[key]) {
                    const new_p = wi[key].new_prob;
                    const change = wi[key].pts_change;
                    if (change < -0.1) {
                        return `<div class="tip-card">
                            <h4>${factor}</h4>
                            <p>${tip}</p>
                            <p style="color:#16a34a; font-weight:600; margin-top:0.5rem;">Estimated new score: ${new_p.toFixed(1)}% (drops by ${Math.abs(change).toFixed(1)} pts)</p>
                        </div>`;
                    } else {
                        return `<div class="tip-card">
                            <h4>${factor}</h4>
                            <p>${tip}</p>
                            <p style="color:#64748b; font-size:0.8rem; margin-top:0.5rem;">Changing this does not significantly improve the score.</p>
                        </div>`;
                    }
                }
                return '';
            };
            
            wiHtml += renderWiCard('Smoking', 'Stopping smoking supports overall health.', 'Smoking');
            wiHtml += renderWiCard('Alcohol Intake', 'Moderating alcohol intake benefits long-term wellness.', 'AlcoholIntake');
            wiHtml += renderWiCard('Physical Activity', 'Regular activity is beneficial for a healthy lifestyle.', 'PhysicalActivity');
            wiHtml += renderWiCard('BMI', 'Maintaining a healthy weight positively impacts well-being.', 'BMI');
            
            if (wi['Combined'] && wi['Combined'].pts_change < -0.1) {
                wiHtml += `<div class="tip-card" style="background:#e0f2fe; border-color:#bae6fd;">
                    <h4>Combined Impact</h4>
                    <p style="color:#0284c7; font-weight:600; margin-top:0.25rem;">Adopting all applicable lifestyle changes could shift the score to ${wi['Combined'].new_prob.toFixed(1)}%.</p>
                </div>`;
            }
            
            document.getElementById('whatIfGrid').innerHTML = wiHtml;
            
            // Scroll to result
            if (window.innerWidth < 768) {
                document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth' });
            }

        } catch (err) {
            console.error(err);
            document.getElementById('globalError').textContent = 'An error occurred connecting to the server.';
            document.getElementById('globalError').style.display = 'block';
        } finally {
            btn.disabled = false;
            btn.textContent = 'Calculate Risk';
        }
    });

    function renderWaterfall(base, contribs, final) {
        const ctx = document.getElementById('waterfallChart').getContext('2d');
        if (waterfallChart) waterfallChart.destroy();
        
        // Sort contributions by absolute size for waterfall, or keep fixed order?
        // Let's sort by actual value for a nice waterfall (negative first, then positive)
        const sorted = Object.entries(contribs).sort((a,b) => a[1] - b[1]);
        
        const labels = ['Base Risk', ...sorted.map(x => x[0]), 'Final Score'];
        
        // Waterfall logic: Chart.js needs a floating bar chart
        // Floating bar: data = [[start, end], [start, end]]
        const data = [];
        const bgColors = [];
        
        data.push([0, base]);
        bgColors.push('#94a3b8');
        
        let current = base;
        for (let [k, v] of sorted) {
            data.push([current, current + v]);
            bgColors.push(v < 0 ? '#22c55e' : '#ef4444');
            current += v;
        }
        
        data.push([0, final]);
        bgColors.push('#3b82f6');
        
        waterfallChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: bgColors,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { autoSkip: false, maxRotation: 45, minRotation: 45 } },
                    y: { title: { display: true, text: 'Probability (%)' } }
                }
            }
        });
    }
});
