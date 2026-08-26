// Sovereign Control System Dashboard Scripts
// © 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

// System status refresh
function refreshSystemStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            updateSystemMetrics(data);
            updateSystemLog(data);
        })
        .catch(error => {
            console.error('Error fetching status:', error);
            addLogEntry(`Error fetching status: ${error}`, true);
        });
}

// Update system metrics on the dashboard
function updateSystemMetrics(data) {
    // Update system health
    document.getElementById('systemHealth').textContent = data.status.system_health;

    // Update uptime
    document.getElementById('systemUptime').textContent = `${Math.round(data.state.uptime_seconds * 10) / 10}s`;

    // Update task completion
    document.getElementById('taskCompletion').textContent = data.status.task_completion;

    // Update tasks in progress
    document.getElementById('tasksInProgress').textContent = data.status.tasks_in_progress;

    // Update quantum metrics
    document.getElementById('fieldCoherence').textContent = `Coherence: ${data.status.quantum_metrics.field_coherence}`;
    document.getElementById('entanglementStrength').textContent = `Entanglement: ${data.status.quantum_metrics.entanglement_strength}`;

    // Update control system metrics
    document.getElementById('authorityStatus').textContent = `Authority: ${data.status.sovereign_authority}`;
    document.getElementById('executionState').textContent = `Execution: ${data.status.execution_state}`;
}

// Add entry to system log
function addLogEntry(message, isError = false) {
    const log = document.getElementById('systemLog');
    const now = new Date().toISOString();
    const entry = document.createElement('p');
    entry.className = 'log-entry';

    if (isError) {
        entry.style.color = '#ff5555';
    }

    entry.textContent = `[${now}] ${message}`;
    log.insertBefore(entry, log.firstChild);
}

// Update system log with latest data
function updateSystemLog(data) {
    // Add a log entry with the latest system state
    addLogEntry(`System health: ${data.status.system_health}, Tasks: ${data.status.tasks_completed} completed`);
}

// Handle form submissions via AJAX
function setupFormHandlers() {
    // Execute directive form
    document.getElementById('directiveForm').addEventListener('submit', function(e) {
        e.preventDefault();
        submitForm(this, '/execute');
    });

    // Optimize system form
    document.getElementById('optimizeForm').addEventListener('submit', function(e) {
        e.preventDefault();
        submitForm(this, '/optimize');
    });

    // Save state form
    document.getElementById('saveForm').addEventListener('submit', function(e) {
        e.preventDefault();
        submitForm(this, '/save');
    });

    // Load state form
    document.getElementById('loadForm').addEventListener('submit', function(e) {
        e.preventDefault();
        submitForm(this, '/load');
    });
}

// Submit form via AJAX
function submitForm(form, endpoint) {
    const formData = new FormData(form);

    fetch(endpoint, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Response:', data);

        // Add log entry
        addLogEntry(`${endpoint.substring(1)} executed: ${JSON.stringify(data).substring(0, 100)}...`);

        // Refresh page after a delay
        setTimeout(() => {
            window.location.reload();
        }, 2000);
    })
    .catch(error => {
        console.error('Error:', error);
        addLogEntry(`Error: ${error}`, true);
    });
}

// Initialize dashboard
function initializeDashboard() {
    // Set up auto-refresh (every 10 seconds)
    setInterval(refreshSystemStatus, 10000);

    // Set up form handlers
    setupFormHandlers();

    // Initial status refresh
    refreshSystemStatus();

    // Add initial log entry
    addLogEntry('Dashboard initialized');
}

// Run initialization when the document is ready
document.addEventListener('DOMContentLoaded', initializeDashboard);
