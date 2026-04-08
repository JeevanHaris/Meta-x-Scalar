async function loadResults() {
    try {
        const response = await fetch('../baseline_results.json');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error loading results:', error);
        return null;
    }
}

function updateUI(taskId, results) {
    if (!results || !results[taskId]) return;
    
    const task = results[taskId];
    const taskNames = {
        'task_easy': 'Basic Ticket Classification',
        'task_medium': 'Full Ticket Triage with Responses',
        'task_hard': 'Advanced Inbox Management with SLA Compliance'
    };

    const taskDescs = {
        'task_easy': 'Grade consists of categorization and priority accuracy across 5 support tickets.',
        'task_medium': 'Requires classification, professional responses, and escalation for critical cases.',
        'task_hard': 'Complex workflow: classification, responses, escalation, duplicate detection, and SLA breach tagging.'
    };

    // Update Header Score
    const scores = Object.values(results).map(r => r.final_score);
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    document.getElementById('avg-score').textContent = avgScore.toFixed(4);

    // Update Current Task Info
    document.getElementById('current-task-name').textContent = taskNames[taskId];
    document.getElementById('current-task-desc').textContent = taskDescs[taskId];
    document.getElementById('task-score').textContent = task.final_score.toFixed(4);
    document.getElementById('score-progress').style.width = (task.final_score * 100) + '%';
    
    document.getElementById('stat-classified').textContent = `${task.tickets_classified}/${task.tickets_total}`;
    document.getElementById('stat-responded').textContent = `${task.tickets_responded}/${task.tickets_total}`;
    document.getElementById('stat-steps').textContent = task.steps_taken;

    // Update Logs
    const actionList = document.getElementById('action-list');
    actionList.innerHTML = '';
    
    if (task.action_log && task.action_log.length > 0) {
        task.action_log.forEach(log => {
            const li = document.createElement('li');
            li.className = `action-item ${log.reward > 0 ? 'reward-positive' : log.reward < 0 ? 'reward-negative' : ''}`;
            
            const action = log.action;
            let details = '';
            if (action.category) details += ` | Category: ${action.category}`;
            if (action.priority) details += ` | Priority: ${action.priority}`;
            if (action.escalate_to) details += ` | Escalate: ${action.escalate_to}`;
            if (action.response_text) details += ` | Response: "${action.response_text.substring(0, 50)}..."`;
            if (action.tags) details += ` | Tags: [${action.tags.join(', ')}]`;

            li.innerHTML = `
                <div class="meta">
                    <span class="step">Step ${log.step}</span>
                    <span class="reward">Reward: ${log.reward > 0 ? '+' : ''}${log.reward.toFixed(2)}</span>
                </div>
                <div class="type">${action.action_type} - ${action.ticket_id}</div>
                <div class="details">${details}</div>
            `;
            actionList.appendChild(li);
        });
    } else {
        actionList.innerHTML = '<li class="action-item">No action logs found for this task.</li>';
    }

    // Update Details
    document.getElementById('grader-details-raw').textContent = JSON.stringify(task.grader_details, null, 2);
}

document.addEventListener('DOMContentLoaded', async () => {
    let results = await loadResults();
    
    // Tab switching
    const taskTabs = document.querySelectorAll('.task-tab');
    taskTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            taskTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            updateUI(tab.dataset.task, results);
        });
    });

    const viewTabs = document.querySelectorAll('.view-tab');
    viewTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            viewTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            document.querySelectorAll('.view-content').forEach(c => c.classList.remove('active'));
            document.getElementById(`${tab.dataset.view}-view`).classList.add('active');
        });
    });

    // Initial load
    async function refresh() {
        let newResults = await loadResults();
        if (newResults) {
            results = newResults;
            const activeTab = document.querySelector('.task-tab.active');
            updateUI(activeTab ? activeTab.dataset.task : 'task_easy', results);
        }
    }

    if (results) {
        updateUI('task_easy', results);
    }
    
    // Auto-refresh every 5 seconds
    setInterval(refresh, 5000);
});
