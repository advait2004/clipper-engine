document.addEventListener('DOMContentLoaded', async () => {
    // ─── Element refs ────────────────────────────────────────────────────────
    const modeSingleBtn = document.getElementById('mode-single-btn');
    const modeFolderBtn = document.getElementById('mode-folder-btn');
    const singleFormCard = document.getElementById('single-form-card');
    const folderFormCard = document.getElementById('folder-form-card');

    // Single mode
    const form = document.getElementById('clip-form');
    const submitBtn = document.getElementById('submit-btn');
    const statusCard = document.getElementById('status-card');
    const jobBadge = document.getElementById('job-badge');
    const jobMessage = document.getElementById('job-message');
    const progressBar = document.getElementById('progress-bar');
    const cancelBtn = document.getElementById('cancel-btn');
    const resultsContainer = document.getElementById('results-container');
    const clipsList = document.getElementById('clips-list');

    // Folder mode
    const folderForm = document.getElementById('folder-form');
    const folderSubmitBtn = document.getElementById('folder-submit-btn');
    const processingMode = document.getElementById('processing-mode');
    const smartEditOptions = document.getElementById('smart-edit-options');
    const batchStatusCard = document.getElementById('batch-status-card');
    const batchTitle = document.getElementById('batch-title');
    const batchBadge = document.getElementById('batch-badge');
    const batchMessage = document.getElementById('batch-message');
    const batchProgressBar = document.getElementById('batch-progress-bar');
    const batchCancelBtn = document.getElementById('batch-cancel-btn');
    const batchJobsContainer = document.getElementById('batch-jobs-container');
    const batchJobsList = document.getElementById('batch-jobs-list');
    const batchResultsContainer = document.getElementById('batch-results-container');
    const batchClipsList = document.getElementById('batch-clips-list');

    let pollInterval = null;
    let currentJobId = null;
    let currentBatchId = null;

    // State for selected styles
    let selectedCaptionSingle = 'karaoke';
    let selectedCaptionFolder = 'karaoke';
    let selectedEditStyle = 'podcast';

    // ─── Load styles from API ────────────────────────────────────────────────
    let stylesData = { edit_styles: [], caption_styles: [], transitions: [] };
    try {
        const res = await fetch('/api/styles');
        stylesData = await res.json();
    } catch (e) {
        console.error('Failed to load styles:', e);
    }

    // Render selectors
    renderCaptionSelector('single-caption-selector', stylesData.caption_styles, selectedCaptionSingle, (id) => {
        selectedCaptionSingle = id;
    });
    renderCaptionSelector('folder-caption-selector', stylesData.caption_styles, selectedCaptionFolder, (id) => {
        selectedCaptionFolder = id;
    });
    renderEditStyleSelector('edit-style-selector', stylesData.edit_styles, selectedEditStyle, (id) => {
        selectedEditStyle = id;
        // Update default caption based on edit style
        const style = stylesData.edit_styles.find(s => s.id === id);
        if (style && style.default_caption) {
            selectedCaptionFolder = style.default_caption;
            renderCaptionSelector('folder-caption-selector', stylesData.caption_styles, selectedCaptionFolder, (cid) => {
                selectedCaptionFolder = cid;
            });
        }
    });

    // ─── Processing Mode toggle ──────────────────────────────────────────────
    processingMode.addEventListener('change', () => {
        smartEditOptions.style.display = processingMode.value === 'extract_highlights' ? 'none' : 'block';
    });

    // ─── Mode Toggle ─────────────────────────────────────────────────────────
    modeSingleBtn.addEventListener('click', () => switchMode('single'));
    modeFolderBtn.addEventListener('click', () => switchMode('folder'));

    function switchMode(mode) {
        modeSingleBtn.classList.toggle('active', mode === 'single');
        modeFolderBtn.classList.toggle('active', mode === 'folder');
        singleFormCard.classList.toggle('hidden', mode !== 'single');
        folderFormCard.classList.toggle('hidden', mode !== 'folder');
        statusCard.classList.add('hidden');
        batchStatusCard.classList.add('hidden');
    }

    // ─── Single Job ──────────────────────────────────────────────────────────
    cancelBtn.addEventListener('click', async () => {
        if (!currentJobId) return;
        cancelBtn.disabled = true;
        cancelBtn.textContent = 'Cancelling...';
        try {
            await fetch(`/api/jobs/${currentJobId}/cancel`, { method: 'POST' });
            if (pollInterval) clearInterval(pollInterval);
            cancelBtn.classList.add('hidden');
            updateBadge(jobBadge, 'error', 'Cancelled');
            jobMessage.textContent = 'Job cancelled by user';
            progressBar.classList.remove('indeterminate');
            progressBar.style.width = '100%';
            progressBar.style.background = '#ef4444';
            resetSingleForm();
        } catch (e) { console.error(e); }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const url = document.getElementById('url').value;
        const viralityMode = document.getElementById('virality-mode')?.checked ?? true;

        statusCard.classList.remove('hidden');
        batchStatusCard.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        clipsList.innerHTML = '';
        updateBadge(jobBadge, 'pending', 'Starting...');
        progressBar.classList.add('indeterminate');
        progressBar.style.width = '100%';
        progressBar.style.background = 'linear-gradient(90deg, #3b82f6, #8b5cf6)';
        cancelBtn.classList.add('hidden');
        submitBtn.disabled = true;
        submitBtn.querySelector('.btn-text').textContent = 'Processing...';
        submitBtn.querySelector('.spinner').classList.remove('hidden');

        try {
            const response = await fetch('/api/jobs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    url, 
                    caption_style: selectedCaptionSingle,
                    virality_mode: viralityMode,
                    clips_per_video: parseInt(document.getElementById('clips-per-video').value, 10) || 1
                })
            });
            if (!response.ok) throw new Error('Failed to create job');
            const data = await response.json();
            currentJobId = data.job_id;
            pollJob(data.job_id);
        } catch (error) {
            handleSingleError(error.message);
        }
    });

    function pollJob(jobId) {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/jobs/${jobId}`);
                if (!res.ok) throw new Error('Failed to fetch job status');
                const job = await res.json();

                if (job.status === 'processing') {
                    updateBadge(jobBadge, 'processing', 'Processing...');
                    jobMessage.textContent = `${job.stage || 'Working...'} (${job.progress || 0}%)`;
                    progressBar.classList.remove('indeterminate');
                    progressBar.style.width = `${job.progress || 0}%`;
                    cancelBtn.classList.remove('hidden');
                    cancelBtn.disabled = false;
                    cancelBtn.textContent = 'Cancel Job';
                }
                else if (job.status === 'done') {
                    clearInterval(pollInterval);
                    cancelBtn.classList.add('hidden');
                    updateBadge(jobBadge, 'done', 'Completed');
                    jobMessage.textContent = 'Clips generated successfully!';
                    progressBar.classList.remove('indeterminate');
                    progressBar.style.width = '100%';
                    progressBar.style.background = '#10b981';
                    if (job.clips && job.clips.length > 0) {
                        resultsContainer.classList.remove('hidden');
                        job.clips.forEach(clip => {
                            const li = document.createElement('li');
                            li.textContent = clip.split(/[\\/]/).pop();
                            clipsList.appendChild(li);
                        });
                    }
                    resetSingleForm();
                }
                else if (job.status === 'error') {
                    clearInterval(pollInterval);
                    cancelBtn.classList.add('hidden');
                    handleSingleError(job.error || 'Unknown error');
                }
            } catch (error) {
                clearInterval(pollInterval);
                handleSingleError(error.message);
            }
        }, 2000);
    }

    function handleSingleError(msg) {
        updateBadge(jobBadge, 'error', 'Error');
        jobMessage.textContent = msg;
        progressBar.classList.remove('indeterminate');
        progressBar.style.width = '100%';
        progressBar.style.background = '#ef4444';
        resetSingleForm();
    }

    function resetSingleForm() {
        submitBtn.disabled = false;
        submitBtn.querySelector('.btn-text').textContent = 'Generate Clips';
        submitBtn.querySelector('.spinner').classList.add('hidden');
    }

    // ─── Batch / Smart Edit ──────────────────────────────────────────────────
    batchCancelBtn.addEventListener('click', async () => {
        if (!currentBatchId) return;
        batchCancelBtn.disabled = true;
        batchCancelBtn.textContent = 'Cancelling...';
        try {
            await fetch(`/api/batch-jobs/${currentBatchId}/cancel`, { method: 'POST' });
            if (pollInterval) clearInterval(pollInterval);
            batchCancelBtn.classList.add('hidden');
            updateBadge(batchBadge, 'error', 'Cancelled');
            batchMessage.textContent = 'Cancelled by user';
            batchProgressBar.classList.remove('indeterminate');
            batchProgressBar.style.width = '100%';
            batchProgressBar.style.background = '#ef4444';
            resetFolderForm();
        } catch (e) { console.error(e); }
    });

    folderForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const folderPath = document.getElementById('folder-path').value;
        const mode = processingMode.value;

        batchStatusCard.classList.remove('hidden');
        statusCard.classList.add('hidden');
        batchJobsContainer.classList.add('hidden');
        batchResultsContainer.classList.add('hidden');
        batchJobsList.innerHTML = '';
        batchClipsList.innerHTML = '';
        batchTitle.textContent = mode === 'merge_all' ? 'Smart Edit' : 'Batch Status';
        updateBadge(batchBadge, 'pending', 'Starting...');
        batchMessage.textContent = mode !== 'extract_highlights' ? 'Initializing AI editor...' : 'Scanning folder...';
        batchProgressBar.classList.add('indeterminate');
        batchProgressBar.style.width = '100%';
        batchProgressBar.style.background = 'linear-gradient(90deg, #3b82f6, #8b5cf6)';
        batchCancelBtn.classList.add('hidden');
        folderSubmitBtn.disabled = true;
        folderSubmitBtn.querySelector('.btn-text').textContent = 'Processing...';
        folderSubmitBtn.querySelector('.spinner').classList.remove('hidden');

        try {
            const numVariations = parseInt(document.getElementById('num-variations').value, 10) || 1;

            const response = await fetch('/api/batch-jobs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    folder_path: folderPath,
                    mode: mode,
                    edit_style: selectedEditStyle,
                    caption_style: selectedCaptionFolder,
                    num_variations: numVariations,
                    clips_per_video: parseInt(document.getElementById('batch-clips-per-video').value, 10) || 1
                })
            });
            if (!response.ok) throw new Error('Failed to create batch job');
            const data = await response.json();
            currentBatchId = data.batch_id;
            pollBatchJob(data.batch_id);
        } catch (error) {
            handleBatchError(error.message);
        }
    });

    function pollBatchJob(batchId) {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/batch-jobs/${batchId}`);
                if (!res.ok) throw new Error('Failed to fetch batch status');
                const batch = await res.json();

                if (batch.status === 'pending') {
                    updateBadge(batchBadge, 'pending', 'Starting...');
                    batchMessage.textContent = batch.stage || 'Initializing...';
                }
                else if (batch.status === 'processing') {
                    updateBadge(batchBadge, 'processing', 'Processing...');
                    batchMessage.textContent = batch.stage || 'Working...';
                    batchProgressBar.classList.remove('indeterminate');
                    batchCancelBtn.classList.remove('hidden');
                    batchCancelBtn.disabled = false;
                    batchCancelBtn.textContent = 'Cancel';

                    // Progress
                    const pct = batch.progress || 0;
                    batchProgressBar.style.width = `${pct}%`;

                    // Per-file status (batch mode only)
                    if (batch.mode !== 'smart_edit' && batch.jobs && batch.jobs.length > 0) {
                        batchJobsContainer.classList.remove('hidden');
                        renderBatchJobs(batch.jobs);
                    }
                }
                else if (batch.status === 'done') {
                    clearInterval(pollInterval);
                    batchCancelBtn.classList.add('hidden');
                    updateBadge(batchBadge, 'done', 'Completed');
                    batchMessage.textContent = batch.stage || 'Done!';
                    batchProgressBar.classList.remove('indeterminate');
                    batchProgressBar.style.width = '100%';
                    batchProgressBar.style.background = '#10b981';

                    if (batch.jobs && batch.jobs.length > 0 && batch.mode !== 'smart_edit') {
                        batchJobsContainer.classList.remove('hidden');
                        renderBatchJobs(batch.jobs);
                    }

                    if (batch.all_clips && batch.all_clips.length > 0) {
                        batchResultsContainer.classList.remove('hidden');
                        batch.all_clips.forEach(clip => {
                            const li = document.createElement('li');
                            li.textContent = clip.split(/[\\/]/).pop();
                            batchClipsList.appendChild(li);
                        });
                    }
                    resetFolderForm();
                }
                else if (batch.status === 'error') {
                    clearInterval(pollInterval);
                    batchCancelBtn.classList.add('hidden');
                    handleBatchError(batch.error || 'Unknown error');
                    if (batch.jobs && batch.jobs.length > 0) {
                        batchJobsContainer.classList.remove('hidden');
                        renderBatchJobs(batch.jobs);
                    }
                }
            } catch (error) {
                clearInterval(pollInterval);
                handleBatchError(error.message);
            }
        }, 2000);
    }

    function renderBatchJobs(jobsList) {
        batchJobsList.innerHTML = '';
        jobsList.forEach(job => {
            const card = document.createElement('div');
            card.className = 'batch-job-item';
            const icon =
                job.status === 'done' ? '✅' :
                job.status === 'error' ? '❌' :
                job.status === 'processing' ? '⏳' : '⏸️';
            const pct = job.status === 'done' ? 100 : (job.progress || 0);
            card.innerHTML = `
                <div class="batch-job-header">
                    <span class="batch-job-icon">${icon}</span>
                    <span class="batch-job-name">${job.file_name || 'unknown'}</span>
                    <span class="badge ${job.status}">${job.status}</span>
                </div>
                <div class="batch-job-progress-container">
                    <div class="batch-job-progress" style="width:${pct}%;background:${
                        job.status === 'done' ? '#10b981' :
                        job.status === 'error' ? '#ef4444' :
                        'linear-gradient(90deg,#3b82f6,#8b5cf6)'}"></div>
                </div>
                ${job.stage ? `<p class="batch-job-stage">${job.stage}</p>` : ''}
                ${job.error ? `<p class="batch-job-error">${job.error}</p>` : ''}
                ${job.clips?.length ? `<p class="batch-job-clips">${job.clips.length} clip(s)</p>` : ''}
            `;
            batchJobsList.appendChild(card);
        });
    }

    function handleBatchError(msg) {
        updateBadge(batchBadge, 'error', 'Error');
        batchMessage.textContent = msg;
        batchProgressBar.classList.remove('indeterminate');
        batchProgressBar.style.width = '100%';
        batchProgressBar.style.background = '#ef4444';
        resetFolderForm();
    }

    function resetFolderForm() {
        folderSubmitBtn.disabled = false;
        folderSubmitBtn.querySelector('.btn-text').textContent = 'Process Folder';
        folderSubmitBtn.querySelector('.spinner').classList.add('hidden');
    }

    // ─── Selector Renderers ──────────────────────────────────────────────────

    function renderCaptionSelector(containerId, styles, activeId, onChange) {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '';

        const previewText = {
            karaoke: 'EACH WORD',
            minimal: 'clean and subtle',
            bold_pop: 'POP IN',
            subtitle_bar: 'Professional subtitles',
            typewriter: 'word by word▌',
        };

        styles.forEach(style => {
            const card = document.createElement('button');
            card.type = 'button';
            card.className = `style-card ${style.id === activeId ? 'active' : ''}`;
            card.innerHTML = `
                <span class="style-card-name">${style.name}</span>
                <span class="style-card-preview" data-style="${style.id}">${previewText[style.id] || style.name}</span>
            `;
            card.addEventListener('click', () => {
                container.querySelectorAll('.style-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                onChange(style.id);
            });
            container.appendChild(card);
        });
    }

    function renderEditStyleSelector(containerId, styles, activeId, onChange) {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '';

        styles.forEach(style => {
            const card = document.createElement('button');
            card.type = 'button';
            card.className = `edit-style-card ${style.id === activeId ? 'active' : ''}`;
            card.innerHTML = `
                <span class="edit-style-icon">${style.icon}</span>
                <span class="edit-style-name">${style.name}</span>
                <span class="edit-style-desc">${style.description}</span>
            `;
            card.addEventListener('click', () => {
                container.querySelectorAll('.edit-style-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                onChange(style.id);
            });
            container.appendChild(card);
        });
    }

    // ─── Shared ──────────────────────────────────────────────────────────────
    function updateBadge(badge, status, text) {
        badge.className = `badge ${status}`;
        badge.textContent = text;
    }
});
