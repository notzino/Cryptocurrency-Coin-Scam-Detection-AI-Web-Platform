document.addEventListener('DOMContentLoaded', function() {
    const predictForm = document.getElementById('predictForm');
    const predictButton = document.getElementById('predictButton');
    const symbolInput = document.querySelector('input[name="symbol"]');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const progressBar = document.getElementById('progressBar');
    const progress = document.getElementById('progress');
    const circleBlock = document.getElementById('circleBlock');
    const confidenceCircle = document.getElementById('confidenceCircle');
    const confidenceCirclePath = document.getElementById('confidenceCirclePath');
    const confidenceText = document.getElementById('confidenceText');
    const coinName = document.getElementById('coinName');

    let csrfToken = '';

    const socket = io('http://127.0.0.1:8000/', {
        withCredentials: true,
        transports: ['polling', 'websocket'],
        extraHeaders: {
            "X-CSRF-Token": csrfToken,
        }
    });

    let formDisabled = false;
    let checkStatusInterval;
    let backoffInterval = 5000;
    let lastProgress = 0;

    socket.on('connect', function() {
        console.log('Socket connected, ID:', socket.id);
        fetch('/restore_state', {
            headers: {
                'X-CSRF-Token': csrfToken
            }
        })
        .then(response => response.json())
        .then(data => {
            csrfToken = data.csrf_token;
            handlePredictionState(data);
        })
        .catch(error => {
            console.error('Error fetching restore state:', error);
        });
    });

    socket.on('progress', function(data) {
        console.log('Progress update:', data);
        if (data && typeof data.progress === 'number') {
            updateProgress(data.progress);
        } else {
            console.error('Invalid progress data received:', data);
        }
    });

    socket.on('prediction_result', function(data) {
        console.log('Prediction result received:', data);
        updateResult(data);
        updateProgress(100);
        enableForm();
    });

    socket.on('disconnect', function() {
        console.log('Socket disconnected');
    });

    function updateResult(data) {
        console.log("Updating result with data:", data);
        let resultClass = '';
        let resultMessage = '';

        coinName.innerText = data.coin_name ? data.coin_name.replace(" USD", "") : '';
        coinName.classList.remove('hidden');

        if (data.status === 'error') {
            resultClass = 'error';
            resultMessage = data.message;
            coinName.classList.add('hidden');
        } else if (data.status === 'limited_data' || data.status === 'scam') {
            resultClass = 'scam';
            resultMessage = `${data.message} (Confidence: ${data.confidence}%)`;
            coinName.classList.add('scam');
            coinName.classList.remove('legit');
        } else if (data.status === 'legit') {
            resultClass = 'legit';
            resultMessage = `${data.message} (Confidence: ${data.confidence}%)`;
            coinName.classList.add('legit');
            coinName.classList.remove('scam');
        }

        result.className = resultClass;
        result.innerText = resultMessage;
        result.classList.remove('hidden');

        if (data.status !== 'error') {
            confidenceText.innerHTML = `${data.confidence}%`;
            const strokeDashArrayValue = `${data.confidence}, 100`;
            confidenceCirclePath.setAttribute('stroke-dasharray', strokeDashArrayValue);
            confidenceCircle.style.display = 'none';
            confidenceCircle.offsetHeight; // Trigger a reflow
            confidenceCircle.style.display = 'block';
            circleBlock.classList.remove('hidden');
            confidenceCircle.classList.remove('hidden');
        }
    }

    function resetProgress() {
        progress.style.width = '0%';
        progress.innerText = '0%';
        lastProgress = 0;
    }

    function updateProgress(value) {
        console.log(`Received progress: ${value}`);
        if (typeof value === 'number' && value >= 0 && value <= 100 && value >= lastProgress) {
            progress.style.width = `${value}%`;
            progress.innerText = `${value}%`;
            lastProgress = value; // Update the last known progress
            console.log(`Progress bar updated to: ${value}%`);
        }
    }

    function disableForm() {
        if (formDisabled) return;
        console.log("Disabling form");
        formDisabled = true;
        symbolInput.disabled = true;
        predictButton.disabled = true;
        loading.classList.remove('hidden');
        progressBar.classList.remove('hidden');
        result.classList.add('hidden');
        circleBlock.classList.add('hidden');
        coinName.classList.add('hidden');
    }

    function enableForm() {
        if (!formDisabled) return;
        console.log("Enabling form");
        formDisabled = false;
        symbolInput.disabled = false;
        predictButton.disabled = false;
        loading.classList.add('hidden');
        progressBar.classList.add('hidden');
        resetProgress();
    }

    function handlePredictionState(data) {
        console.log("Handling prediction state with data:", data);
        csrfToken = data.csrf_token;
        if (data.task_status === 'in_progress') {
            disableForm();
            updateProgress(data.progress || lastProgress);
            if (!checkStatusInterval) {
                checkStatusInterval = setInterval(checkSessionStatus, backoffInterval);
            }
        } else if (data.task_status === 'done') {
            enableForm();
            if (checkStatusInterval) {
                clearInterval(checkStatusInterval);
                checkStatusInterval = null;
            }
            if (data.last_result) {
                updateResult(data.last_result);
            }
        } else {
            enableForm();
            if (checkStatusInterval) {
                clearInterval(checkStatusInterval);
                checkStatusInterval = null;
            }
        }
    }

    function checkSessionStatus() {
        fetch('/restore_state', {
            headers: {
                'X-CSRF-Token': csrfToken
            }
        })
        .then(response => response.json())
        .then(data => {
            console.log('Checked session status:', data);
            handlePredictionState(data);
            backoffInterval = 5000;
        })
        .catch(error => {
            console.error('Error checking session status:', error);
            if (error.message === 'Too Many Requests') {
                console.error('Too many requests, backing off.');
                backoffInterval = Math.min(backoffInterval * 2, 60000);
                clearInterval(checkStatusInterval);
                checkStatusInterval = setInterval(checkSessionStatus, backoffInterval);
            }
        });
    }

    predictForm.addEventListener('submit', function(e) {
        e.preventDefault();
        let symbol = e.target.symbol.value.trim().toUpperCase();
        if (symbol.endsWith('-USD')) {
            symbol = symbol.slice(0, -4);
        }

        console.log('Form submitted:', symbol);

        if (formDisabled) {
            console.log('Form is disabled, aborting.');
            return;
        }

        disableForm();
        resetProgress();

        console.log('Sending POST request to /predict');

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': csrfToken
            },
            body: JSON.stringify({ symbol: symbol, session_id: socket.id })
        })
        .then(response => {
            if (response.status === 429) {
                return response.text().then(text => { throw new Error('Too Many Requests: Please wait before trying again.') });
            }
            if (!response.ok) {
                return response.text().then(text => { throw new Error(text) });
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'error') {
                console.error('Prediction error:', data.message);
                enableForm();
                result.classList.remove('hidden');
                result.className = 'error';
                result.innerText = data.message;
            } else if (data.status === 'pending') {
                console.log('Prediction has been enqueued');
                result.classList.remove('hidden');
                result.className = 'pending';
                result.innerText = 'Prediction has been enqueued. Please wait...';
            }
        })
        .catch(error => {
            console.error('Error during prediction request:', error);
            enableForm();
            result.classList.remove('hidden');
            result.className = 'error';
            result.innerText = `An error occurred: ${error.message}`;
        });
    });
});
