document.addEventListener('DOMContentLoaded', () => { // Best practice: run after DOM is loaded

    // --- Element References ---
    const fileInput = document.getElementById('file-input');
    const uploadBox = document.getElementById('upload-box');
    const fileList = document.getElementById('file-list');
    const processBtn = document.getElementById('process-btn');
    const progressBar = document.getElementById('progress');
    const statusText = document.getElementById('status-text');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const answerTypeSelect = document.getElementById('answer-type');
    const exportBtn = document.getElementById('export-btn');
    const exportModal = document.getElementById('export-modal');
    const exportConfirmBtn = document.getElementById('export-confirm-btn');
    const exportCancelBtn = document.getElementById('export-cancel-btn');
    const closeBtn = document.querySelector('.modal .close'); // More specific selector
    const exportForm = document.getElementById('export-form');
    const processingProgressSection = document.getElementById('processing-progress-section');

    // Study Plan Elements
    const studyPlanSection = document.getElementById('study-plan-section');
    const studyPlanList = document.getElementById('study-plan-list');
    const planProgress = document.getElementById('plan-progress');
    const planProgressText = document.getElementById('plan-progress-text');

    // --- State Variables ---
    let filesToUpload = []; // Use let as it will be reassigned
    let processingInterval;
    let chatMessages = [];
    let currentStudyPlan = []; // Store the fetched study plan data {id, text, completed}

    // --- Utility Functions ---
    function sanitizeHTML(str) {
        const temp = document.createElement('div');
        temp.textContent = str;
        return temp.innerHTML;
    }

    function formatAIResponse(text) {
        // Basic formatting: Convert markdown-like elements to HTML
        let html = text;

        // Bold (**text** or __text__)
        html = html.replace(/\*\*(.*?)\*\*|__(.*?)__/g, '<strong>$1$2</strong>');
        // Italics (*text* or _text_)
        html = html.replace(/\*(.*?)\*|_(.*?)_/g, '<em>$1$2</em>');
        // Strikethrough (~~text~~)
        html = html.replace(/~~(.*?)~~/g, '<del>$1</del>');

        // Code blocks (```lang\n code \n``` or ```\n code \n```)
        html = html.replace(/```(\w+)?\s*([\s\S]*?)```/g, (match, lang, code) => {
            const languageClass = lang ? ` class="language-${lang}"` : '';
            const sanitizedCode = sanitizeHTML(code.trim());
            return `<pre><code${languageClass}>${sanitizedCode}</code></pre>`;
        });
         // Inline code (`code`)
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Unordered lists (* item or - item)
        html = html.replace(/^\s*[-*]\s+(.*)/gm, '<li>$1</li>');
        html = html.replace(/<\/li>\n<li>/g, '</li><li>'); // Fix potential spacing issues
        html = html.replace(/<ul>\s*<li>/g, '<ul><li>'); // Remove space after <ul>
        html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>'); // Wrap LIs in ULs if needed

        // Ordered lists (1. item)
        html = html.replace(/^\s*\d+\.\s+(.*)/gm, '<oli>$1</oli>'); // Use temporary <oli>
        html = html.replace(/<\/oli>\n<oli>/g, '</oli><oli>');
        html = html.replace(/<ol>\s*<oli>/g, '<ol><oli>');
        html = html.replace(/(<oli>.*<\/oli>)/s, '<ol>$1</ol>');
        html = html.replace(/<oli>/g, '<li>').replace(/<\/oli>/g, '</li>'); // Replace temp tag

         // Blockquotes (> text) - handle multi-line
        html = html.replace(/^>\s+(.*(?:\n(?:^>\s+.*|\s*))*)/gm, (match, content) => {
            const inner = content.replace(/^>\s?/gm, ''); // Remove leading '>'
            return `<blockquote>${inner.trim()}</blockquote>`;
        });

        // Replace remaining newlines with <p> tags, avoiding lists/pre/blockquote
        // This is complex. A simpler approach might be just <br> or rely on CSS white-space
        // Simplified: Wrap paragraphs separated by double newlines
        const blocks = html.split(/\n{2,}/);
        html = blocks.map(block => {
            // Avoid wrapping tags that are already block-level or list items
            if (block.match(/^<(ul|ol|li|pre|blockquote|h[1-6])/)) {
                return block;
            }
            return `<p>${block.replace(/\n/g, '<br>')}</p>`; // Convert single newlines within paragraph to <br>
        }).join('');

        // Cleanup potentially empty paragraphs
        html = html.replace(/<p>\s*<\/p>/g, '');
        html = html.replace(/<p><br\s*\/?>\s*<\/p>/g, '');

        return html;
    }

    // --- File Handling & Upload ---

    function handleFiles(files) {
         const newFiles = Array.from(files).filter(file => {
            // Basic validation (can be improved)
            const allowedTypes = ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'png', 'jpg', 'jpeg'];
            const extension = file.name.split('.').pop().toLowerCase();
            return allowedTypes.includes(extension);
            // Add size check if needed: && file.size <= MAX_SIZE
         });
         // Maybe add duplicates check later if needed
         filesToUpload = [...newFiles]; // Replace current selection
         updateFileList();
         // Reset progress and status if new files are selected
         resetProcessingState();
         resetStudyPlanUI();
         studyPlanSection.style.display = 'none';
         chatInput.disabled = true;
         sendBtn.disabled = true;
         exportBtn.disabled = true;
    }

     uploadBox.addEventListener('click', () => fileInput.click()); // Make box clickable

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
        // Clear the input value to allow selecting the same file again
        fileInput.value = '';
    });

    function updateFileList() {
        fileList.innerHTML = ''; // Clear existing list
        if (filesToUpload.length === 0) {
            fileList.innerHTML = '<li>No files selected.</li>';
            processBtn.disabled = true;
        } else {
            filesToUpload.forEach(file => {
                const listItem = document.createElement('li');
                const extension = file.name.split('.').pop().toLowerCase();
                let iconClass = 'fa-file'; // Default
                if (['pdf'].includes(extension)) iconClass = 'fa-file-pdf pdf';
                else if (['doc', 'docx'].includes(extension)) iconClass = 'fa-file-word word';
                else if (['xls', 'xlsx'].includes(extension)) iconClass = 'fa-file-excel excel';
                else if (['ppt', 'pptx'].includes(extension)) iconClass = 'fa-file-powerpoint ppt';
                else if (['jpg', 'jpeg', 'png', 'gif'].includes(extension)) iconClass = 'fa-file-image image';

                // Sanitize file name before inserting
                const safeFileName = sanitizeHTML(file.name);
                listItem.innerHTML = `<i class="fas ${iconClass}"></i> ${safeFileName}`;
                listItem.classList.add('file-list-item');
                fileList.appendChild(listItem);
            });
            processBtn.disabled = false;
        }
    }

    function resetProcessingState() {
         if (processingInterval) {
             clearInterval(processingInterval);
             processingInterval = null;
         }
         progressBar.style.width = '0%';
         statusText.textContent = 'Select files and click Process';
         processingProgressSection.style.display = 'block'; // Show progress section
         processBtn.disabled = filesToUpload.length === 0; // Disable if no files
         fileInput.disabled = false; // Re-enable file input
    }

    processBtn.addEventListener('click', () => {
        if (filesToUpload.length === 0) {
            alert('Please select files to process.');
            return;
        }
        uploadFiles();
    });

    function uploadFiles() {
        const formData = new FormData();
        filesToUpload.forEach(file => {
            formData.append('files[]', file);
        });

        // --- UI Changes for Uploading/Processing ---
        fileInput.disabled = true;
        processBtn.disabled = true;
        processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...'; // Indicate activity
        statusText.textContent = 'Uploading files...';
        progressBar.style.width = '0%';
        processingProgressSection.style.display = 'block';
        studyPlanSection.style.display = 'none'; // Hide study plan during processing
        resetStudyPlanUI(); // Clear old plan visually
        chatInput.disabled = true; // Disable chat during processing
        sendBtn.disabled = true;
        exportBtn.disabled = true;
        // Clear previous chat messages? Optional. For now, keep them.
        // chatContainer.innerHTML = '';
        // chatMessages = [];

        fetch('/upload', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                     throw new Error(errData.error || `Upload failed: ${response.status}`);
                }).catch(() => { // Fallback if response is not JSON
                    return response.text().then(text => { throw new Error(`Upload failed: ${response.status} - ${text}`) });
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Upload successful:', data);
            if (data.warnings) {
                // Handle warnings (e.g., skipped files) - maybe show a small notification
                console.warn("Upload Warnings:", data.warnings);
            }
            statusText.textContent = 'Upload complete. Starting analysis...';
            startProcessingStatusCheck(); // Start polling for processing status
        })
        .catch(error => {
            console.error('Error uploading files:', error);
            statusText.textContent = `Upload Error: ${error.message}`;
            progressBar.style.width = '0%';
            // Re-enable relevant controls
            fileInput.disabled = false;
            processBtn.disabled = false; // Allow retry
             processBtn.innerHTML = '<i class="fas fa-cogs"></i> Process Documents';
        });
    }

    // --- Processing Status Check ---
    function startProcessingStatusCheck() {
        if (processingInterval) {
            clearInterval(processingInterval); // Clear existing interval if any
        }

        processingInterval = setInterval(() => {
            fetch('/status')
            .then(response => {
                if (response.status === 404) { // Handle session not found explicitly
                     throw new Error('Session expired or not found. Please reload the page.');
                }
                if (!response.ok) {
                    throw new Error(`Status check failed: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Processing status:', data);
                progressBar.style.width = `${data.progress || 0}%`;
                statusText.textContent = data.message || 'Checking status...';

                if (data.status === 'completed') {
                    clearInterval(processingInterval);
                    processingInterval = null;
                    statusText.textContent = 'Analysis Complete!';
                    progressBar.style.backgroundColor = 'var(--success-color)'; // Green on success
                    // Enable Chat
                    chatInput.disabled = false;
                    sendBtn.disabled = false;
                    exportBtn.disabled = chatMessages.length === 0; // Enable export if messages exist
                    // Fetch and display the study plan
                    fetchAndDisplayStudyPlan();
                    fileInput.disabled = false; // Re-enable file input for changing files
                    processBtn.disabled = false; // Allow re-processing if needed
                    processBtn.innerHTML = '<i class="fas fa-redo"></i> Re-process Files'; // Change button text

                } else if (data.status === 'failed') {
                    clearInterval(processingInterval);
                    processingInterval = null;
                    statusText.textContent = `Error: ${data.message}`;
                    progressBar.style.width = '100%'; // Show full bar for error
                    progressBar.style.backgroundColor = 'var(--error-color)'; // Red bar on failure
                    fileInput.disabled = false;
                    processBtn.disabled = false; // Allow retry
                    processBtn.innerHTML = '<i class="fas fa-cogs"></i> Process Documents';

                } else if (data.status === 'processing') {
                     // Still processing, keep polling
                     progressBar.style.backgroundColor = 'var(--primary-color)'; // Normal blue during processing
                } else if (data.status === 'no_session' || data.status === 'idle' || data.status === 'unknown') {
                     // Handle states where processing hasn't started or session lost
                     clearInterval(processingInterval);
                     processingInterval = null;
                     resetProcessingState(); // Reset to initial state
                     statusText.textContent = data.message || 'Ready to process.';
                     if (data.status === 'no_session') {
                         alert('Session lost. Please reload the page.');
                     }
                }
            })
            .catch(error => {
                console.error('Error checking processing status:', error);
                clearInterval(processingInterval);
                processingInterval = null;
                statusText.textContent = `Status Error: ${error.message}`;
                progressBar.style.backgroundColor = 'var(--error-color)';
                fileInput.disabled = false;
                processBtn.disabled = filesToUpload.length === 0;
                processBtn.innerHTML = '<i class="fas fa-cogs"></i> Process Documents';
                studyPlanSection.style.display = 'none';
            });
        }, 1500); // Check every 1.5 seconds
    }

    // --- Chat Functionality ---

    // Auto-resize textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto'; // Reset height
        chatInput.style.height = `${chatInput.scrollHeight}px`; // Set to content height
    });

     chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent newline
            sendChatMessage();
        }
    });

    sendBtn.addEventListener('click', sendChatMessage);

    function sendChatMessage() {
        const question = chatInput.value.trim();
        const answerType = answerTypeSelect.value;

        if (question === '' || chatInput.disabled) {
            return;
        }

        addMessage('user', question); // Display user message immediately
        chatInput.value = ''; // Clear input
        chatInput.style.height = 'auto'; // Reset height
        chatInput.disabled = true;
        sendBtn.disabled = true;
        // Optionally add a temporary "AI is thinking..." message
        addMessage('ai', '<i class="fas fa-spinner fa-spin"></i> Thinking...', true); // Add thinking indicator


        fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question, answer_type: answerType })
        })
        .then(response => {
             if (!response.ok) {
                return response.json().then(errData => {
                     throw new Error(errData.error || `Query failed: ${response.status}`);
                });
             }
             return response.json();
        })
        .then(data => {
            console.log('AI raw response:', data.answer);
            // Remove thinking message and add actual response
            removeLastMessageIfThinking();
            addMessage('ai', data.answer);
        })
        .catch(error => {
            console.error('Error querying documents:', error);
            removeLastMessageIfThinking(); // Remove thinking message on error too
            addMessage('system', `Error: ${error.message}`);
        })
        .finally(() => {
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.focus();
            exportBtn.disabled = chatMessages.length === 0; // Enable export if messages exist
        });
    }

    function addMessage(role, content, isThinking = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${role}-message`);
        if (isThinking) {
            messageDiv.classList.add('thinking'); // Add class for potential removal
        }

        const messageContentDiv = document.createElement('div');
        messageContentDiv.classList.add('message-content');

        // Sanitize user/system input, format AI response
        if (role === 'user' || role === 'system') {
            messageContentDiv.innerHTML = `<p>${sanitizeHTML(content)}</p>`; // Basic paragraph wrapping
        } else if (role === 'ai') {
            if (isThinking) {
                 messageContentDiv.innerHTML = content; // Allow HTML for spinner
            } else {
                messageContentDiv.innerHTML = formatAIResponse(content); // Format AI response
            }
        }

        messageDiv.appendChild(messageContentDiv);
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight; // Scroll to bottom

        // Store message for export (use raw content for AI, sanitized for others)
        if (!isThinking) {
            const storeContent = (role === 'ai') ? content : sanitizeHTML(content);
             chatMessages.push({ role: role, content: storeContent });
             exportBtn.disabled = false; // Enable export once a message is added
        }
    }

     function removeLastMessageIfThinking() {
        const lastMessage = chatContainer.querySelector('.message.thinking');
        if (lastMessage) {
            lastMessage.remove();
        }
    }


    // --- Study Plan Functionality ---

    function resetStudyPlanUI() {
        studyPlanList.innerHTML = '<li>Process documents to generate plan.</li>';
        planProgress.style.width = '0%';
        planProgressText.textContent = '0% Complete';
        currentStudyPlan = [];
        // studyPlanSection.style.display = 'none'; // Don't hide, just reset content
    }

    function fetchAndDisplayStudyPlan() {
        console.log("Fetching study plan...");
        studyPlanSection.style.display = 'block'; // Ensure section is visible
        studyPlanList.innerHTML = '<li><i class="fas fa-spinner fa-spin"></i> Loading study plan...</li>'; // Loading indicator

        fetch('/study_plan')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to fetch study plan: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Study plan received:", data);
                currentStudyPlan = data.study_plan || []; // Store the plan
                renderStudyPlan(currentStudyPlan);
                updateStudyProgress(currentStudyPlan);
            })
            .catch(error => {
                console.error("Error fetching study plan:", error);
                studyPlanList.innerHTML = `<li>Error loading plan: ${error.message}</li>`;
                planProgress.style.width = '0%';
                planProgressText.textContent = 'Error';
            });
    }

    function renderStudyPlan(planData) {
        studyPlanList.innerHTML = ''; // Clear previous items or loading message

        if (!planData || planData.length === 0) {
            studyPlanList.innerHTML = '<li>No study topics found or generated.</li>';
            updateStudyProgress([]); // Ensure progress resets if no topics
            return;
        }

        planData.forEach(topic => {
            if (!topic || !topic.id || typeof topic.text !== 'string') {
                 console.warn("Skipping invalid topic data:", topic);
                 return; // Skip malformed topic entries
            }

            const li = document.createElement('li');
            li.dataset.topicId = topic.id; // Store ID on the element

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = topic.completed || false; // Default to false if missing
            checkbox.id = `topic-check-${topic.id}`; // Unique ID for label association
            // Add listener directly to checkbox
            checkbox.addEventListener('change', handleTopicToggle);

            const label = document.createElement('label');
            label.htmlFor = checkbox.id; // Associate label with checkbox
            label.textContent = topic.text;

            if (checkbox.checked) {
                li.classList.add('completed');
            }

            li.appendChild(checkbox);
            li.appendChild(label);
            studyPlanList.appendChild(li);
        });
    }

    function handleTopicToggle(event) {
        const checkbox = event.target;
        const li = checkbox.closest('li'); // Find the parent list item
        if (!li || !li.dataset.topicId) return; // Safety check

        const topicId = li.dataset.topicId;
        const isCompleted = checkbox.checked;

        // 1. Update local state (find the topic and update its 'completed' status)
        const topicIndex = currentStudyPlan.findIndex(t => t.id === topicId);
        if (topicIndex > -1) {
            currentStudyPlan[topicIndex].completed = isCompleted;
            console.log(`Local state updated for ${topicId}: ${isCompleted}`);
        } else {
            console.warn(`Topic ID ${topicId} not found in local currentStudyPlan`);
            return; // Don't proceed if topic not found locally
        }

        // 2. Update UI immediately (add/remove 'completed' class)
        li.classList.toggle('completed', isCompleted);

        // 3. Update progress bar
        updateStudyProgress(currentStudyPlan);

        // 4. Send update to backend
        fetch('/update_topic_status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic_id: topicId, completed: isCompleted })
        })
        .then(response => response.json()) // Assume response is always JSON
        .then(data => {
            if (!data.success) {
                 console.error(`Failed to update topic ${topicId} status on server: ${data.error || 'Unknown error'}`);
                 // Optional: Revert UI changes or show user error
                 // checkbox.checked = !isCompleted; // Revert checkbox
                 // li.classList.toggle('completed', !isCompleted);
                 // updateStudyProgress(currentStudyPlan); // Update progress back
                 // alert(`Error saving progress for topic: ${topicId}`);
            } else {
                 console.log(`Topic ${topicId} status saved successfully on server.`);
            }
        })
        .catch(error => {
            console.error(`Network or server error updating topic ${topicId} status:`, error);
            // Optional: Revert UI changes or show user error
            // alert(`Network error saving progress for topic: ${topicId}`);
        });
    }

    function updateStudyProgress(planData) {
        if (!planData || planData.length === 0) {
            planProgress.style.width = '0%';
            planProgressText.textContent = '0% Complete';
            return;
        }

        const totalTopics = planData.length;
        const completedTopics = planData.filter(topic => topic.completed).length;
        const percentage = totalTopics > 0 ? Math.round((completedTopics / totalTopics) * 100) : 0;

        planProgress.style.width = `${percentage}%`;
        planProgressText.textContent = `${percentage}% Complete (${completedTopics}/${totalTopics})`;
    }


    // --- Export Chat Functionality ---

    exportBtn.addEventListener('click', () => {
        if (chatMessages.length > 0) {
           exportModal.style.display = 'flex'; // Show modal
        } else {
           alert("No chat messages to export yet.");
        }
    });

    closeBtn.addEventListener('click', () => exportModal.style.display = 'none');
    exportCancelBtn.addEventListener('click', () => exportModal.style.display = 'none');

    // Handle form submission for export
    exportForm.addEventListener('submit', (e) => {
        e.preventDefault(); // Prevent default form submission
        handleExportConfirm();
    });

    function handleExportConfirm() {
        const format = document.querySelector('input[name="export-format"]:checked').value;
        const confirmButton = document.getElementById('export-confirm-btn');
        confirmButton.disabled = true;
        confirmButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Exporting...';

        fetch('/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: chatMessages, format: format })
        })
        .then(response => {
            if (!response.ok) {
                 return response.json().then(err => { throw new Error(err.error || `Export failed: ${response.status}`) });
            }
            return response.json();
        })
        .then(data => {
            downloadFile(data.content, data.filename);
            exportModal.style.display = 'none'; // Hide modal on success
        })
        .catch(error => {
            console.error('Error exporting chat:', error);
            alert(`Error exporting chat: ${error.message}`);
        })
        .finally(() => {
             confirmButton.disabled = false; // Re-enable button
             confirmButton.innerHTML = '<i class="fas fa-file-export"></i> Export';
        });
    }

    function downloadFile(content, filename) {
        const element = document.createElement('a');
        let mimeType = 'text/plain';
        if (filename.endsWith('.html')) mimeType = 'text/html';
        else if (filename.endsWith('.md')) mimeType = 'text/markdown'; // Common practice, though no official standard

        const blob = new Blob([content], { type: `${mimeType};charset=utf-8` });
        const url = URL.createObjectURL(blob);

        element.setAttribute('href', url);
        element.setAttribute('download', filename);
        element.style.display = 'none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
        URL.revokeObjectURL(url); // Clean up blob URL
    }

    // --- Initial Setup on Page Load ---
    updateFileList(); // Show "No files selected" initially
    resetProcessingState(); // Set initial processing state UI
    resetStudyPlanUI(); // Set initial study plan UI

}); // End of DOMContentLoaded