let currentConversationId = null;

// Function to load conversations
async function loadConversations() {
  try {
    const response = await fetch('http://127.0.0.1:8080/conversations');
    const conversations = await response.json();
    const conversationsList = document.getElementById('conversationsList');
    conversationsList.innerHTML = '';

    conversations.forEach(conversationId => {
      const conversationElement = document.createElement('div');
      conversationElement.className = 'conversation-item';
      if (conversationId === currentConversationId) {
        conversationElement.classList.add('active');
      }

      // Create conversation title span
      const titleSpan = document.createElement('span');
      titleSpan.textContent = `Conversation ${conversationId.slice(0, 8)}...`;
      conversationElement.appendChild(titleSpan);

      // Create delete button
      const deleteButton = document.createElement('button');
      deleteButton.className = 'delete-conversation-btn';
      deleteButton.textContent = 'x';
      deleteButton.title = 'Delete conversation';
      deleteButton.onclick = (e) => deleteConversation(e, conversationId);
      conversationElement.appendChild(deleteButton);

      // Add click handler for loading conversation
      conversationElement.onclick = () => loadConversation(conversationId);

      conversationsList.appendChild(conversationElement);
    });
  } catch (error) {
    console.error('Error loading conversations:', error);
  }
}

// Function to load a specific conversation
async function loadConversation(conversationId) {
  try {
    currentConversationId = conversationId;
    const response = await fetch(`http://127.0.0.1:8080/conversations/${conversationId}`);
    const messages = await response.json();

    const outputContainer = document.getElementById('output');
    outputContainer.innerHTML = '';

    messages.forEach(message => {
      const messageDiv = document.createElement('div');
      messageDiv.className = 'message';

      const contentDiv = document.createElement('div');
      contentDiv.className = message.sender.toLowerCase() === 'user' ? 'user-message' : 'ai-message';
      contentDiv.textContent = message.content;

      messageDiv.appendChild(contentDiv);
      outputContainer.appendChild(messageDiv);
    });

    // Update active state in sidebar
    document.querySelectorAll('.conversation-item').forEach(item => {
      item.classList.remove('active');
      if (item.textContent.includes(conversationId.slice(0, 8))) {
        item.classList.add('active');
      }
    });
  } catch (error) {
    console.error('Error loading conversation:', error);
  }
}

// Function to start a new conversation
async function startNewConversation() {
  try {
    const response = await fetch('http://127.0.0.1:8080/conversations/start', {
      method: 'GET'
    });
    const conversationId = await response.json();
    currentConversationId = conversationId;
    document.getElementById('output').innerHTML = '';
    await loadConversations();
    return conversationId;
  } catch (error) {
    console.error('Error starting new conversation:', error);
  }
}

// Add this new function to handle conversation deletion
async function deleteConversation(event, conversationId) {
  event.stopPropagation(); // Prevent conversation selection when clicking delete

  try {
    const response = await fetch(`http://127.0.0.1:8080/conversations/${conversationId}`, {
      method: 'DELETE'
    });

    if (response.ok) {
      if (conversationId === currentConversationId) {
        currentConversationId = null;
        document.getElementById('output').innerHTML = '';
      }
      await loadConversations();
    } else {
      console.error('Failed to delete conversation');
    }
  } catch (error) {
    console.error('Error deleting conversation:', error);
  }
}

// Add this function to get all parameters
function getModelParameters() {
  return {
    sample_len: parseInt(document.getElementById('sampleLen').value),
    temperature: parseFloat(document.getElementById('temperature').value),
    top_p: parseFloat(document.getElementById('topP').value),
    repeat_penalty: parseFloat(document.getElementById('repeatPenalty').value),
    repeat_last_n: parseInt(document.getElementById('repeatLastN').value),
    model: document.getElementById('modelSelect').value,
    revision: document.getElementById('revision').value,
    seed: parseInt(document.getElementById('seed').value)
  };
}

// Initialize the app
document.addEventListener('DOMContentLoaded', async () => {
  // Load existing conversations on page load
  await loadConversations();

  document.getElementById('newConversationBtn').addEventListener('click', startNewConversation);

  document.getElementById("sendButton").addEventListener("click", async () => {
    const sendButton = document.getElementById("sendButton");
    const userInput = document.getElementById("userInput");
    const model = document.getElementById("modelSelect").value;
    const prompt = userInput.value;

    if (!prompt.trim()) return;

    // Start new conversation if none is selected
    if (!currentConversationId) {
      currentConversationId = await startNewConversation();
    }

    sendButton.disabled = true;
    sendButton.style.backgroundColor = '#cccccc';
    userInput.disabled = true;

    const outputContainer = document.getElementById("output");
    const messageDiv = document.createElement("div");
    messageDiv.className = "message";

    const userDiv = document.createElement("div");
    userDiv.className = "user-message";
    userDiv.textContent = prompt;
    messageDiv.appendChild(userDiv);

    const aiDiv = document.createElement("div");
    aiDiv.className = "ai-message";
    messageDiv.appendChild(aiDiv);

    outputContainer.appendChild(messageDiv);
    // Auto-scroll when adding new message
    outputContainer.scrollTop = outputContainer.scrollHeight;

    try {
      const response = await fetch(`http://127.0.0.1:8080/conversations/${currentConversationId}/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt,
          ...getModelParameters()
        }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        aiDiv.textContent += chunk;
        // Auto-scroll to the bottom of the output container
        outputContainer.scrollTop = outputContainer.scrollHeight;
      }
    } catch (error) {
      console.error("Error:", error);
      aiDiv.textContent = "Error: " + error.message;
    } finally {
      sendButton.disabled = false;
      sendButton.style.backgroundColor = '#007bff';
      userInput.disabled = false;
      userInput.value = "";
      userInput.focus();
    }
  });

  // Add Enter key support
  document.getElementById("userInput").addEventListener("keypress", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      document.getElementById("sendButton").click();
    }
  });
});
