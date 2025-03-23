const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const chatbox = document.querySelector(".chatbox");

let userMessage = ""; // Store the user's message
const API_KEY = "AIzaSyCYW3ppYU2CTQpmpXJ2zsLBMNhEpP1VdqM"; // Your Gemini API Key
const inputInitHeight = chatInput.scrollHeight;

// Function to create a chat <li> element with the given message and class name
const createChatLi = (message, className) => {
  const chatLi = document.createElement("li");
  chatLi.classList.add("chat", className);

  // If the class is 'outgoing', just a <p> tag, else add bot icon
  const chatContent = className === "outgoing"
    ? `<p></p>`
    : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;

  chatLi.innerHTML = chatContent;
  chatLi.querySelector("p").textContent = message;
  return chatLi;
};

// Function to generate bot response using Google Gemini API
const generateResponse = (incomingChatli) => {
  const messageElement = incomingChatli.querySelector("p");

  const requestOptions = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: userMessage }] }],
    }),
  };

  fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyCYW3ppYU2CTQpmpXJ2zsLBMNhEpP1VdqM`, requestOptions)
    .then((res) => res.json())
    .then((data) => {
      if (data && data.candidates && data.candidates[0].content.parts[0].text) {
        messageElement.textContent = data.candidates[0].content.parts[0].text.trim();
      } else {
        messageElement.classList.add("error");
        messageElement.textContent = "Sorry, I couldn't process that.";
      }
    })
    .catch((err) => {
      console.error(err);
      messageElement.classList.add("error");
      messageElement.textContent = "Oops! Something went wrong. Try again.";
    })
    .finally(() => {
      chatbox.scrollTo(0, chatbox.scrollHeight);
    });
};

// Function to handle user input and trigger chat
const handleChat = () => {
  userMessage = chatInput.value.trim(); // Get the user's message

  if (!userMessage) return; // Don't send empty messages

  // Clear input and reset height
  chatInput.value = "";
  chatInput.style.height = `${inputInitHeight}px`;

  // Append the user's message as an outgoing message
  const outgoingChatli = createChatLi(userMessage, "outgoing");
  chatbox.appendChild(outgoingChatli);
  chatbox.scrollTo(0, chatbox.scrollHeight);

  // Simulate a typing response from the bot
  setTimeout(() => {
    const incomingChatli = createChatLi("Typing...", "incoming");
    chatbox.appendChild(incomingChatli);
    generateResponse(incomingChatli);
  }, 600);
};

// Event listener to adjust input area height dynamically based on content
chatInput.addEventListener("input", () => {
  chatInput.style.height = `${inputInitHeight}px`; // Reset to initial height
  chatInput.style.height = `${chatInput.scrollHeight}px`; // Adjust to content height
});

// Event listener to send message when Enter key is pressed (without Shift) on large screens
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
    e.preventDefault();
    handleChat();
  }
});

// Send message when the send button is clicked
sendChatBtn.addEventListener("click", handleChat);

// Close the chatbot when the close button is clicked
closeBtn.addEventListener("click", () => {
  document.body.classList.remove("show-chatbot");
});

// Toggle chatbot visibility when the chatbot toggler is clicked
chatbotToggler.addEventListener("click", () => {
  document.body.classList.toggle("show-chatbot");
});
document.addEventListener("DOMContentLoaded", () => {
  const chatInput = document.querySelector(".chat-input textarea");
  const sendChatBtn = document.querySelector("#send-btn");
  const chatbox = document.querySelector(".chatbox");

  const createChatLi = (message, className) => {
      const chatLi = document.createElement("li");
      chatLi.classList.add("chat", className);
      chatLi.innerHTML = `<p>${message}</p>`;
      return chatLi;
  };

  const handleChat = async () => {
      let userMessage = chatInput.value.trim();
      if (!userMessage) return;

      chatInput.value = "";
      chatbox.appendChild(createChatLi(userMessage, "outgoing"));
      chatbox.scrollTo(0, chatbox.scrollHeight);

      try {
          const response = await fetch("/get_response", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ message: userMessage })
          });

          const data = await response.json();
          chatbox.appendChild(createChatLi(data.response, "incoming"));
      } catch (error) {
          console.error("Error:", error);
          chatbox.appendChild(createChatLi("Error: Unable to fetch response", "incoming"));
      }
  };

  sendChatBtn.addEventListener("click", handleChat);
  chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          handleChat();
      }
  });
});
