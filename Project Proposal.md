# Project Proposal: Large Language Model Serving

## Motivation

The growing demand for large language models (LLMs) across industries—from customer service chatbots to AI-driven content generation—has created the need for efficient, scalable back-end services. We aim to build a platform capable of serving multiple large language models with support for streaming, enhancing user experience by reducing the need to switch between models. Instead of delivering results all at once after the model completes its response, we plan to implement result streaming for an even smoother experience. Our team’s goal is to build a robust, efficient back-end inference service that leverages Rust’s performance and safety.

## Objective and Key Features

The primary objective of this project is to design and implement a back-end inference service using Rust that can efficiently serve large language models (LLMs) with support for streaming inference results. By building this project, we aim to address the performance, scalability, and user-experience challenges often associated with serving LLMs.

### Key Features

- **Multi-Model Management**: Load and manage multiple large language models within a single service, allowing clients to choose the model for each inference request.
- **API Endpoints**: Provide RESTful API endpoints for clients to interact with the service, making it easy to integrate the system into existing infrastructures.
- **Streaming Support**: Streaming capabilities for inference results, improving latency for long responses.
- **Chat Interface**: A simple, client-side chat interface to test interactions with the back-end service.

## Tentative Plan

### Week 1: Setup and Initial Design

- **Task**: Research and finalize the technologies to be used (Candle vs. Mistral.rs, Rocket vs. Actix Web).
- **Team Member 1**: Research Candle and Mistral.rs, compare performance, ease of use, and compatibility with multiple models.
- **Team Member 2**: Explore Rocket and Actix Web, evaluating their support for high-concurrency requests and streaming capabilities.

### Week 2: Core Backend Development

- **Task**: Implement basic API endpoints and set up the model loading and management logic.
- **Team Member 1**: Implement the backend service using the chosen web framework (either Rocket or Actix Web).
- **Team Member 2**: Design and implement the API endpoints, focusing on the ability to load and serve multiple LLMs.

### Week 3: Implement Model Inference

- **Task**: Integrate Candle or Mistral.rs for model inference.
- **Team Member 1**: Focus on integrating the inference engine into the backend, ensuring that it can handle requests for multiple models.
- **Team Member 2**: Ensure that the API endpoints correctly interface with the inference engine, returning appropriate responses.

### Week 4: Implement Streaming Capabilities

- **Task**: Add support for streaming inference results to clients.
- **Team Member 1**: Implement the streaming feature for inference, focusing on efficient handling of long responses.
- **Team Member 2**: Test and refine the streaming feature to ensure smooth real-time interaction for users.

### Week 5: Build Basic Chat Interface and Final Testing

- **Task**: Create a basic web-based chat interface that interacts with the back-end service and allows users to send queries and receive streamed responses.
- **Team Member 1**: Build the front-end chat interface and connect it to the backend API.
- **Team Member 2**: Perform final testing of the entire system, focusing on performance, concurrency, and user experience.

### Week 6: Documentation and Final Submission

- **Task**: Complete project documentation and ensure that all files are ready for submission.
- **Both Team Members**: Write and finalize documentation, record the demo video, ensuring all key aspects of the project are explained and documented clearly.
