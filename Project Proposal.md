# Course Project Proposal: Large Language Model Serving

Qi Zhang 1010190106

## Motivation

As industries increasingly use large language models (LLMs) for a variety of applications, from customer support chatbots to AI-driven content creation, there is a need for efficient, scalable backend infrastructure to support these models. This project aims to develop a platform that serves multiple LLMs, with streaming capabilities to enhance the user experience by reducing the waiting time for responses. Instead of waiting for an entire response, users will receive results in real-time, improving interactivity. Our team plans to build a robust, efficient back-end inference service that leverages Rust’s performance and safety.

## Objective and Key Features

The main objective is to design and implement a backend inference service in Rust to serve large language models efficiently, with support for real-time streaming. By building this project, we aim to address the performance, scalability, and user-experience challenges often associated with serving LLMs.

### Key Features

- **Multi-Model Management**: Load and manage multiple large language models within a single service, allowing clients to choose the model for each inference request.
- **API Endpoints**: Provide RESTful API endpoints for clients to interact with the service, making it easy to integrate the system into existing infrastructures.
- **Streaming Support**: Streaming capabilities for inference results, reducing latency for long responses.
- **Chat Interface**: A simple, client-side chat interface to test interactions with the back-end service.

## Tentative Plan

### Week 1: Setup and Initial Design

- **Task**: Research and finalize the technology stack (e.g., Candle vs. Mistral.rs for inference, Rocket vs. Actix Web for the web framework).
- **Team Member 1**: Evaluate Candle and Mistral.rs, focusing on performance, ease of use, and compatibility.
- **Qi**: Analyze Rocket and Actix Web for high-concurrency handling and streaming support.

### Week 2: Core Backend Development

- **Task**: Develop the basic API endpoints and set up the model loading and management system.
- **Team Member 1**: Build the backend service using the chosen framework (either Rocket or Actix Web).
- **Qi**: Develop the API endpoints, focusing on the multi-model management functionality.

### Week 3: Implement Model Inference

- **Task**: Integrate the chosen inference engine (Candle or Mistral.rs).
- **Team Member 1**: Integrate the inference engine with the backend, ensuring it supports multiple models.
- **Qi**: Ensure API endpoints connect properly with the inference engine, delivering appropriate responses.

### Week 4: Implement Streaming Capabilities

- **Task**: Add real-time streaming support for inference responses.
- **Team Member 1**:  Implement the streaming feature, optimizing it for long responses.
- **Qi**: Test and refine streaming to ensure smooth user interactions.

### Week 5: Build Basic Chat Interface and Final Testing

- **Task**: Create a basic web-based chat interface for user interactions with the backend, receiving responses via streaming.
- **Team Member 1**: Build the front-end chat interface and connect it to the backend API.
- **Qi**: Perform final testing of the entire system, focusing on performance, concurrency, and user experience.

### Week 6: Documentation and Final Submission

- **Task**: Complete project documentation and ensure that all files are ready for submission.
- **Both Team Members**: Finalize documentation, create a demo video, and ensure all project aspects are documented.
