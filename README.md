# Final Report

* Qi Zhang 1010190106 <qqi.zhang@mail.utoronto.ca>
* Shidi Zhao 1003648378 <shidi.zhao@mail.utoronto.ca>

## Video Demo

<https://youtu.be/rVQOfHuLHKM>

## Motivation

As industries increasingly use large language models (LLMs) for a variety of applications, from customer support chatbots to AI-driven content creation, there is a need for efficient, scalable backend infrastructure to support these models. This project aims to develop a platform that serves multiple LLMs, with streaming capabilities to enhance the user experience by reducing the waiting time for responses. Instead of waiting for an entire response, users will receive results in real-time, improving interactivity. Our team plans to build a robust, efficient back-end inference service that leverages Rust’s performance and safety.

## Objectives

The main objective is to design and implement a backend inference service in Rust to serve large language models efficiently.

## Features

Our project offers streaming support for inference results, a sqlite DB to store conversation history, simple interface for easy interaction and API endpoints for clients.

⚠️Currently, our project only supports MacOS with [Metal](https://support.apple.com/en-ca/102894) support and Windows with CUDA support (Default supports Mac).
To run on Windows machines, change the setting in code to use cuda:

```diff
- let device = candle_core::Device::new_metal(0).unwrap();
+ let device = candle_core::Device::new_cuda(0).unwrap();
```

and in Cargo.toml, change the features to cuda:

```diff
- candle-nn = { git = "https://github.com/huggingface/candle.git", features = ["metal"]  }
- candle-core = { git = "https://github.com/huggingface/candle.git", features = ["metal"] }
- candle-transformers = { git = "https://github.com/huggingface/candle.git", features = ["metal"]  }
+ candle-nn = { git = "https://github.com/huggingface/candle.git", features = ["cuda"]  }
+ candle-core = { git = "https://github.com/huggingface/candle.git", features = ["cuda"] }
+ candle-transformers = { git = "https://github.com/huggingface/candle.git", features = ["cuda"]  }
```

## Reproducibility Guide

### Prerequiste

* Make sure the Rust environment is properly set up on your machine.
* Ensure SQLite is installed on your Mac machine.

### Detailed steps

1. Clone the repository to your local Mac machine:

   ```bash
   git clone https://github.com/Large-Language-Model-Serving/Large-Language-Model-Serving.git
   ```

2. Navigate to the backend folder:

   ```bash
   cd backend
   ```

3. Build the backend in release mode:

   ```bash
   cargo build --release
   ```

4. Start the backend server:

   ```bash
   ./target/release/back-end-inference-service
   ```

5. Open the frontend page in your default web browser:

   ```plaintext
   frontend/index.html
   ```

## User’s Guide

⚠️ When using a model for the first time, it would take some time to download and build the model. This problem will not occur after the first use.

### Interface

Here is the basic chat inferface for our model.
![image](https://github.com/user-attachments/assets/7e366339-05e5-44fe-8fb8-bfde0a0c78d0)
To start conversation, click the "New Conversation" button on the top left corner.
Then, you can select the model you prefer and enter the prompt below, you can also change the parameters to the right.
![image](https://github.com/user-attachments/assets/1e4a191e-6d17-4792-8e59-79f0ef35121f)
After the conversation, you can delete the conversation by clicking the delete button, or leave it there for viewing it next time.

### API

Currently, we support the following API endpoints.

```bash
#[get("/conversations")]
```

It returns all conversation ids in the database.

```bash
#[get("/conversations/start")]
```

It creates a new conversation and returns its id.

```bash
#[get("/conversations/{conversation_id}")]
```

It retrieves the conversation history specified by id.

```bash
#[delete("/conversations/{conversation_id}")]
```

It deletes the conversation specified by id.

```bash
#[post("/conversations/{conversation_id}/generate")]
```

It posts a request in json format like the following, and gets a stream of words generated by the model.

```bash
    {
        "prompt": "Hello, world!",
        "sample_len": 300,
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.2,
        "repeat_last_n": 50,
        "model": "Qwen0_5b",
        "revision": "main",
        "seed": 42
    }
```

## Contributions by each team member

### Qi

* Implemented API endpoints
* Implemented inference result streaming
* Implemented conversation history storage
* Implemented large language model management.

### Shidi

* Investigated and decided the framework to use
* Set up the basic code for one model inferencing
* Implemented the front-end for all features
