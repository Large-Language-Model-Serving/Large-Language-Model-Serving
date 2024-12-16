use anyhow::{Error as E, Result};
use candle_transformers::models::gemma::Model as ModelGemma;
use candle_transformers::models::gemma2::Model as ModelGemma2;
use candle_transformers::models::qwen2::ModelForCausalLM as ModelQwen;
use candle_transformers::models::qwen2_moe::Model as ModelQwenMoe;
use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use sqlx::SqlitePool;
use actix_web::web;
use tokio::sync::mpsc::{self, Sender, Receiver};

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize, Default)]
pub enum Which {
    // Gemma
    #[value(name = "gemma-2b")]
    Gemma2B,
    #[value(name = "gemma-7b")]
    Gemma7B,
    #[value(name = "gemma-2b-it")]
    GemmaInstruct2B,
    #[value(name = "gemma-7b-it")]
    GemmaInstruct7B,
    #[value(name = "gemma-1.1-2b-it")]
    GemmaInstructV1_1_2B,
    #[value(name = "gemma-1.1-7b-it")]
    GemmaInstructV1_1_7B,
    #[value(name = "gemma-code-2b")]
    GemmaCode2B,
    #[value(name = "gemma-code-7b")]
    GemmaCode7B,
    #[value(name = "gemma-code-2b-it")]
    GemmaCodeInstruct2B,
    #[value(name = "gemma-code-7b-it")]
    GemmaCodeInstruct7B,
    #[value(name = "gemma2-2b")]
    Gemma2_2B,
    #[value(name = "gemma2-2b-it")]
    Gemma2Instruct2B,
    #[value(name = "gemma2-9b")]
    Gemma2_9B,
    #[value(name = "gemma2-9b-it")]
    Gemma2Instruct9B,

    // Qwen
    #[default]
    #[value(name = "Qwen0.5b")]
    Qwen0_5b,
    #[value(name = "Qwen1.8b")]
    Qwen1_8b,
    #[value(name = "Qwen4b")]
    Qwen4b,
    #[value(name = "Qwen7b")]
    Qwen7b,
    #[value(name = "Qwen14b")]
    Qwen14b,
    #[value(name = "Qwen72b")]
    Qwen72b,
    #[value(name = "Qwen-moe-a2.7b")]
    QwenMoeA27b,
    #[value(name = "Qwen2-0.5b")]
    Qwen2_0_5b,
    #[value(name = "Qwen2-1.5b")]
    Qwen2_1_5b,
    #[value(name = "Qwen2-7b")]
    Qwen2_7b,
    #[value(name = "Qwen2-72b")]
    Qwen2_72b,
}

impl Which {
    pub fn is_gemma(&self) -> bool {
        match self {
            Self::Gemma2B
            | Self::Gemma7B
            | Self::GemmaInstruct2B
            | Self::GemmaInstruct7B
            | Self::GemmaInstructV1_1_2B
            | Self::GemmaInstructV1_1_7B
            | Self::GemmaCode2B
            | Self::GemmaCode7B
            | Self::GemmaCodeInstruct2B
            | Self::GemmaCodeInstruct7B => true,
            _ => false,
        }
    }
    pub fn is_gemma2(&self) -> bool {
        match self {
            Self::Gemma2_2B | Self::Gemma2Instruct2B | Self::Gemma2_9B | Self::Gemma2Instruct9B => true,
            _ => false,
        }
    }

    pub fn is_qwen(&self) -> bool {
        match self {
            Self::Qwen0_5b
            | Self::Qwen1_8b
            | Self::Qwen4b
            | Self::Qwen7b
            | Self::Qwen14b
            | Self::Qwen72b
            | Self::QwenMoeA27b
            | Self::Qwen2_0_5b
            | Self::Qwen2_1_5b
            | Self::Qwen2_7b
            | Self::Qwen2_72b => true,
            _ => false,
        }
    }

    pub fn is_qwen_moe(&self) -> bool {
        match self {
            Self::QwenMoeA27b => true,
            _ => false,
        }
    }
}

#[derive(Clone)]
pub enum Model {
    Gemma(ModelGemma),
    Gemma2(ModelGemma2),
    Qwen(ModelQwen),
    QwenMoe(ModelQwenMoe),
}

impl Model {
    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Gemma(m) => m.forward(input_ids, pos),
            Self::Gemma2(m) => m.forward(input_ids, pos),
            Self::QwenMoe(ref mut m) => m.forward(input_ids, pos),
            Self::Qwen(ref mut m) => m.forward(input_ids, pos),
        }
    }

    pub fn is_gemma(&self) -> bool {
        match self {
            Self::Gemma(_) => true,
            Self::Gemma2(_) => true,
            _ => false,
        }
    }

    pub fn is_qwen(&self) -> bool {
        match self {
            Self::Qwen(_) => true,
            Self::QwenMoe(_) => true,
            _ => false,
        }
    }
}

pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: Arc<Mutex<TokenOutputStream>>,
    tokenizer2: TokenOutputStream,
    logits_processor: Arc<Mutex<LogitsProcessor>>,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = Arc::new(Mutex::new(LogitsProcessor::new(seed, temp, top_p)));
        Self {
            model,
            tokenizer: Arc::new(Mutex::new(TokenOutputStream::new(tokenizer.clone()))),
            tokenizer2: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize, pool: web::Data<SqlitePool>, conversation_id: String) -> Result<Receiver<String>> {
        use std::io::Write;

        let (sender, receiver): (Sender<String>, Receiver<String>) = mpsc::channel(1);
        
        let tokenizer = Arc::clone(&self.tokenizer); 
        let pool = pool.clone();
        let conversation_id = conversation_id.clone();
        let sample_len = sample_len.clone();
        
        let device = self.device.clone();
        let mut model = self.model.clone();
        let logits_processor = Arc::clone(&self.logits_processor);
        let repeat_penalty = self.repeat_penalty;
        let repeat_last_n = self.repeat_last_n;

        self.tokenizer2.clear();
        let tokens = self.tokenizer2
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer2.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let tokens = Arc::new(Mutex::new(tokens));

        let eos_token: u32 = if self.model.is_qwen() {
            match self.tokenizer2.get_token("<|endoftext|>") {
                Some(token) => token,
                None => anyhow::bail!("cannot find the <|endoftext|> token"),
            }
        } else if self.model.is_gemma() {
            match self.tokenizer2.get_token("<eos>") {
                Some(token) => token,
                None => anyhow::bail!("cannot find the <eos> token"),
            }
        } else {
            anyhow::bail!("cannot find the eos token");
        };

        actix_rt::spawn(async move {
            if let Err(e) = async {

                let mut response = String::new();
                for index in 0..sample_len {
                    let mut tokens = tokens.lock().unwrap();
                    let context_size = if index > 0 { 1 } else { tokens.len() };
                    let start_pos = tokens.len().saturating_sub(context_size);
                    let ctxt = &tokens[start_pos..];
                    let input = match Tensor::new(ctxt, &device).and_then(|t| t.unsqueeze(0)) {
                        Ok(t) => t,
                        Err(e) => {
                            println!("Error: {}", e);
                            break;
                        }
                    };
                    let logits = model.forward(&input, start_pos).unwrap();
                    let logits = match logits.squeeze(0).and_then(|l| l.squeeze(0)).and_then(|l| l.to_dtype(DType::F32)) {
                        Ok(l) => l,
                        Err(e) => {
                            println!("Error: {}", e);
                            break;
                        }
                    };
                    let logits = if repeat_penalty == 1. {
                        logits
                    } else {
                        let start_at = tokens.len().saturating_sub(repeat_last_n);
                        candle_transformers::utils::apply_repeat_penalty(
                            &logits,
                            repeat_penalty,
                            &tokens[start_at..],
                        ).unwrap()
                    };

                    let next_token = {
                        let mut logits_processor = logits_processor.lock().unwrap();
                        logits_processor.sample(&logits).unwrap()
                    };
                
                    tokens.push(next_token);
                    if next_token == eos_token {
                        break;
                    }

                    let token_string = {
                        let mut tokenizer = tokenizer.lock().unwrap();
                        tokenizer.next_token(next_token)
                    };
                    if let Ok(Some(t)) = token_string {
                        println!("{t}");
                        response.push_str(&t);
                        if let Err(_) = sender.send(t).await {
                            eprintln!("Failed to send token"); // Log the error if sending fails
                        } // Send token text to the receiver
                    } else {
                        eprintln!("Failed to get token");
                    }
                }
                sqlx::query(
                    r#"
                    INSERT INTO messages (conversation_id, sender, content) VALUES (?, ?, ?)
                    "#,
                )
                .bind(&conversation_id)
                .bind("Assistant")
                .bind(&response)
                .execute(pool.get_ref()).await.expect("Failed to insert message into database");
                
                Ok::<(), Box<dyn std::error::Error>>(())
            }.await {
                eprintln!("Error in spawned task: {:?}", e);
            }

        });

        Ok(receiver)
    }
}