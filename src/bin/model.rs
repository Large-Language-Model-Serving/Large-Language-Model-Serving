#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::gemma::{Config as ConfigGemma, Model as ModelGemma};
use candle_transformers::models::gemma2::{Config as ConfigGemma2, Model as ModelGemma2};

use candle_transformers::models::qwen2::{Config as ConfigQwen, ModelForCausalLM as ModelQwen};
use candle_transformers::models::qwen2_moe::{Config as ConfigQwenMoe, Model as ModelQwenMoe};


use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{self, Sender, Receiver};
use std::sync::{Arc, Mutex};

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
        //let logits_processor = LogitsProcessor::new(seed, temp, top_p);
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

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<Receiver<String>> {
        use std::io::Write;

        let (sender, receiver): (Sender<String>, Receiver<String>) = mpsc::channel();
        let tokenizer = Arc::clone(&self.tokenizer); 
        
        let device = self.device.clone();
        let mut model = self.model.clone();
        let logits_processor = Arc::clone(&self.logits_processor);
        let repeat_penalty = self.repeat_penalty;
        let repeat_last_n = self.repeat_last_n;

        self.tokenizer2.clear();
        let mut tokens = self.tokenizer2
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

        let mut generated_tokens = 0usize;
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
         
        let start_gen = std::time::Instant::now();
        std::thread::spawn(move || {
            let mut tokenizer = tokenizer.lock().unwrap();
            let mut logits_processor = logits_processor.lock().unwrap();
            for index in 0..sample_len {
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

                let next_token = logits_processor.sample(&logits).unwrap();
                tokens.push(next_token);
                generated_tokens += 1;
                if next_token == eos_token {
                    break;
                }
                if let Ok(Some(t)) = tokenizer.next_token(next_token) {
                    //print!("{t}");
                    //std::io::stdout().flush()?;
                    if let Err(_) = sender.send(t){
                        eprintln!("Failed to send token"); // Log the error if sending fails
                    } // Send token text to the receiver
                } else {
                    eprintln!("Failed to get token");
                }
            }

        });

        Ok(receiver)
        // let dt = start_gen.elapsed();
        // if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
        //     print!("{rest}");
        // }
        // std::io::stdout().flush()?;
        // println!(
        //     "\n{generated_tokens} tokens generated ({:.2} token/s)",
        //     generated_tokens as f64 / dt.as_secs_f64(),
        // );
        // Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long, default_value = "hello")]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The model to use.
    #[arg(long, default_value = "Qwen0.5b")]
    model: Which,

    #[arg(long)]
    use_flash_attn: bool,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let access_token = "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";   // Hugging Face Access Token
    let api_builder = ApiBuilder::new();
    let api_builder_token =  api_builder.with_token(Some(String::from(access_token)));
    let api = api_builder_token.build()?;
    let model_id = match &args.model_id {
        Some(model_id) => model_id.to_string(),
        None => match args.model {
            Which::GemmaInstructV1_1_2B => "google/gemma-1.1-2b-it".to_string(),
            Which::GemmaInstructV1_1_7B => "google/gemma-1.1-7b-it".to_string(),
            Which::Gemma2B => "google/gemma-2b".to_string(),
            Which::Gemma7B => "google/gemma-7b".to_string(),
            Which::GemmaInstruct2B => "google/gemma-2b-it".to_string(),
            Which::GemmaInstruct7B => "google/gemma-7b-it".to_string(),
            Which::GemmaCode2B => "google/codegemma-2b".to_string(),
            Which::GemmaCode7B => "google/codegemma-7b".to_string(),
            Which::GemmaCodeInstruct2B => "google/codegemma-2b-it".to_string(),
            Which::GemmaCodeInstruct7B => "google/codegemma-7b-it".to_string(),
            Which::Gemma2_2B => "google/gemma-2-2b".to_string(),
            Which::Gemma2Instruct2B => "google/gemma-2-2b-it".to_string(),
            Which::Gemma2_9B => "google/gemma-2-9b".to_string(),
            Which::Gemma2Instruct9B => "google/gemma-2-9b-it".to_string(),
            Which::Qwen2_0_5b => "Qwen/Qwen2-0.5B".to_string(),
            Which::Qwen2_1_5b => "Qwen/Qwen2-1.5B".to_string(),
            Which::Qwen2_7b => "Qwen/Qwen2-7B".to_string(),
            Which::Qwen2_72b => "Qwen/Qwen2-72B".to_string(),
            Which::Qwen0_5b => "Qwen/Qwen1.5-0.5B".to_string(),
            Which::Qwen1_8b => "Qwen/Qwen1.5-1.8B".to_string(),
            Which::Qwen4b => "Qwen/Qwen1.5-4B".to_string(),
            Which::Qwen7b => "Qwen/Qwen1.5-7B".to_string(),
            Which::Qwen14b => "Qwen/Qwen1.5-14B".to_string(),
            Which::Qwen72b => "Qwen/Qwen1.5-72B".to_string(),
            Which::QwenMoeA27b => "Qwen/Qwen1.5-MoE-A2.7B".to_string(),
        },
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let config_filename = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("config.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => match args.model {
            Which::Qwen0_5b | Which::Qwen2_0_5b | Which::Qwen2_1_5b | Which::Qwen1_8b => {
                vec![repo.get("model.safetensors")?]
            }
            _ => candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
        },
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let device = candle_core::Device::new_cuda(0)?;
    // let device = candle_core::Device::new_metal(0)?;     // On Mac
    let dtype = if device.is_cuda() {
        DType::F16  // BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = if args.model.is_gemma() {
        let config: ConfigGemma = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        let model = ModelGemma::new(args.use_flash_attn, &config, vb)?;
        Model::Gemma(model)
    } else if args.model.is_gemma2() {
        let config: ConfigGemma2 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        let model = ModelGemma2::new(args.use_flash_attn, &config, vb)?;
        Model::Gemma2(model)
    } else if args.model.is_qwen() {
        let config: ConfigQwen = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        let model = ModelQwen::new(&config, vb)?;
        Model::Qwen(model)
    } else if args.model.is_qwen_moe() {
        let config: ConfigQwenMoe = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        let model = ModelQwenMoe::new(&config, vb)?;
        Model::QwenMoe(model)
    } else {
        unreachable!()
    };

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}