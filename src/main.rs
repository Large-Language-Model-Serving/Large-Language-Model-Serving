mod model;

use anyhow::Ok;
use model::{TextGeneration, Model, Which};
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use anyhow::{Error as E, Result};
use candle_transformers::models::gemma::{Config as ConfigGemma, Model as ModelGemma};
use candle_transformers::models::gemma2::{Config as ConfigGemma2, Model as ModelGemma2};
use candle_transformers::models::qwen2::{Config as ConfigQwen, ModelForCausalLM as ModelQwen};
use candle_transformers::models::qwen2_moe::{Config as ConfigQwenMoe, Model as ModelQwenMoe};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;
use actix_web::web::Bytes;
use std::result::Result::Ok as ResultOk;
use actix_web::middleware::Logger;
use futures_util::stream::StreamExt;


#[derive(Serialize, Deserialize, Default)]
struct GenerateRequest {
    prompt: String,
    sample_len: usize,
    temperature: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    model: Which,
    revision: String,
    use_flash_attn: bool,
    seed: u64,
} 


#[derive(Serialize, Deserialize)]
struct GenerateResponse {
    output: String,
}

struct AppState {
    device: Device,
}

fn set_model(data: web::Data<AppState>, req: web::Json<GenerateRequest>) -> Result<TextGeneration, anyhow::Error>{
    let access_token = "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";   // Hugging Face Access Token
    let api_builder = ApiBuilder::new();
    let api_builder_token =  api_builder.with_token(Some(String::from(access_token)));
    let api = api_builder_token.build()?;
    let model_id = match req.model {
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
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        req.revision.clone(),
    ));
    let tokenizer_filename =  repo.get("tokenizer.json")?;
    
    let config_filename = repo.get("config.json")?;
    let filenames =  match req.model {
            Which::Qwen0_5b | Which::Qwen2_0_5b | Which::Qwen2_1_5b | Which::Qwen1_8b => {
                vec![repo.get("model.safetensors")?]
            }
            _ => candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,  
    };
 
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let dtype = if data.device.is_cuda() {
        DType::F16  // BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &data.device)? };
    let model = if req.model.is_gemma() {
        let config: ConfigGemma = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        let model = ModelGemma::new(false, &config, vb)?;
        Model::Gemma(model)
    } else if req.model.is_gemma2() {
        let config: ConfigGemma2 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        let model = ModelGemma2::new(false, &config, vb)?;
        Model::Gemma2(model)
    } else if req.model.is_qwen() {
        let config: ConfigQwen = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        let model = ModelQwen::new(&config, vb)?;
        Model::Qwen(model)
    } else if req.model.is_qwen_moe() {
        let config: ConfigQwenMoe = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        let model = ModelQwenMoe::new(&config, vb)?;
        Model::QwenMoe(model)
    } else {
        unreachable!()
    };

    let pipeline = TextGeneration::new(
        model,
        tokenizer,
        req.seed,
        req.temperature,
        req.top_p,
        req.repeat_penalty,
        req.repeat_last_n,
        &data.device,
    );

    Ok(pipeline)
}

#[get("/")]
async fn hello() -> impl Responder {
    HttpResponse::Ok().body("Hello world!")
}


#[post("/generate")]
async fn generate(
    data: web::Data<AppState>,
    req: web::Json<GenerateRequest>,
) -> impl Responder {
    let prompt = req.prompt.clone();
    let sample_len = req.sample_len.clone();
    let mut generator = match set_model(data, req) {
        ResultOk(gen) => gen,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Error: {}", e)),
    };
    let receiver = match generator.run(&prompt, sample_len) {
        ResultOk(recv) => recv,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Error: {}", e)),
    };
    

    // Create a stream from the receiver
    let stream = Box::pin(tokio_stream::iter(receiver).map(|token| {
    // Convert each token into a Result<Bytes, anyhow::Error>
        Result::<Bytes, anyhow::Error>::Ok(Bytes::from(token))
    }));

    // Return the streaming response
    HttpResponse::Ok()
        .content_type("text/plain") 
        .streaming(stream)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let device = candle_core::Device::new_cuda(0).unwrap();

    let state = web::Data::new(AppState {
        device,
    });

    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default()) // Logs HTTP requests and responses
            .app_data(state.clone())
            .service(hello)
            .service(generate)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
