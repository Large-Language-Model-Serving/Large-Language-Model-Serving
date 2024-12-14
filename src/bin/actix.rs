mod model;

use anyhow::Ok;
use candle_transformers::models::segment_anything::sam;
use model::{TextGeneration, Model, Which};
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use serde::{de, Deserialize, Serialize};
use std::default;
use std::sync::{Arc, Mutex};

use anyhow::{Error as E, Result};
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

impl GenerateRequest{
    fn default() -> Self {
        Self {
            prompt: "Hi, please introduce yourself.".to_string(),
            sample_len: 10000,
            temperature: None,
            top_p: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            model: Which::Qwen0_5b,
            revision: "main".to_string(),
            use_flash_attn: false,
            seed: 111222333,
        }
    }   
}

#[derive(Serialize, Deserialize)]
struct GenerateResponse {
    output: String,
}

struct AppState {
    //generator: Arc<Mutex<TextGeneration>>,
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
    //println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    //let start = std::time::Instant::now();
    //let device = candle_core::Device::new_cuda(0)?;
    // let device = candle_core::Device::new_metal(0)?;     // On Mac
    let dtype = if data.device.is_cuda() {
        DType::F16  // BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &data.device)? };
    let model = if req.model.is_gemma() {
        let config: ConfigGemma = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        //let model = ModelGemma::new(req.use_flash_attn, &config, vb)?;
        let model = ModelGemma::new(false, &config, vb)?;
        Model::Gemma(model)
    } else if req.model.is_gemma2() {
        let config: ConfigGemma2 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        //let model = ModelGemma2::new(req.use_flash_attn, &config, vb)?;
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

   // println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
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

// #[post("/switch_model")]
// async fn switch_model(
//     data: web::Data<AppState>,
//     req: web::Json<GenerateRequest>,
// ) -> impl Responder {
//     let state = data.clone();
//     let mut generator = state.generator.lock().unwrap();
//     *generator = set_model(data, req).unwrap();
//     //let mut generator = data.generator.lock().unwrap();
//     //generator.model = req.model.unwrap();

    
//     HttpResponse::Ok().body("Model switched successfully!")
// }   

#[post("/generate")]
async fn generate(
    data: web::Data<AppState>,
    req: web::Json<GenerateRequest>,
) -> impl Responder {
    //let mut generator = data.generator.lock().unwrap();
    //let result = generator.run(&req.prompt, req.sample_len);
    let prompt = req.prompt.clone();
    let sample_len = req.sample_len.clone();
    let mut generator = set_model(data, req).unwrap();
    let receiver = generator.run(&prompt, sample_len).unwrap();
    for token in receiver {
        println!("{}", token); // Process each token as it arrives
    }
    // match result {
    //     Ok(output) => HttpResponse::Ok().json(GenerateResponse { output }),
    //     Err(e) => HttpResponse::InternalServerError().body(format!("Error: {}", e)),
    // }
    HttpResponse::Ok().body("Hello world!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // let args = Args::parse();
    //let model = /* Load model logic here */;
    //let tokenizer = /* Load tokenizer logic here */;
    println!("loading the model");
    let device = candle_core::Device::new_cuda(0).unwrap();
    println!("setted the device");
    let default = GenerateRequest::default();
    println!("setted the default");

    let access_token = "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";   // Hugging Face Access Token
    let api_builder = ApiBuilder::new();
    let api_builder_token =  api_builder.with_token(Some(String::from(access_token)));
    let api = api_builder_token.build().unwrap();
    let model_id = "Qwen/Qwen2-0.5B".to_string();
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        default.revision.clone(),
    ));
    println!("loaded the repo");
    let tokenizer_filename =  repo.get("tokenizer.json").unwrap();
    println!("loaded the tokenizer");
    
    let config_filename = repo.get("config.json").unwrap();
    println!("loaded the config");
    let filenames =  vec![repo.get("model.safetensors").unwrap()];
    println!("loaded the model");
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg).unwrap();
    println!("setted the tokenizer");

    let dtype = if device.is_cuda() {
        DType::F16  // BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device).unwrap() };
    let config: ConfigQwen = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
    let model = Model::Qwen(ModelQwen::new(&config, vb).unwrap());
    println!("setted the model");

    let generator = TextGeneration::new(model, tokenizer, default.seed, default.temperature, default.top_p, default.repeat_penalty, default.repeat_last_n, &device);
    println!("setted the generator");
    let state = web::Data::new(AppState {
       // generator: Arc::new(Mutex::new(generator)),
        device,
    });
    println!("loaded the model");
    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .service(hello)
            .service(generate)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
