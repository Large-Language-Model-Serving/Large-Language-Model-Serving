mod model;
mod db;

use actix_cors::Cors;
use actix_web::{delete, get, post, web, App, HttpResponse, HttpServer, Responder};
use actix_web::web::Bytes;
use actix_web::middleware::Logger;
use anyhow::Ok;
use anyhow::{Error as E, Result};

use candle_transformers::models::gemma::{Config as ConfigGemma, Model as ModelGemma};
use candle_transformers::models::gemma2::{Config as ConfigGemma2, Model as ModelGemma2};
use candle_transformers::models::qwen2::{Config as ConfigQwen, ModelForCausalLM as ModelQwen};
use candle_transformers::models::qwen2_moe::{Config as ConfigQwenMoe, Model as ModelQwenMoe};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use futures_util::stream::StreamExt;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use model::{TextGeneration, Model, Which};
use tokenizers::Tokenizer;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::fmt::Write;
use std::result::Result::Ok as ResultOk;





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
    seed: u64,
} 


#[derive(Serialize, Deserialize)]
struct GenerateResponse {
    output: String,
}

#[derive(Serialize, Deserialize, sqlx::FromRow)]
struct Conversation {
    id: i64,
    conversation_id: String,
    created_at: String,
}

struct AppState {
    device: Device,
}

fn set_model(
    data: web::Data<AppState>, 
    req: web::Json<GenerateRequest>
) -> Result<TextGeneration, anyhow::Error>{
    let access_token = "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";   // Hugging Face Access Token
    let api_builder = ApiBuilder::new();
    let api_builder_token = api_builder.with_token(Some(String::from(access_token)));
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
    let tokenizer_filename = repo.get("tokenizer.json")?;
    
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

#[get("/conversations")]
async fn fetch_all_conversations(pool: web::Data<SqlitePool>) -> impl Responder {
    match db::get_all_conversation_ids(pool.get_ref()).await {
        ResultOk(conversation_ids) => HttpResponse::Ok().json(conversation_ids),
        Err(e) => {
            eprintln!("Error fetching conversation IDs: {:?}", e);
            HttpResponse::InternalServerError().body("Failed to fetch conversation IDs")
        }
    }
}

#[get("/conversations/start")]
async fn start_conversation(pool: web::Data<SqlitePool>) -> impl Responder {
    let conversation_id = Uuid::new_v4().to_string();

    let result = sqlx::query(
        "INSERT INTO conversations (conversation_id) VALUES (?)",
    )
    .bind(&conversation_id)
    .execute(pool.get_ref())
    .await;

    match result {
        ResultOk(_) => HttpResponse::Created().json(conversation_id),
        Err(e) => {
            eprintln!("Error starting conversation: {:?}", e);
            HttpResponse::InternalServerError().body("Failed to start conversation")
        }
    }
}

#[get("/conversations/{conversation_id}")]
async fn view_conversation(
    pool: web::Data<SqlitePool>,
    path: web::Path<String>,
) -> impl Responder {
    let conversation_id = path.into_inner();

    match db::get_conversation(pool.clone(), conversation_id).await {
        ResultOk(messages) => HttpResponse::Ok().json(messages),
        Err(e) => {
            eprintln!("Error fetching conversation: {:?}", e);
            HttpResponse::InternalServerError().body("Failed to fetch conversation")
        }
    }
}

#[delete("/conversations/{conversation_id}")]
async fn delete_conversation(
    pool: web::Data<SqlitePool>,
    path: web::Path<String>,
) -> impl Responder {
    let conversation_id = path.into_inner();

    let result = sqlx::query(
        "DELETE FROM conversations WHERE conversation_id = ?",
    )
    .bind(conversation_id)
    .execute(pool.get_ref())
    .await;

    match result {
        ResultOk(_) => HttpResponse::Ok().finish(),
        Err(e) => {
            eprintln!("Error deleting conversation: {:?}", e);
            HttpResponse::InternalServerError().body("Failed to delete conversation")
        }
    }
}


#[post("/conversations/{conversation_id}/generate")]
async fn generate(
    pool: web::Data<SqlitePool>,
    path: web::Path<String>,
    data: web::Data<AppState>,
    req: web::Json<GenerateRequest>,
) -> impl Responder {
    let conversation_id = path.into_inner();
    let conversation = match db::get_conversation(pool.clone(), conversation_id.clone()).await {
        ResultOk(conversation) => conversation,
        Err(e) => {
            eprintln!("Error fetching conversation: {:?}", e);
            return HttpResponse::InternalServerError().body("Failed to fetch conversation history");
        }
    };

    let mut formatted_prompt = String::new();
    for message in conversation {
        let _ = writeln!(
            &mut formatted_prompt,
            "{}: {}",
            message.sender,
            message.content
        );
    }

    let prompt = req.prompt.clone();
    // Append the new user input
    let _ = writeln!(&mut formatted_prompt, "User: {}", prompt);

    sqlx::query(
        r#"
        INSERT INTO messages (conversation_id, sender, content) VALUES (?, ?, ?)
        "#,
    )
    .bind(&conversation_id)
    .bind("User")
    .bind(&prompt)
    .execute(pool.get_ref()).await.expect("Failed to insert message into database");

    // Add a label for the language model's expected response
    let _ = writeln!(&mut formatted_prompt, "Assistant:");
    
    let sample_len = req.sample_len.clone();
    let mut generator = match set_model(data, req) {
        ResultOk(gen) => gen,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Error: {}", e)),
    };
    let receiver = match generator.run(&formatted_prompt, sample_len, pool.clone(), conversation_id.clone()) {
        ResultOk(recv) => recv,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Error: {}", e)),
    };

    // Wrap the receiver in a `ReceiverStream`
    let stream = ReceiverStream::new(receiver).map(|token| {
        println!("Token received: {}", token);
        // Convert the token to bytes
        Ok(Bytes::from(token)) as Result<Bytes, anyhow::Error>
    });

    // Return the streaming response
    HttpResponse::Ok()
        .content_type("text/plain") 
        .streaming(Box::pin(stream))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let device = candle_core::Device::new_cuda(0).unwrap();

    let state = web::Data::new(AppState {device});

    use std::fs::File;
    use std::path::Path;

    if !Path::new("src/conversations.db").exists() {
        File::create("src/conversations.db").unwrap();
        println!("Database created successfully");
    } else {
        println!("Database already exists");
    }

    let pool = match SqlitePool::connect("sqlite://src/conversations.db").await {
        ResultOk(e) => e,
        Err(e) => 
        { 
            eprintln!("Error connecting to database: {}", e);
            std::process::exit(1);
        }
    };

    let _ = db::create_tables(&pool).await.map_err(|e| {
        eprintln!("Error creating tables: {}", e);
        std::process::exit(1);
    });

    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default()) // Logs HTTP requests and responses
            .wrap(
                Cors::default()
                    .allow_any_origin() // Allow requests from any origin
                    .allow_any_method() // Allow any HTTP method (e.g., GET, POST)
                    .allow_any_header() // Allow any HTTP header
            )
            .app_data(state.clone())
            .app_data(web::Data::new(pool.clone()))
            .service(hello)
            .service(generate)
            .service(start_conversation)
            .service(fetch_all_conversations)
            .service(view_conversation)
            .service(delete_conversation)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
