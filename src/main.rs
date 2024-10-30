use anyhow::{Error, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama as model;
use hf_hub::api::sync::Api;
use model::{Llama, LlamaConfig};
use std::io::Write;
use tokenizers::Tokenizer;

mod token_output_stream;

const EOS_TOKEN: &str = "</s>";
// const DEFAULT_PROMPT: &str = "My favorite theorem is ";

/// Loads the safetensors files for a model from the hub based on a json index file.
pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> candle_core::Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => candle_core::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle_core::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle_core::Error::wrap))
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok(safetensors_files)
}
struct Args {
    /// Run on CPU rather than on GPU.
    // cpu: bool,

    /// The temperature used to generate samples.
    // temperature: f64,

    /// Nucleus sampling probability cutoff.
    // top_p: Option<f64>,

    /// Only sample among the top K samples.
    // top_k: Option<usize>,

    /// The seed to use when generating random samples.
    // seed: u64,

    /// The length of the sample to generate (in tokens).
    // sample_len: usize,

    /// Disable the key-value cache.
    // no_kv_cache: bool,

    /// The initial prompt.
    // prompt: Option<String>,

    /// Use different dtype than f16
    // dtype: Option<String>,

    /// Enable tracing (generates a trace-timestamp.json file).
    // tracing: bool,

    // model_id: Option<String>,

    // revision: Option<String>,

    /// The model size to use.
    // which: Which,

    // use_flash_attn: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    let args = Args {
        repeat_penalty: 1.1,
        repeat_last_n: 128,
    };

    // Specify the device (CPU or GPU if available)
    let device = Device::Cpu;

    // Load the tokenizer
    let api = Api::new()?;
    // let api = HfHub::new("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");
    let repo = api.model("meta-llama/Llama-3.2-3B-Instruct".to_string());
    let config_filename = repo.get("config.json").unwrap();
    let tokenizer_filename = repo.get("tokenizer.json").unwrap();
    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let use_flash_attn = false;
    let config = config.into_config(use_flash_attn);
    let weights_filenames = hub_load_safetensors(&repo, "model.safetensors.index.json")?;
    // let config: BertModel = serde_json::from_str(&std::fs::read(config_filename)?)?;
    // let config: BertModel = serde_json::from_str(&config).expect("Failed to parse config string");
    let dtype = DType::F16;
    let mut cache = model::Cache::new(true, dtype, &config, &device)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_filenames, dtype, &device)? };
    println!("start loading");
    let llama = Llama::load(vb, &config)?;
    println!("finished loading");
    let tokenizer = Tokenizer::from_file(tokenizer_filename).expect("Failed to load tokenizer");
    // Load the model weights and configuration
    // let config_path = "models/bert-base-uncased/config.json";
    // // let weights_path = "models/bert-base-uncased/pytorch_model.bin";
    // let weights_path = "models/bert-base-uncased/model.safetensors";

    // let vb = VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)?;
    // println!("here");
    // let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device) }?;
    // let config = std::fs::read_to_string(config_path)?;
    // let config: Config = serde_json::from_str(&config).expect("Failed to parse config string");
    let eos_token_id = config.eos_token_id.or_else(|| {
        tokenizer
            .token_to_id(EOS_TOKEN)
            .map(model::LlamaEosToks::Single)
    });
    let prompt = "My favorite theorem is ";
    // let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(Error::msg)?
        .get_ids()
        .to_vec();
    let mut tokenizer = token_output_stream::TokenOutputStream::new(tokenizer);

    println!("starting the inference loop");
    print!("{prompt}");
    let mut logits_processor = {
        let temperature = 0.8;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            Sampling::All { temperature }
        };
        let seed: u64 = 299792458;
        LogitsProcessor::from_sampling(seed, sampling)
    };

    let mut start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    let sample_len: usize = 10000;
    for index in 0..sample_len {
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            (1, index_pos)
        } else {
            (tokens.len(), 0)
        };
        if index == 1 {
            start_gen = std::time::Instant::now()
        }
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, context_index, &mut cache)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        match eos_token_id {
            Some(model::LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                break;
            }
            Some(model::LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                break;
            }
            _ => (),
        }
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(Error::msg)? {
        print!("{rest}");
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated - 1) as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
