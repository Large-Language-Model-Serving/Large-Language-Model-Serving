use dioxus::prelude::*;

use components::Navbar;
use views::{Blog, Home};
use serde::{Deserialize, Serialize};
use reqwest::Client;
use futures_core::stream::Stream;

mod components;
mod views;

#[derive(Debug, Clone, Routable, PartialEq)]
#[rustfmt::skip]
enum Route {
    #[layout(Navbar)]
    #[route("/")]
    Home {},
    #[route("/:id")]
    Blog { id: String },
}

#[derive(Serialize, Deserialize)]
struct GenerateRequest {
    prompt: String,
    sample_len: usize,
    temperature: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    model: String,
    revision: String,
    seed: u64,
}

#[derive(Deserialize)]
struct GenerateResponse {
    output: String,
}

#[derive(Serialize, Deserialize)]
pub struct Message {
    id: i64,
    conversation_id: String,
    pub sender: String,
    pub content: String,
    timestamp: String,
}

pub async fn fetch_all_conversations() -> dioxus::Result<Vec<String>> {
    let client = Client::new();
    let response = client
        .get("http://127.0.0.1:8000/conversations")
        .send()
        .await?;
    let conversation_ids: Vec<String> = response.json().await?;
    Ok(conversation_ids)
}

async fn create_conversation() -> dioxus::Result<String> {
    let client = Client::new();
    let response = client
        .get("http://127.0.0.1:8000/conversations/start")
        .send()
        .await?;
    let conversation_id: String = response.json().await?;
    Ok(conversation_id)
}

async fn fetch_messages(conversation_id: String) -> dioxus::Result<Vec<Message>> {
    let client = Client::new();
    let response = client
        .get(format!("http://127.0.0.1:8000/conversations/{}", conversation_id))
        .send()
        .await?;
    let messages: Vec<Message> = response.json().await?;
    Ok(messages)
}

async fn delete_conversation(conversation_id: String) -> dioxus::Result<()> {
    let client = Client::new();
    let response = client
        .delete(format!("http://127.0.0.1:8000/conversations/{}", conversation_id))
        .send()
        .await?;
    Ok(())
}

async fn generate(conversation_id: String, req: GenerateRequest) -> Result<impl Stream<Item = Result<bytes::Bytes, reqwest::Error>>, reqwest::Error> {
    let client = Client::new();
    let mut stream = client
        .post(format!("http://127.0.0.1:8000/conversations/{}/generate", conversation_id))
        .json(&req)
        .send()
        .await?
        .bytes_stream();
    Ok(stream)
}

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/styling/main.css");

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    // Build cool things ✌️
    let mut conversation_ids = use_signal(|| fetch_all_conversations().unwrap());
    //let mut conversation_ids = fetch_all_conversations().unwrap();
    if conversation_ids.is_empty() {
        create_conversation().unwrap();
        conversation_ids = fetch_all_conversations().unwrap();
    }

    rsx! {
        // Global app resources
        document::Link { rel: "icon", href: FAVICON }
        document::Link { rel: "stylesheet", href: MAIN_CSS }



        Router::<Route> {}
    }
}


// fn App(cx: Scope) -> Element {
//     let client = use_ref(cx, || Client::new());
//     let chat_history = use_state(cx, || vec![]);
//     let prompt = use_state(cx, || String::new());
//     let model = use_state(cx, || String::from("GemmaInstructV1_1_2B"));
//     let temperature = use_state(cx, || 1.0);
//     let top_p = use_state(cx, || 1.0);
//     let is_loading = use_state(cx, || false);

//     let send_message = {
//         let client = client.clone();
//         let chat_history = chat_history.clone();
//         let prompt = prompt.clone();
//         let model = model.clone();
//         let temperature = temperature.clone();
//         let top_p = top_p.clone();
//         let is_loading = is_loading.clone();

//         move |_| {
//             let client = client.clone();
//             let prompt_value = prompt.get().clone();
//             let model_value = model.get().clone();
//             let temperature_value = *temperature.get();
//             let top_p_value = *top_p.get();

//             is_loading.set(true);

//             cx.spawn(async move {
//                 let req = GenerateRequest {
//                     prompt: prompt_value.clone(),
//                     sample_len: 128,
//                     temperature: Some(temperature_value),
//                     top_p: Some(top_p_value),
//                     repeat_penalty: 1.2,
//                     repeat_last_n: 64,
//                     model: model_value.clone(),
//                     revision: String::from("main"),
//                     seed: 42,
//                 };

//                 match client.read().post("http://127.0.0.1:8080/conversations/123/generate")
//                     .json(&req)
//                     .send()
//                     .await
//                 {
//                     Ok(response) => {
//                         if let Ok(data) = response.json::<GenerateResponse>().await {
//                             chat_history.modify(|history| {
//                                 let mut updated = history.clone();
//                                 updated.push(("User".to_string(), prompt_value));
//                                 updated.push(("Assistant".to_string(), data.output));
//                                 updated
//                             });
//                         }
//                     }
//                     Err(err) => {
//                         eprintln!("Error: {:?}", err);
//                     }
//                 }

//                 is_loading.set(false);
//             });
//         }
//     };

//     cx.render(rsx! {
//         div {
//             class: "chat-interface",
//             h1 { "LLM Chat Interface" }

//             div {
//                 class: "chat-history",
//                 chat_history.get().iter().map(|(sender, message)| {
//                     rsx!(
//                         div {
//                             class: "message",
//                             span { class: "sender", "{sender}:" }
//                             span { class: "content", "{message}" }
//                         }
//                     )
//                 })
//             }

//             div {
//                 class: "controls",
//                 select {
//                     value: "{model}",
//                     oninput: |e| model.set(e.value.clone()),
//                     option { value: "GemmaInstructV1_1_2B", "GemmaInstructV1_1_2B" }
//                     option { value: "Qwen7b", "Qwen7b" }
//                     // Add other model options here...
//                 }
//                 input {
//                     r#type: "number",
//                     step: "0.1",
//                     value: "{temperature}",
//                     oninput: |e| temperature.set(e.value.parse().unwrap_or(1.0)),
//                 }
//                 input {
//                     r#type: "number",
//                     step: "0.1",
//                     value: "{top_p}",
//                     oninput: |e| top_p.set(e.value.parse().unwrap_or(1.0)),
//                 }
//                 input {
//                     r#type: "text",
//                     placeholder: "Type your message...",
//                     value: "{prompt}",
//                     oninput: |e| prompt.set(e.value.clone()),
//                 }
//                 button {
//                     onclick: send_message,
//                     "Send"
//                 }
//             }

//             if *is_loading.get() {
//                 div { class: "loading", "Loading..." }
//             }
//         }
//     })
// }

// fn App(cx: Scope) -> Element {
//     let state = use_state(cx, || ChatState::default());
//     let client = use_ref(cx, || Client::new());

//     cx.render(rsx! {
//         div {
//             h1 { "LLM Chat Interface" }
//             form {
//                 onsubmit: move |evt| {
//                     evt.prevent_default();
//                     let client = client.clone();
//                     let state = state.clone();
//                     let input = state.input.clone();
//                     let model = state.model.clone();

//                     // Create the GenerateRequest
//                     let req = GenerateRequest {
//                         prompt: input.clone(),
//                         sample_len: 100,
//                         temperature: Some(state.temperature),
//                         top_p: Some(state.top_p),
//                         repeat_penalty: 1.0,
//                         repeat_last_n: 50,
//                         model: model.clone(),
//                         revision: "main".to_string(),
//                         seed: 42,
//                     };

//                     // Send the request to the server
//                     wasm_bindgen_futures::spawn_local(async move {
//                         if let Ok(response) = client.read().post("http://127.0.0.1:8080/conversations/1/generate")
//                             .json(&req)
//                             .send().await {
//                                 if let Ok(body) = response.json::<GenerateResponse>().await {
//                                     state.modify(|s| {
//                                         s.messages.push(format!("User: {}", input));
//                                         s.messages.push(format!("Assistant: {}", body.output));
//                                         s.input.clear();
//                                     });
//                                 }
//                             }
//                     });
//                 },
//                 div {
//                     label { "Model: " }
//                     select {
//                         onchange: move |evt| {
//                             state.modify(|s| s.model = evt.value.clone());
//                         },
//                         option { value: "GemmaInstructV1_1_2B", "GemmaInstructV1_1_2B" }
//                         option { value: "Qwen2_7b", "Qwen2_7b" }
//                         option { value: "Qwen72b", "Qwen72b" }
//                     }
//                 }
//                 div {
//                     label { "Temperature: " }
//                     input {
//                         r#type: "number",
//                         value: "{state.temperature}",
//                         step: "0.1",
//                         onchange: move |evt| {
//                             if let Ok(temp) = evt.value.parse() {
//                                 state.modify(|s| s.temperature = temp);
//                             }
//                         }
//                     }
//                 }
//                 div {
//                     label { "Top-p: " }
//                     input {
//                         r#type: "number",
//                         value: "{state.top_p}",
//                         step: "0.1",
//                         onchange: move |evt| {
//                             if let Ok(top_p) = evt.value.parse() {
//                                 state.modify(|s| s.top_p = top_p);
//                             }
//                         }
//                     }
//                 }
//                 div {
//                     label { "Prompt: " }
//                     textarea {
//                         rows: "5",
//                         cols: "50",
//                         value: "{state.input}",
//                         onchange: move |evt| {
//                             state.modify(|s| s.input = evt.value.clone());
//                         }
//                     }
//                 }
//                 button { r#type: "submit", "Send" }
//             }
//             div {
//                 h2 { "Chat History" }
//                 state.messages.iter().map(|message| rsx! {
//                     p { "{message}" }
//                 })
//             }
//         }
//     })
// }