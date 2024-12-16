use crate::components::Hero;
use dioxus::prelude::*;

#[component]
pub fn Home() -> Element {
    let conversation_ids = use_signal(|| fetch_all_conversations().unwrap());
    rsx! {
        ul {
            for conversation_id in conversation_ids.iter() {
                li { 
                    a { href: format!("/{}", conversation_id), "{conversation_id}" }
                    //"{conversation_id}" 
                }
            }
        }
    //    Hero {}

    }
}
