use sqlx::{self, SqlitePool};
use sqlx::Error;
use sqlx::Row;
use serde::{Deserialize, Serialize};
use actix_web::web;

#[derive(Serialize, Deserialize, sqlx::FromRow)]
pub struct Message {
    id: i64,
    conversation_id: String,
    pub sender: String,
    pub content: String,
    timestamp: String,
}

pub async fn create_tables(pool: &SqlitePool) -> Result<(), Error> {
    let create_conversations = r#"
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    "#;

    let create_messages = r#"
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            sender TEXT NOT NULL, -- "user" or "model"
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
        );
    "#;

    // Execute queries
    sqlx::query(create_conversations).execute(pool).await?;
    sqlx::query(create_messages).execute(pool).await?;

    println!("Tables `conversations` and `messages` created successfully");
    Ok(())
}


pub async fn get_all_conversation_ids(pool: &SqlitePool) -> Result<Vec<String>, Error> {
    let rows = sqlx::query(
        r#"
        SELECT conversation_id FROM conversations
        "#
    )
    .fetch_all(pool)
    .await?;

    // Map the rows to a vector of strings
    let conversation_ids = rows.iter().map(|row| row.get("conversation_id")).collect();

    Ok(conversation_ids)
}

pub async fn get_conversation(pool: web::Data<SqlitePool>, conversation_id: String) -> Result<Vec<Message>, Error> {
    let result: Result<Vec<Message>, Error> = sqlx::query_as(
        r#"
        SELECT id, conversation_id, sender, content, timestamp
        FROM messages
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
        "#
    )
    .bind(&conversation_id)
    .fetch_all(pool.get_ref())
    .await;

    result
}