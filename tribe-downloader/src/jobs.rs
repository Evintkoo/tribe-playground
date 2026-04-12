//! jobs.rs — Download job queue for TRIBE v2.
use std::{collections::HashMap, sync::Arc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

// ── ModelKind ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Clip,
    Llama,
    Wav2Vec,
}

impl std::fmt::Display for ModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Clip    => "clip",
            Self::Llama   => "llama",
            Self::Wav2Vec => "wav2vec",
        })
    }
}

impl std::str::FromStr for ModelKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "clip"    => Ok(Self::Clip),
            "llama"   => Ok(Self::Llama),
            "wav2vec" => Ok(Self::Wav2Vec),
            other     => Err(format!("unknown model '{other}'; use clip, llama, or wav2vec")),
        }
    }
}

// ── JobStatus ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "PascalCase")]
pub enum JobStatus {
    Queued,
    Running,
    Done,
    Failed,
}

// ── Job ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Job {
    pub id:          Uuid,
    pub model:       ModelKind,
    pub status:      JobStatus,
    pub pct:         f32,
    pub bytes_done:  u64,
    pub total_bytes: Option<u64>,
    pub error:       Option<String>,
}

// ── JobStore ──────────────────────────────────────────────────────────────────

pub type JobStore = Arc<RwLock<HashMap<Uuid, Job>>>;

pub fn new_job_store() -> JobStore {
    Arc::new(RwLock::new(HashMap::new()))
}

// ── enqueue ───────────────────────────────────────────────────────────────────

/// Insert a new job for `model` and send its ID to the worker channel.
/// Returns `Err` if a Queued or Running job already exists for that model.
pub async fn enqueue(
    store: &JobStore,
    tx:    &tokio::sync::mpsc::Sender<Uuid>,
    model: ModelKind,
) -> Result<Uuid, String> {
    let mut guard = store.write().await;

    for job in guard.values() {
        if job.model == model && matches!(job.status, JobStatus::Queued | JobStatus::Running) {
            return Err(format!("job for {model} already active (id={})", job.id));
        }
    }

    let id = Uuid::new_v4();
    guard.insert(id, Job {
        id,
        model,
        status:      JobStatus::Queued,
        pct:         0.0,
        bytes_done:  0,
        total_bytes: None,
        error:       None,
    });
    drop(guard);

    tx.send(id).await.map_err(|e| e.to_string())?;
    Ok(id)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn enqueue_returns_uuid_and_rejects_duplicate() {
        let store = new_job_store();
        let (tx, _rx) = tokio::sync::mpsc::channel::<Uuid>(8);

        let id = enqueue(&store, &tx, ModelKind::Clip).await.unwrap();
        assert!(!id.is_nil());

        // Second enqueue for same model while first is Queued → Err
        let err = enqueue(&store, &tx, ModelKind::Clip).await;
        assert!(err.is_err());

        // Different model is fine
        let id2 = enqueue(&store, &tx, ModelKind::Llama).await.unwrap();
        assert_ne!(id, id2);
    }
}
