// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared wire contracts for cache-aware multimodal E/P/D routing.

use std::collections::BTreeMap;

use blake2::{
    Blake2bVar,
    digest::{Update, VariableOutput},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub fn vllm_embedding_cache_key(content_id: &str) -> String {
    format!("{:x}", Sha256::digest(content_id.as_bytes()))
}

pub fn sglang_image_cache_key(url: &str) -> String {
    blake3::hash(url.as_bytes()).to_hex().to_string()
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRole {
    #[serde(rename = "E")]
    Encode,
    #[serde(rename = "P")]
    Prefill,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum ResidencyAction {
    Add,
    Remove,
    Clear,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum ObjectSourceKind {
    Url,
    Decoded,
    Inline,
    Uuid,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MmSourceKind {
    PLocal,
    PRemote,
    ECache,
    ECompute,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct EmbeddingNamespace {
    pub model_id: String,
    pub model_revision: String,
    pub processor_fingerprint: String,
}

impl EmbeddingNamespace {
    pub fn from_processor_config(
        model_id: impl Into<String>,
        model_revision: impl Into<String>,
        processor_config: &serde_json::Value,
    ) -> anyhow::Result<Self> {
        let canonical = canonical_json(processor_config)?;
        Ok(Self {
            model_id: model_id.into(),
            model_revision: model_revision.into(),
            processor_fingerprint: blake2b_hex(&canonical, 16)?,
        })
    }

    pub fn cache_key(&self, content_id: &str) -> anyhow::Result<String> {
        let payload = [
            self.model_id.as_str(),
            self.model_revision.as_str(),
            self.processor_fingerprint.as_str(),
            content_id,
        ]
        .join("\0");
        blake2b_hex(payload.as_bytes(), 32)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct MmTokenSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MmObjectRef {
    pub object_index: usize,
    pub modality: String,
    pub content_id: String,
    pub embedding_cache_key: String,
    pub model_token_spans: Vec<MmTokenSpan>,
    pub source_kind: ObjectSourceKind,
    #[serde(default)]
    pub model_visible_metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MmObjectPlan {
    pub object_index: usize,
    pub source_kind: MmSourceKind,
    pub source_worker_id: u64,
    pub source_worker_generation: u64,
    pub estimated_cost_ms: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MmRoutingPlan {
    pub target_p_worker_id: u64,
    pub target_p_generation: u64,
    pub objects: Vec<MmObjectPlan>,
    pub predicted_benefit_ms: f64,
    pub score_components: BTreeMap<String, f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct MmCacheResidencyEvent {
    pub worker_id: u64,
    pub worker_generation: u64,
    pub worker_role: WorkerRole,
    pub namespace: EmbeddingNamespace,
    pub cache_key: Option<String>,
    pub action: ResidencyAction,
    pub bytes: u64,
    pub observed_at_ms: u64,
}

fn canonical_json(value: &serde_json::Value) -> anyhow::Result<Vec<u8>> {
    fn canonicalize(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Array(items) => {
                serde_json::Value::Array(items.iter().map(canonicalize).collect())
            }
            serde_json::Value::Object(items) => {
                let ordered = items
                    .iter()
                    .map(|(key, value)| (key.clone(), canonicalize(value)))
                    .collect::<BTreeMap<_, _>>();
                serde_json::to_value(ordered).expect("BTreeMap JSON serialization cannot fail")
            }
            value => value.clone(),
        }
    }

    Ok(serde_json::to_vec(&canonicalize(value))?)
}

fn blake2b_hex(payload: &[u8], digest_size: usize) -> anyhow::Result<String> {
    let mut hasher = Blake2bVar::new(digest_size)
        .map_err(|error| anyhow::anyhow!("invalid BLAKE2b digest size: {error}"))?;
    hasher.update(payload);
    let mut digest = vec![0; digest_size];
    hasher
        .finalize_variable(&mut digest)
        .map_err(|error| anyhow::anyhow!("failed to finalize BLAKE2b digest: {error}"))?;
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn namespace_hash_is_stable_across_object_key_order() {
        let first = EmbeddingNamespace::from_processor_config(
            "example/model",
            "revision-1",
            &json!({"max_pixels": 1280, "nested": {"b": 2, "a": 1}}),
        )
        .unwrap();
        let second = EmbeddingNamespace::from_processor_config(
            "example/model",
            "revision-1",
            &json!({"nested": {"a": 1, "b": 2}, "max_pixels": 1280}),
        )
        .unwrap();

        assert_eq!(first, second);
        assert_eq!(
            first.processor_fingerprint,
            "f445a889cca07df19d3ac2ff949dfb67"
        );
        assert_eq!(
            first.cache_key("image-content-id").unwrap(),
            "4abb0632bc2809625e5a16f8af1bb61f6209223da3303c4dca76defe0e46f119"
        );
    }

    #[test]
    fn vllm_cache_key_matches_sha256_contract() {
        assert_eq!(
            vllm_embedding_cache_key("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sglang_image_cache_key_matches_blake3_contract() {
        assert_eq!(
            sglang_image_cache_key("abc"),
            "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
        );
    }

    #[test]
    fn residency_event_wire_shape_matches_python_contract() {
        let event = MmCacheResidencyEvent {
            worker_id: 7,
            worker_generation: 3,
            worker_role: WorkerRole::Encode,
            namespace: EmbeddingNamespace {
                model_id: "example/model".into(),
                model_revision: "revision-1".into(),
                processor_fingerprint: "fingerprint".into(),
            },
            cache_key: Some("cache-key".into()),
            action: ResidencyAction::Add,
            bytes: 4096,
            observed_at_ms: 1234,
        };

        assert_eq!(
            serde_json::to_value(event).unwrap(),
            json!({
                "worker_id": 7,
                "worker_generation": 3,
                "worker_role": "E",
                "namespace": {
                    "model_id": "example/model",
                    "model_revision": "revision-1",
                    "processor_fingerprint": "fingerprint"
                },
                "cache_key": "cache-key",
                "action": "ADD",
                "bytes": 4096,
                "observed_at_ms": 1234
            })
        );
    }
}
