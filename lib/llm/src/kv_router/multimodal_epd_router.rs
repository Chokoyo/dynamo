// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selection-only multimodal E/P/D stage that runs before Encode dispatch.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, HashSet},
    str::FromStr,
    sync::Arc,
};

use anyhow::Result;
use dynamo_runtime::{
    pipeline::{ManyOut, Operator, ServerStreamingEngine, SingleIn, async_trait},
    protocols::annotated::Annotated,
};

use crate::{
    kv_router::{
        EncoderRouter,
        prefill_router::{PrefillCandidateSnapshot, PrefillRouter},
    },
    protocols::common::{
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        multimodal_epd::{
            MmObjectPlan, MmRoutingPlan, MmSourceKind, sglang_image_cache_key,
            vllm_embedding_cache_key,
        },
        preprocessor::{MmEpdPrefillSelection, MmEpdRoutingMode, MultimodalData},
    },
};

const ROUTING_MODE_ENV: &str = "DYN_MULTIMODAL_EPD_ROUTING_MODE";
const PREFILL_TOKEN_MS_ENV: &str = "DYN_MULTIMODAL_EPD_PREFILL_TOKEN_MS";
const PREFILL_LOAD_SCALE_ENV: &str = "DYN_MULTIMODAL_EPD_PREFILL_LOAD_SCALE";
const DECODE_BLOCK_MS_ENV: &str = "DYN_MULTIMODAL_EPD_DECODE_BLOCK_MS";
const ACTIVE_REQUEST_MS_ENV: &str = "DYN_MULTIMODAL_EPD_ACTIVE_REQUEST_MS";
const LOCAL_EC_SAVED_MS_ENV: &str = "DYN_MULTIMODAL_EPD_LOCAL_EC_SAVED_MS";
const REMOTE_TRANSFER_MS_ENV: &str = "DYN_MULTIMODAL_EPD_REMOTE_TRANSFER_MS";
const ENCODE_COMPUTE_MS_ENV: &str = "DYN_MULTIMODAL_EPD_ENCODE_COMPUTE_MS";
const FANOUT_PENALTY_MS_ENV: &str = "DYN_MULTIMODAL_EPD_FANOUT_PENALTY_MS";

pub(crate) fn cache_index_enabled_from_env() -> bool {
    std::env::var(ROUTING_MODE_ENV)
        .ok()
        .and_then(|value| value.parse::<MmEpdRoutingMode>().ok())
        .is_some_and(|mode| mode != MmEpdRoutingMode::Off)
}

impl FromStr for MmEpdRoutingMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "off" => Ok(Self::Off),
            "observe" => Ok(Self::Observe),
            "enforce" => Ok(Self::Enforce),
            value => anyhow::bail!(
                "invalid {ROUTING_MODE_ENV} value {value:?}; expected off, observe, or enforce"
            ),
        }
    }
}

pub struct MultimodalEpdRouter {
    prefill_router: Arc<PrefillRouter>,
    encoder_router: Arc<EncoderRouter>,
    mode: MmEpdRoutingMode,
    score_config: JointScoreConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObjectCacheKeyAliases {
    aliases: [String; 2],
    can_compute: bool,
}

#[derive(Debug, Clone, Copy)]
struct JointScoreConfig {
    prefill_token_ms: f64,
    prefill_load_scale: f64,
    decode_block_ms: f64,
    active_request_ms: f64,
    local_ec_saved_ms: f64,
    remote_transfer_ms: f64,
    encode_compute_ms: f64,
    fanout_penalty_ms: f64,
}

impl JointScoreConfig {
    fn from_env() -> Result<Self> {
        Ok(Self {
            prefill_token_ms: env_nonnegative_f64(PREFILL_TOKEN_MS_ENV, 0.002)?,
            prefill_load_scale: env_nonnegative_f64(PREFILL_LOAD_SCALE_ENV, 0.1)?,
            decode_block_ms: env_nonnegative_f64(DECODE_BLOCK_MS_ENV, 0.01)?,
            active_request_ms: env_nonnegative_f64(ACTIVE_REQUEST_MS_ENV, 0.05)?,
            local_ec_saved_ms: env_nonnegative_f64(LOCAL_EC_SAVED_MS_ENV, 1.0)?,
            remote_transfer_ms: env_nonnegative_f64(REMOTE_TRANSFER_MS_ENV, 0.05)?,
            encode_compute_ms: env_nonnegative_f64(ENCODE_COMPUTE_MS_ENV, 1.0)?,
            fanout_penalty_ms: env_nonnegative_f64(FANOUT_PENALTY_MS_ENV, 0.15)?,
        })
    }
}

fn env_nonnegative_f64(name: &str, default: f64) -> Result<f64> {
    let value = match std::env::var(name) {
        Ok(value) => value
            .parse::<f64>()
            .map_err(|error| anyhow::anyhow!("invalid {name} value {value:?}: {error}"))?,
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => return Err(error.into()),
    };
    anyhow::ensure!(
        value.is_finite() && value >= 0.0,
        "{name} must be finite and nonnegative"
    );
    Ok(value)
}

impl MultimodalEpdRouter {
    pub fn from_env(
        prefill_router: Arc<PrefillRouter>,
        encoder_router: Arc<EncoderRouter>,
    ) -> Result<Arc<Self>> {
        let mode = std::env::var(ROUTING_MODE_ENV)
            .unwrap_or_default()
            .parse()?;
        Ok(Arc::new(Self {
            prefill_router,
            encoder_router,
            mode,
            score_config: JointScoreConfig::from_env()?,
        }))
    }

    fn should_plan(request: &PreprocessedRequest) -> bool {
        !request.is_probe
            && request
                .multi_modal_data
                .as_ref()
                .is_some_and(|media| media.values().any(|items| !items.is_empty()))
    }

    fn record_selection(
        request: &mut PreprocessedRequest,
        mode: MmEpdRoutingMode,
        worker_id: u64,
        dp_rank: Option<u32>,
    ) {
        let routing_info = request.mm_routing_info.get_or_insert_with(Default::default);
        routing_info.epd_prefill_selection = Some(MmEpdPrefillSelection {
            mode,
            worker_id,
            dp_rank,
        });
        if mode == MmEpdRoutingMode::Enforce {
            let routing = request.routing_mut();
            routing.prefill_worker_id = Some(worker_id);
            routing.prefill_dp_rank = dp_rank;
        }
    }

    fn record_plan(request: &mut PreprocessedRequest, plan: MmRoutingPlan) {
        request
            .mm_routing_info
            .get_or_insert_with(Default::default)
            .epd_routing_plan = Some(plan);
    }

    fn ordered_image_cache_key_aliases(
        request: &PreprocessedRequest,
    ) -> Option<Vec<ObjectCacheKeyAliases>> {
        let media = request.multi_modal_data.as_ref()?;
        if media
            .iter()
            .any(|(modality, items)| modality != "image_url" && !items.is_empty())
        {
            return None;
        }
        let images = media.get("image_url")?;
        if images.is_empty() {
            return None;
        }
        images
            .iter()
            .map(|image| match image {
                MultimodalData::Url(url) => Some(ObjectCacheKeyAliases {
                    aliases: [
                        vllm_embedding_cache_key(url.as_str()),
                        sglang_image_cache_key(url.as_str()),
                    ],
                    can_compute: true,
                }),
                MultimodalData::RawUrl(url) => Some(ObjectCacheKeyAliases {
                    aliases: [vllm_embedding_cache_key(url), sglang_image_cache_key(url)],
                    can_compute: true,
                }),
                MultimodalData::UuidOnly(uuid) => Some(ObjectCacheKeyAliases {
                    aliases: [uuid.clone(), uuid.clone()],
                    can_compute: false,
                }),
                MultimodalData::Decoded(_) => None,
            })
            .collect()
    }

    fn build_plan(
        &self,
        request: &PreprocessedRequest,
        target_p_worker_id: u64,
    ) -> Option<MmRoutingPlan> {
        if !self
            .prefill_router
            .live_worker_ids()
            .contains(&target_p_worker_id)
        {
            return None;
        }

        let cache_key_aliases = Self::ordered_image_cache_key_aliases(request)?;
        let cache_keys = cache_key_aliases
            .iter()
            .map(|object| object.aliases[0].clone())
            .collect::<Vec<_>>();
        let can_compute = cache_key_aliases
            .iter()
            .map(|object| object.can_compute)
            .collect::<Vec<_>>();
        let live_encode_workers = self.encoder_router.live_worker_ids();
        let p_locations = cache_key_aliases
            .iter()
            .map(|object| {
                let mut workers = object
                    .aliases
                    .iter()
                    .flat_map(|cache_key| self.prefill_router.live_workers_for_cache_key(cache_key))
                    .collect::<Vec<_>>();
                workers.sort_unstable();
                workers.dedup();
                workers
            })
            .collect::<Vec<_>>();
        let e_locations = cache_key_aliases
            .iter()
            .map(|object| {
                let mut workers = object
                    .aliases
                    .iter()
                    .flat_map(|cache_key| self.encoder_router.live_workers_for_cache_key(cache_key))
                    .collect::<Vec<_>>();
                workers.sort_unstable();
                workers.dedup();
                workers
            })
            .collect::<Vec<_>>();
        Self::build_plan_from_locations(
            &cache_keys,
            target_p_worker_id,
            &live_encode_workers,
            &p_locations,
            &e_locations,
            &can_compute,
        )
    }

    fn build_plan_from_locations(
        cache_keys: &[String],
        target_p_worker_id: u64,
        live_encode_workers: &[u64],
        p_locations: &[Vec<u64>],
        e_locations: &[Vec<u64>],
        can_compute: &[bool],
    ) -> Option<MmRoutingPlan> {
        if cache_keys.len() != p_locations.len()
            || cache_keys.len() != e_locations.len()
            || cache_keys.len() != can_compute.len()
        {
            return None;
        }
        let mut objects = Vec::with_capacity(cache_keys.len());
        let mut source_use_counts = HashMap::<u64, usize>::new();

        for object_index in 0..cache_keys.len() {
            let (source_kind, source_worker_id) =
                if p_locations[object_index].contains(&target_p_worker_id) {
                    (MmSourceKind::PLocal, target_p_worker_id)
                } else if let Some(worker_id) = Self::preferred_remote_prefill_worker(
                    &p_locations[object_index],
                    target_p_worker_id,
                    &source_use_counts,
                ) {
                    (MmSourceKind::PRemote, worker_id)
                } else if let Some(worker_id) =
                    Self::preferred_encode_worker(&e_locations[object_index], &source_use_counts)
                {
                    (MmSourceKind::ECache, worker_id)
                } else {
                    if !can_compute[object_index] {
                        return None;
                    }
                    let worker_id =
                        Self::preferred_encode_worker(live_encode_workers, &source_use_counts)?;
                    (MmSourceKind::ECompute, worker_id)
                };

            if source_kind != MmSourceKind::PLocal {
                *source_use_counts.entry(source_worker_id).or_default() += 1;
            }

            objects.push(MmObjectPlan {
                object_index,
                source_kind,
                source_worker_id,
                source_worker_generation: source_worker_id,
                estimated_cost_ms: 0.0,
            });
        }

        Some(MmRoutingPlan {
            target_p_worker_id,
            target_p_generation: target_p_worker_id,
            objects,
            predicted_benefit_ms: 0.0,
            score_components: BTreeMap::new(),
        })
    }

    fn preferred_encode_worker(
        workers: &[u64],
        source_use_counts: &HashMap<u64, usize>,
    ) -> Option<u64> {
        workers.iter().copied().min_by(|left, right| {
            let left_count = source_use_counts.get(left).copied().unwrap_or_default();
            let right_count = source_use_counts.get(right).copied().unwrap_or_default();
            left_count.cmp(&right_count).then_with(|| left.cmp(right))
        })
    }

    fn preferred_remote_prefill_worker(
        workers: &[u64],
        target_p_worker_id: u64,
        source_use_counts: &HashMap<u64, usize>,
    ) -> Option<u64> {
        let remote_workers = workers
            .iter()
            .copied()
            .filter(|worker_id| *worker_id != target_p_worker_id)
            .collect::<Vec<_>>();
        Self::preferred_encode_worker(&remote_workers, source_use_counts)
    }

    fn score_plan(
        &self,
        candidate: &PrefillCandidateSnapshot,
        plan: MmRoutingPlan,
    ) -> MmRoutingPlan {
        let projected_prefill_ms = self
            .prefill_router
            .predict_prefill_duration_ms(
                candidate.potential_prefill_tokens,
                candidate.cached_tokens,
            )
            .unwrap_or(
                candidate.potential_prefill_tokens as f64 * self.score_config.prefill_token_ms,
            );
        Self::score_plan_with_projected_prefill_ms(
            candidate,
            plan,
            projected_prefill_ms,
            self.score_config,
        )
    }

    fn score_plan_with_projected_prefill_ms(
        candidate: &PrefillCandidateSnapshot,
        mut plan: MmRoutingPlan,
        projected_prefill_ms: f64,
        score_config: JointScoreConfig,
    ) -> MmRoutingPlan {
        let mut cache_hit_objects = 0usize;
        let mut transfer_cost_ms = 0.0;
        let mut compute_cost_ms = 0.0;
        let mut remote_sources = HashSet::new();

        for object in &mut plan.objects {
            object.estimated_cost_ms = match object.source_kind {
                MmSourceKind::PLocal => {
                    cache_hit_objects += 1;
                    0.0
                }
                MmSourceKind::PRemote | MmSourceKind::ECache => {
                    cache_hit_objects += 1;
                    remote_sources.insert(object.source_worker_id);
                    transfer_cost_ms += score_config.remote_transfer_ms;
                    score_config.remote_transfer_ms
                }
                MmSourceKind::ECompute => {
                    remote_sources.insert(object.source_worker_id);
                    transfer_cost_ms += score_config.remote_transfer_ms;
                    compute_cost_ms += score_config.encode_compute_ms;
                    score_config.remote_transfer_ms + score_config.encode_compute_ms
                }
            };
        }

        // TODO: Ablate each heuristic score component independently before choosing
        // production defaults. Calibrate the weights from measured TTFT and verify
        // which terms improve routing beyond KV overlap plus projected prefill cost.
        let kv_benefit_ms = candidate.cached_tokens as f64 * score_config.prefill_token_ms;
        let ec_saved_benefit_ms = cache_hit_objects as f64 * score_config.local_ec_saved_ms;
        let prefill_load_cost_ms = projected_prefill_ms * score_config.prefill_load_scale;
        let decode_load_cost_ms =
            candidate.potential_decode_blocks as f64 * score_config.decode_block_ms;
        let active_request_cost_ms =
            candidate.active_requests as f64 * score_config.active_request_ms;
        let fanout_cost_ms =
            remote_sources.len().saturating_sub(1) as f64 * score_config.fanout_penalty_ms;
        let joint_score_ms = kv_benefit_ms + ec_saved_benefit_ms
            - prefill_load_cost_ms
            - decode_load_cost_ms
            - active_request_cost_ms
            - transfer_cost_ms
            - compute_cost_ms
            - fanout_cost_ms;

        plan.predicted_benefit_ms = joint_score_ms;
        plan.score_components = BTreeMap::from([
            ("active_request_cost_ms".into(), active_request_cost_ms),
            ("cached_tokens".into(), candidate.cached_tokens as f64),
            ("compute_cost_ms".into(), compute_cost_ms),
            ("decode_load_cost_ms".into(), decode_load_cost_ms),
            ("fanout_cost_ms".into(), fanout_cost_ms),
            (
                "kv_overlap_blocks".into(),
                candidate.effective_overlap_blocks,
            ),
            ("kv_benefit_ms".into(), kv_benefit_ms),
            ("local_or_remote_ec_saved_ms".into(), ec_saved_benefit_ms),
            ("prefill_load_cost_ms".into(), prefill_load_cost_ms),
            ("projected_prefill_ms".into(), projected_prefill_ms),
            ("transfer_cost_ms".into(), transfer_cost_ms),
            ("joint_score_ms".into(), joint_score_ms),
        ]);
        plan
    }

    fn compare_scored_candidates(
        left: &(PrefillCandidateSnapshot, MmRoutingPlan),
        right: &(PrefillCandidateSnapshot, MmRoutingPlan),
    ) -> Ordering {
        left.1
            .predicted_benefit_ms
            .partial_cmp(&right.1.predicted_benefit_ms)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cached_tokens.cmp(&right.0.cached_tokens))
            .then_with(|| {
                right
                    .0
                    .potential_prefill_tokens
                    .cmp(&left.0.potential_prefill_tokens)
            })
            .then_with(|| right.0.worker_id.cmp(&left.0.worker_id))
            .then_with(|| right.0.dp_rank.cmp(&left.0.dp_rank))
    }

    fn select_best_plan(
        &self,
        request: &PreprocessedRequest,
        candidates: Vec<PrefillCandidateSnapshot>,
    ) -> Option<(PrefillCandidateSnapshot, MmRoutingPlan)> {
        candidates
            .into_iter()
            .filter_map(|candidate| {
                let plan = self.build_plan(request, candidate.worker_id)?;
                let plan = self.score_plan(&candidate, plan);
                Some((candidate, plan))
            })
            .max_by(Self::compare_scored_candidates)
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for MultimodalEpdRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let (mut request, context) = request.into_parts();
        if self.mode == MmEpdRoutingMode::Off || !Self::should_plan(&request) {
            return next.generate(context.map(|_| request)).await;
        }

        let (token_ids, block_mm_infos) = request.block_mm_routing_info();
        let routing = request.routing.clone().unwrap_or_default();
        let outcome = self
            .prefill_router
            .query_prefill_candidates(
                token_ids,
                block_mm_infos,
                routing.lora_name,
                routing.cache_namespace,
                routing.priority_jump.unwrap_or_default(),
                routing.strict_priority.unwrap_or_default(),
                routing.allowed_worker_ids,
                routing.routing_constraints.unwrap_or_default(),
            )
            .await;

        match outcome {
            Ok(candidates) => {
                let candidate_count = candidates.len();
                let selected = self.select_best_plan(&request, candidates);
                if let Some((candidate, plan)) = selected {
                    let worker_id = candidate.worker_id;
                    let dp_rank = Some(candidate.dp_rank);
                    let joint_score_ms = plan.predicted_benefit_ms;
                    let remote_source_count = plan
                        .objects
                        .iter()
                        .filter(|object| object.source_kind != MmSourceKind::PLocal)
                        .map(|object| object.source_worker_id)
                        .collect::<HashSet<_>>()
                        .len();
                    if self.mode == MmEpdRoutingMode::Observe {
                        Self::record_selection(&mut request, self.mode, worker_id, dp_rank);
                        Self::record_plan(&mut request, plan);
                        tracing::debug!(
                            mode = ?self.mode,
                            worker_id,
                            ?dp_rank,
                                candidate_count,
                                joint_score_ms,
                                remote_source_count,
                                "Observed joint KV/EC/load multimodal EPD plan"
                        );
                    } else {
                        Self::record_selection(&mut request, self.mode, worker_id, dp_rank);
                        Self::record_plan(&mut request, plan);
                        tracing::debug!(
                            mode = ?self.mode,
                            worker_id,
                            ?dp_rank,
                                candidate_count,
                                joint_score_ms,
                                remote_source_count,
                                "Enforcing joint KV/EC/load multimodal EPD plan"
                        );
                    }
                } else {
                    tracing::debug!(
                        mode = ?self.mode,
                        candidate_count,
                        "No complete multimodal EPD candidate plan; using legacy path"
                    );
                }
            }
            Err(error) => {
                tracing::debug!(
                    mode = ?self.mode,
                    %error,
                    "Multimodal EPD target-P observation failed; using legacy path"
                );
            }
        }

        next.generate(context.map(|_| request)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::{
        OutputOptions, SamplingOptions, StopConditions,
        preprocessor::{MultimodalData, MultimodalDataMap},
    };

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap()
    }

    fn multimodal_request(items: Vec<MultimodalData>) -> PreprocessedRequest {
        let mut request = request();
        request.multi_modal_data =
            Some(MultimodalDataMap::from([("image_url".to_string(), items)]));
        request
    }

    fn score_config() -> JointScoreConfig {
        JointScoreConfig {
            prefill_token_ms: 1.0,
            prefill_load_scale: 1.0,
            decode_block_ms: 0.0,
            active_request_ms: 0.0,
            local_ec_saved_ms: 5.0,
            remote_transfer_ms: 1.0,
            encode_compute_ms: 2.0,
            fanout_penalty_ms: 3.0,
        }
    }

    fn candidate(worker_id: u64, cached_tokens: usize) -> PrefillCandidateSnapshot {
        PrefillCandidateSnapshot {
            worker_id,
            dp_rank: 0,
            effective_overlap_blocks: cached_tokens as f64,
            cached_tokens,
            potential_prefill_tokens: 0,
            potential_decode_blocks: 0,
            active_requests: 0,
        }
    }

    #[test]
    fn routing_mode_defaults_off_and_rejects_unknown_values() {
        assert_eq!(
            "".parse::<MmEpdRoutingMode>().unwrap(),
            MmEpdRoutingMode::Off
        );
        assert_eq!(
            "observe".parse::<MmEpdRoutingMode>().unwrap(),
            MmEpdRoutingMode::Observe
        );
        assert!("other".parse::<MmEpdRoutingMode>().is_err());
    }

    #[test]
    fn observe_records_metadata_without_pinning_prefill() {
        let mut request = request();

        MultimodalEpdRouter::record_selection(&mut request, MmEpdRoutingMode::Observe, 7, Some(2));

        assert!(request.routing.is_none());
        assert_eq!(
            request
                .mm_routing_info
                .unwrap()
                .epd_prefill_selection
                .unwrap(),
            MmEpdPrefillSelection {
                mode: MmEpdRoutingMode::Observe,
                worker_id: 7,
                dp_rank: Some(2),
            }
        );
    }

    #[test]
    fn observed_plan_is_retained_without_enforcement() {
        let mut request = request();
        let plan = MmRoutingPlan {
            target_p_worker_id: 7,
            target_p_generation: 7,
            objects: vec![],
            predicted_benefit_ms: 1.0,
            score_components: BTreeMap::new(),
        };

        MultimodalEpdRouter::record_selection(&mut request, MmEpdRoutingMode::Observe, 7, None);
        MultimodalEpdRouter::record_plan(&mut request, plan.clone());

        assert!(request.routing.is_none());
        assert_eq!(
            request.mm_routing_info.unwrap().epd_routing_plan,
            Some(plan)
        );
    }

    #[test]
    fn enforce_records_and_pins_prefill() {
        let mut request = request();

        MultimodalEpdRouter::record_selection(&mut request, MmEpdRoutingMode::Enforce, 7, Some(2));

        let routing = request.routing.unwrap();
        assert_eq!(routing.prefill_worker_id, Some(7));
        assert_eq!(routing.prefill_dp_rank, Some(2));
    }

    #[test]
    fn plan_prefers_p_local_then_remote_p_then_e_cache_then_e_compute() {
        let cache_keys = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let plan = MultimodalEpdRouter::build_plan_from_locations(
            &cache_keys,
            10,
            &[20, 21],
            &[vec![10], vec![12], vec![], vec![]],
            &[vec![], vec![21], vec![21], vec![]],
            &[true, true, true, true],
        )
        .unwrap();

        assert_eq!(plan.target_p_worker_id, 10);
        assert_eq!(plan.target_p_generation, 10);
        assert_eq!(plan.objects.len(), 4);
        assert_eq!(plan.objects[0].source_kind, MmSourceKind::PLocal);
        assert_eq!(plan.objects[0].source_worker_id, 10);
        assert_eq!(plan.objects[1].source_kind, MmSourceKind::PRemote);
        assert_eq!(plan.objects[1].source_worker_id, 12);
        assert_eq!(plan.objects[2].source_kind, MmSourceKind::ECache);
        assert_eq!(plan.objects[2].source_worker_id, 21);
        assert_eq!(plan.objects[3].source_kind, MmSourceKind::ECompute);
        assert_eq!(plan.objects[3].source_worker_id, 20);
    }

    #[test]
    fn joint_score_allows_kv_to_outweigh_local_ec() {
        let remote_plan = MultimodalEpdRouter::build_plan_from_locations(
            &["a".to_string()],
            10,
            &[20],
            &[vec![]],
            &[vec![]],
            &[true],
        )
        .unwrap();
        let local_plan = MultimodalEpdRouter::build_plan_from_locations(
            &["a".to_string()],
            11,
            &[20],
            &[vec![11]],
            &[vec![]],
            &[true],
        )
        .unwrap();
        let remote_candidate = candidate(10, 10);
        let local_candidate = candidate(11, 0);
        let remote_scored = MultimodalEpdRouter::score_plan_with_projected_prefill_ms(
            &remote_candidate,
            remote_plan,
            0.0,
            score_config(),
        );
        let local_scored = MultimodalEpdRouter::score_plan_with_projected_prefill_ms(
            &local_candidate,
            local_plan,
            0.0,
            score_config(),
        );

        assert!(remote_scored.predicted_benefit_ms > local_scored.predicted_benefit_ms);
        assert_eq!(remote_scored.predicted_benefit_ms, 7.0);
        assert_eq!(local_scored.predicted_benefit_ms, 5.0);
    }

    #[test]
    fn joint_score_allows_local_ec_to_outweigh_modest_kv() {
        let remote_plan = MultimodalEpdRouter::build_plan_from_locations(
            &["a".to_string()],
            10,
            &[20],
            &[vec![]],
            &[vec![20]],
            &[true],
        )
        .unwrap();
        let local_plan = MultimodalEpdRouter::build_plan_from_locations(
            &["a".to_string()],
            11,
            &[20],
            &[vec![11]],
            &[vec![]],
            &[true],
        )
        .unwrap();
        let remote_candidate = candidate(10, 1);
        let local_candidate = candidate(11, 0);
        let mut config = score_config();
        config.prefill_token_ms = 0.5;
        let remote_scored = MultimodalEpdRouter::score_plan_with_projected_prefill_ms(
            &remote_candidate,
            remote_plan,
            0.0,
            config,
        );
        let local_scored = MultimodalEpdRouter::score_plan_with_projected_prefill_ms(
            &local_candidate,
            local_plan,
            0.0,
            config,
        );

        assert!(local_scored.predicted_benefit_ms > remote_scored.predicted_benefit_ms);
        assert_eq!(remote_scored.predicted_benefit_ms, 4.5);
        assert_eq!(local_scored.predicted_benefit_ms, 5.0);
    }

    #[test]
    fn projected_prefill_load_changes_target_p() {
        let plan = MultimodalEpdRouter::build_plan_from_locations(
            &["a".to_string()],
            10,
            &[],
            &[vec![10]],
            &[vec![]],
            &[false],
        )
        .unwrap();
        let low_load_candidate = candidate(10, 0);
        let high_load_candidate = candidate(11, 0);
        let low_load = MultimodalEpdRouter::score_plan_with_projected_prefill_ms(
            &low_load_candidate,
            plan.clone(),
            1.0,
            score_config(),
        );
        let mut high_load_plan = plan;
        high_load_plan.target_p_worker_id = 11;
        high_load_plan.target_p_generation = 11;
        high_load_plan.objects[0].source_worker_id = 11;
        high_load_plan.objects[0].source_worker_generation = 11;
        let high_load = MultimodalEpdRouter::score_plan_with_projected_prefill_ms(
            &high_load_candidate,
            high_load_plan,
            10.0,
            score_config(),
        );

        assert_eq!(
            MultimodalEpdRouter::compare_scored_candidates(
                &(low_load_candidate, low_load),
                &(high_load_candidate, high_load),
            ),
            Ordering::Greater
        );
    }

    #[test]
    fn transfer_and_fanout_costs_reduce_joint_score() {
        let single_source_plan = MultimodalEpdRouter::build_plan_from_locations(
            &["a".to_string(), "b".to_string()],
            10,
            &[20, 21],
            &[vec![], vec![]],
            &[vec![20], vec![20]],
            &[true, true],
        )
        .unwrap();
        let fanout_plan = MultimodalEpdRouter::build_plan_from_locations(
            &["a".to_string(), "b".to_string()],
            11,
            &[20, 21],
            &[vec![], vec![]],
            &[vec![20], vec![21]],
            &[true, true],
        )
        .unwrap();
        let single_source = MultimodalEpdRouter::score_plan_with_projected_prefill_ms(
            &candidate(10, 0),
            single_source_plan,
            0.0,
            score_config(),
        );
        let fanout = MultimodalEpdRouter::score_plan_with_projected_prefill_ms(
            &candidate(11, 0),
            fanout_plan,
            0.0,
            score_config(),
        );

        assert_eq!(single_source.score_components["fanout_cost_ms"], 0.0);
        assert_eq!(fanout.score_components["fanout_cost_ms"], 3.0);
        assert!(single_source.predicted_benefit_ms > fanout.predicted_benefit_ms);
    }

    #[test]
    fn compute_misses_are_balanced_across_encode_workers() {
        let cache_keys = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let plan = MultimodalEpdRouter::build_plan_from_locations(
            &cache_keys,
            10,
            &[20, 21],
            &[vec![], vec![], vec![], vec![]],
            &[vec![], vec![], vec![], vec![]],
            &[true, true, true, true],
        )
        .unwrap();

        assert_eq!(
            plan.objects
                .iter()
                .map(|object| object.source_worker_id)
                .collect::<Vec<_>>(),
            vec![20, 21, 20, 21]
        );
    }

    #[test]
    fn incomplete_remote_plan_fails_closed_without_encode_workers() {
        let cache_keys = vec!["a".to_string()];
        assert!(
            MultimodalEpdRouter::build_plan_from_locations(
                &cache_keys,
                10,
                &[],
                &[vec![]],
                &[vec![]],
                &[true],
            )
            .is_none()
        );
    }

    #[test]
    fn all_p_local_plan_does_not_require_encode_workers() {
        let cache_keys = vec!["a".to_string()];
        let plan = MultimodalEpdRouter::build_plan_from_locations(
            &cache_keys,
            10,
            &[],
            &[vec![10]],
            &[vec![]],
            &[false],
        )
        .unwrap();
        assert_eq!(plan.objects[0].source_kind, MmSourceKind::PLocal);
    }

    #[test]
    fn remote_p_plan_does_not_require_encode_workers() {
        let cache_keys = vec!["a".to_string()];
        let plan = MultimodalEpdRouter::build_plan_from_locations(
            &cache_keys,
            10,
            &[],
            &[vec![11]],
            &[vec![]],
            &[false],
        )
        .unwrap();
        assert_eq!(plan.objects[0].source_kind, MmSourceKind::PRemote);
        assert_eq!(plan.objects[0].source_worker_id, 11);
    }

    #[test]
    fn uuid_only_alias_is_cache_only() {
        let aliases =
            MultimodalEpdRouter::ordered_image_cache_key_aliases(&multimodal_request(vec![
                MultimodalData::UuidOnly("cached-image".into()),
            ]))
            .unwrap();

        assert_eq!(aliases[0].aliases, ["cached-image", "cached-image"]);
        assert!(!aliases[0].can_compute);
    }

    #[test]
    fn uuid_only_plan_uses_remote_holder_but_never_compute() {
        let cache_keys = vec!["cached-image".to_string()];
        let remote = MultimodalEpdRouter::build_plan_from_locations(
            &cache_keys,
            10,
            &[20],
            &[vec![]],
            &[vec![20]],
            &[false],
        )
        .unwrap();
        assert_eq!(remote.objects[0].source_kind, MmSourceKind::ECache);
        assert_eq!(remote.objects[0].source_worker_id, 20);

        assert!(
            MultimodalEpdRouter::build_plan_from_locations(
                &cache_keys,
                10,
                &[20],
                &[vec![]],
                &[vec![]],
                &[false],
            )
            .is_none()
        );
    }

    #[test]
    fn unsupported_modalities_fail_closed() {
        let mut video_request = request();
        video_request.multi_modal_data = Some(MultimodalDataMap::from([(
            "video_url".to_string(),
            vec![MultimodalData::RawUrl(
                "https://example.com/video.mp4".into(),
            )],
        )]));
        assert!(MultimodalEpdRouter::ordered_image_cache_key_aliases(&video_request).is_none());
    }
}
