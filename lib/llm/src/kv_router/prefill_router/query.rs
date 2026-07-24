// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use dynamo_kv_router::protocols::{BlockExtraInfo, RoutingConstraints, WorkerId, WorkerWithDpRank};

use super::{
    InnerPrefillRouter, PrefillCandidateSnapshot, PrefillError, PrefillLifecycleState,
    PrefillQueryOutcome, PrefillRouter,
};

impl PrefillRouter {
    /// Query the best prefill worker without executing a request.
    ///
    /// This query is advisory and does not book scheduler or occupancy state;
    /// concurrent callers may observe the same worker.
    #[expect(clippy::too_many_arguments)]
    pub async fn query_prefill_worker(
        &self,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> Result<PrefillQueryOutcome> {
        if self.lifecycle_state() != PrefillLifecycleState::Active {
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        }
        let prefill_router = self
            .prefill_router
            .get()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;

        match prefill_router {
            InnerPrefillRouter::KvRouter(router) => {
                let outcome = router
                    .chooser
                    .find_best_match_details(
                        None,
                        token_ids,
                        block_mm_infos,
                        None,
                        false,
                        false,
                        lora_name,
                        cache_namespace,
                        priority_jump,
                        strict_priority,
                        None,
                        None,
                        allowed_worker_ids,
                        routing_constraints,
                    )
                    .await?;
                match outcome {
                    crate::kv_router::FindBestMatchOutcome::Routed { worker, .. } => {
                        Ok(PrefillQueryOutcome::Routed {
                            worker_id: worker.worker_id,
                            dp_rank: Some(worker.dp_rank),
                        })
                    }
                    crate::kv_router::FindBestMatchOutcome::QueueRejected { rejection } => {
                        Ok(PrefillQueryOutcome::QueueRejected { rejection })
                    }
                }
            }
            InnerPrefillRouter::SimpleRouter(router) => {
                let worker_id = router
                    .peek_next_worker()
                    .ok_or_else(|| anyhow::anyhow!("No workers available for prefill"))?;
                Ok(PrefillQueryOutcome::Routed {
                    worker_id,
                    dp_rank: None,
                })
            }
        }
    }

    #[expect(clippy::too_many_arguments)]
    pub async fn query_prefill_candidates(
        &self,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> Result<Vec<PrefillCandidateSnapshot>> {
        if self.lifecycle_state() != PrefillLifecycleState::Active {
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        }
        let prefill_router = self
            .prefill_router
            .get()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;

        match prefill_router {
            InnerPrefillRouter::KvRouter(router) => {
                let mut worker_ids = self.live_worker_ids();
                if let Some(allowed_worker_ids) = allowed_worker_ids.as_ref() {
                    worker_ids.retain(|worker_id| allowed_worker_ids.contains(worker_id));
                }
                worker_ids.sort_unstable();

                let mut routed = Vec::with_capacity(worker_ids.len());
                for worker_id in worker_ids {
                    let singleton = HashSet::from([worker_id]);
                    match router
                        .chooser
                        .find_best_match_details(
                            None,
                            token_ids,
                            block_mm_infos,
                            None,
                            false,
                            false,
                            lora_name.clone(),
                            cache_namespace.clone(),
                            priority_jump,
                            strict_priority,
                            None,
                            None,
                            Some(singleton),
                            routing_constraints.clone(),
                        )
                        .await
                    {
                        Ok(crate::kv_router::FindBestMatchOutcome::Routed {
                            worker,
                            effective_overlap_blocks,
                            cached_tokens,
                            ..
                        }) => routed.push((worker, effective_overlap_blocks, cached_tokens)),
                        Ok(crate::kv_router::FindBestMatchOutcome::QueueRejected { .. }) => {}
                        Err(error) => {
                            tracing::debug!(worker_id, %error, "prefill candidate is not schedulable")
                        }
                    }
                }

                let effective_cached_tokens = routed
                    .iter()
                    .map(|(worker, _, cached_tokens)| (*worker, *cached_tokens))
                    .collect::<HashMap<WorkerWithDpRank, usize>>();
                let loads = router
                    .chooser
                    .get_scheduler_potential_loads(token_ids.len(), effective_cached_tokens)
                    .into_iter()
                    .map(|load| ((load.worker_id, load.dp_rank), load))
                    .collect::<HashMap<_, _>>();

                Ok(routed
                    .into_iter()
                    .map(|(worker, effective_overlap_blocks, cached_tokens)| {
                        let effective_prefill_tokens =
                            token_ids.len().saturating_sub(cached_tokens);
                        let load = loads.get(&(worker.worker_id, worker.dp_rank));
                        PrefillCandidateSnapshot {
                            worker_id: worker.worker_id,
                            dp_rank: worker.dp_rank,
                            effective_overlap_blocks,
                            cached_tokens,
                            potential_prefill_tokens: load
                                .map_or(effective_prefill_tokens, |load| {
                                    load.potential_prefill_tokens
                                }),
                            potential_decode_blocks: load
                                .map_or(0, |load| load.potential_decode_blocks),
                            active_requests: load.map_or(0, |load| load.active_requests),
                        }
                    })
                    .collect())
            }
            InnerPrefillRouter::SimpleRouter(router) => {
                let Some(worker_id) = router.peek_next_worker() else {
                    return Ok(Vec::new());
                };
                if allowed_worker_ids
                    .as_ref()
                    .is_some_and(|allowed| !allowed.contains(&worker_id))
                {
                    return Ok(Vec::new());
                }
                Ok(vec![PrefillCandidateSnapshot {
                    worker_id,
                    dp_rank: 0,
                    effective_overlap_blocks: 0.0,
                    cached_tokens: 0,
                    potential_prefill_tokens: token_ids.len(),
                    potential_decode_blocks: 0,
                    active_requests: 0,
                }])
            }
        }
    }

    pub fn predict_prefill_duration_ms(&self, effective_isl: usize, prefix: usize) -> Option<f64> {
        self.prefill_load_estimator
            .as_ref()?
            .predict_prefill_duration(1, effective_isl, prefix)
            .ok()
            .map(|duration| duration.as_secs_f64() * 1_000.0)
    }

    pub fn register_workers(&self, worker_ids: &HashSet<WorkerId>) {
        if let Some(InnerPrefillRouter::KvRouter(router)) = self.prefill_router.get() {
            router.chooser.register_workers(worker_ids);
        }
    }
}
