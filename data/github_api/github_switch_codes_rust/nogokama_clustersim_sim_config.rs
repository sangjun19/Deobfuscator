// Repository: nogokama/clustersim
// File: simulator/src/config/sim_config.rs

use serde::{Deserialize, Serialize};

use dslab_core::Id;

use crate::networks::resolver::NetworkType;

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct HostConfig {
    pub id: Id,
    pub name: String,
    pub group_prefix: Option<String>,
    pub trace_id: Option<u64>,
    pub cpus: u32,
    pub memory: u64,

    pub cpu_speed: Option<f64>,
    pub disk_capacity: Option<u64>,
    pub disk_read_bw: Option<f64>,
    pub disk_write_bw: Option<f64>,
    pub local_newtork_bw: Option<f64>,
    pub local_newtork_latency: Option<f64>,
}

impl HostConfig {
    pub fn from_group_config(group: &GroupHostConfig, idx: Option<u32>) -> Self {
        let mut group_prefix = None;
        let name = if group.count.unwrap_or(1) == 1 {
            group
                .name
                .clone()
                .unwrap_or_else(|| panic!("name is required for host group with count = 1"))
        } else {
            group_prefix = Some(group.name_prefix.clone().unwrap());
            format!(
                "{}-{}",
                group.name_prefix.clone().unwrap_or_else(|| panic!(
                    "name_prefix is required for host group with count > 1"
                )),
                idx.unwrap()
            )
        };
        Self {
            id: 0,
            name,
            group_prefix,
            cpus: group.cpus,
            trace_id: None,
            memory: group.memory,
            cpu_speed: group.cpu_speed,
            disk_capacity: group.disk_capacity,
            disk_read_bw: group.disk_read_bw,
            disk_write_bw: group.disk_write_bw,
            local_newtork_bw: group.local_newtork_bw,
            local_newtork_latency: group.local_newtork_latency,
        }
    }

    pub fn from_cpus_memory(id: u64, cpus: u32, memory: u64) -> Self {
        Self {
            id: 0,
            trace_id: Some(id),
            name: format!("host-{}", id),
            group_prefix: None,
            cpus,
            memory,
            cpu_speed: None,
            disk_capacity: None,
            disk_read_bw: None,
            disk_write_bw: None,
            local_newtork_bw: None,
            local_newtork_latency: None,
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct TraceHostsConfig {
    pub path: String,
    pub resources_multiplier: f64,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RawSimulationConfig {
    pub hosts: Option<Vec<GroupHostConfig>>,
    pub trace_hosts: Option<TraceHostsConfig>,
    pub scheduler: Option<SchedulerConfig>,
    pub workload: Option<Vec<ClusterWorkloadConfig>>,
    pub network: Option<NetworkConfig>,
    pub monitoring: Option<MonitoringConfig>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct GroupHostConfig {
    pub name: Option<String>,
    pub name_prefix: Option<String>,

    pub cpus: u32,
    pub memory: u64,

    pub cpu_speed: Option<f64>,
    pub disk_capacity: Option<u64>,
    pub disk_read_bw: Option<f64>,
    pub disk_write_bw: Option<f64>,
    pub local_newtork_bw: Option<f64>,
    pub local_newtork_latency: Option<f64>,

    pub count: Option<u32>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct NetworkLinkConfig {
    pub bandwidth: f64,
    pub latency: f64,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct NetworkConfig {
    pub r#type: NetworkType,
    pub local: NetworkLinkConfig,
    pub global: Option<NetworkLinkConfig>,
    pub uplink: Option<NetworkLinkConfig>,
    pub downlink: Option<NetworkLinkConfig>,
    pub switch: Option<NetworkLinkConfig>,
    pub l1_switch_count: Option<usize>,
    pub l2_switch_count: Option<usize>,
}

#[derive(Default, Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct SchedulerConfig {
    pub hosts_invoke_interval: Option<f64>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct ClusterWorkloadConfig {
    pub r#type: String,
    pub path: Option<String>,
    pub options: Option<serde_yaml::Value>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum MonitoringLevel {
    None,
    Basic,
    Groups,
    Detailed,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Default)]
pub struct MonitoringConfig {
    pub host_load_compression_time_interval: Option<f64>,
    pub scheduler_queue_compression_time_interval: Option<f64>,
    pub display_host_load: Option<bool>,
    pub collect_user_queues: Option<bool>,
    pub host_logs_file_name: Option<String>,
    pub scheduler_queue_logs_file_name: Option<String>,
    pub fair_share_logs_file_name: Option<String>,
    pub collect_executions_scheduled_time: Option<bool>,
    pub output_dir: Option<String>,
}

/// Represents simulation configuration.
#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct SimulationConfig {
    /// Used trace dataset.
    pub workload: Option<Vec<ClusterWorkloadConfig>>,
    /// Configurations of physical hosts.
    pub hosts: Vec<GroupHostConfig>,
    pub trace_hosts: Option<TraceHostsConfig>,
    /// Configurations of schedulers.
    pub scheduler: SchedulerConfig,
    pub network: Option<NetworkConfig>,
    pub monitoring: Option<MonitoringConfig>,
}

impl SimulationConfig {
    /// Creates simulation config by reading parameter values from YAM file
    /// (uses default values if some parameters are absent).
    pub fn from_file(file_name: &str) -> Self {
        let raw: RawSimulationConfig = serde_yaml::from_str(
            &std::fs::read_to_string(file_name)
                .unwrap_or_else(|e| panic!("Can't read file {}: {}", file_name, e)),
        )
        .unwrap_or_else(|e| panic!("Can't parse YAML from file {}: {}", file_name, e));

        Self {
            workload: raw.workload,
            hosts: raw.hosts.unwrap_or_default(),
            trace_hosts: raw.trace_hosts,
            scheduler: raw.scheduler.unwrap_or_default(),
            network: raw.network,
            monitoring: raw.monitoring,
        }
    }
}
