// Repository: JichouP/switchbot-exporter
// File: src/infrastructure/api/switchbot.rs

use self::client::SwitchBotClient;
use crate::domain::switchbot::{
    get_devices::GetDevicesResponse,
    get_devices_status::{GetDevicesMeterPlusStatusResponse, GetDevicesPlugMiniStatusResponse},
    SwitchBotApi,
};
mod client;

pub struct SwitchBotApiImpl {}

impl SwitchBotApiImpl {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl SwitchBotApi for SwitchBotApiImpl {
    async fn get_devices(&self) -> anyhow::Result<GetDevicesResponse> {
        let client = SwitchBotClient::new();
        client.get::<GetDevicesResponse>("/devices").await
    }

    async fn get_meter_plus_devices_status(
        &self,
        device_id: &str,
    ) -> anyhow::Result<GetDevicesMeterPlusStatusResponse> {
        let client = SwitchBotClient::new();
        client
            .get::<GetDevicesMeterPlusStatusResponse>(&format!("/devices/{}/status", device_id))
            .await
    }

    async fn get_plug_mini_devices_status(
        &self,
        device_id: &str,
    ) -> anyhow::Result<GetDevicesPlugMiniStatusResponse> {
        let client = SwitchBotClient::new();
        client
            .get::<GetDevicesPlugMiniStatusResponse>(&format!("/devices/{}/status", device_id))
            .await
    }
}
