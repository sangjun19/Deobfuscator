// Repository: markusmoenig/Eldiron
// File: creator/src/tools/info.rs

use crate::prelude::*;
use rusterix::Value;
use ToolEvent::*;

use crate::editor::INFOVIEWER;

pub struct InfoTool {
    id: TheId,

    info_mode: i32,
}

impl Tool for InfoTool {
    fn new() -> Self
    where
        Self: Sized,
    {
        Self {
            id: TheId::named("Info Tool"),
            info_mode: 0,
        }
    }
    fn id(&self) -> TheId {
        self.id.clone()
    }
    fn info(&self) -> String {
        str!("Info Tool.")
    }
    fn icon_name(&self) -> String {
        str!("info")
    }
    fn accel(&self) -> Option<char> {
        None //Some('x')
    }

    fn tool_event(
        &mut self,
        tool_event: ToolEvent,
        _tool_context: ToolContext,
        ui: &mut TheUI,
        ctx: &mut TheContext,
        _project: &mut Project,
        server_ctx: &mut ServerContext,
    ) -> bool {
        match tool_event {
            Activate => {
                ctx.ui.send(TheEvent::SetStackIndex(
                    TheId::named("Main Stack"),
                    PanelIndices::InfoViewer as usize,
                ));

                if let Some(layout) = ui.get_hlayout("Game Tool Params") {
                    layout.clear();

                    let mut info_switch = TheGroupButton::new(TheId::named("Info Switch"));
                    info_switch
                        .add_text_status("Attributes".to_string(), "Show attributes.".to_string());
                    info_switch.add_text_status(
                        "Inventory".to_string(),
                        "Show the inventory.".to_string(),
                    );

                    info_switch.set_item_width(80);
                    info_switch.set_index(self.info_mode);
                    layout.add_widget(Box::new(info_switch));
                }

                // ui.set_widget_value("InfoView", ctx, TheValue::Text(project.config.clone()));
                server_ctx.curr_map_tool_type = MapToolType::General;

                INFOVIEWER.write().unwrap().visible = true;
                true
            }
            DeActivate => {
                INFOVIEWER.write().unwrap().visible = false;
                true
            }
            _ => false,
        }
    }

    fn handle_event(
        &mut self,
        event: &TheEvent,
        _ui: &mut TheUI,
        _ctx: &mut TheContext,
        _project: &mut Project,
        _server_ctx: &mut ServerContext,
    ) -> bool {
        #[allow(clippy::single_match)]
        match event {
            TheEvent::IndexChanged(id, index) => {
                if id.name == "Info Switch" {
                    self.info_mode = *index as i32;
                    INFOVIEWER.write().unwrap().info_mode = self.info_mode;
                }
            }
            TheEvent::KeyDown(TheValue::Char(char)) => {
                let mut rusterix = crate::editor::RUSTERIX.write().unwrap();
                if rusterix.server.state == rusterix::ServerState::Running {
                    rusterix
                        .server
                        .local_player_event("key_down".into(), Value::Str(char.to_string()));
                }
            }
            TheEvent::KeyUp(TheValue::Char(char)) => {
                let mut rusterix = crate::editor::RUSTERIX.write().unwrap();
                if rusterix.server.state == rusterix::ServerState::Running {
                    rusterix
                        .server
                        .local_player_event("key_up".into(), Value::Str(char.to_string()));
                }
            }
            _ => {}
        }

        false
    }
}
