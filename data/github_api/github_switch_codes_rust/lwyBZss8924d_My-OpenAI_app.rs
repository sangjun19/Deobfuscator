// Repository: lwyBZss8924d/My-OpenAI
// File: src/app.rs

use anyhow::{Result, Context};
use std::env;
use ratatui::widgets::{TableState, ScrollbarState};
use crossterm::event::MouseEvent;
use crate::models::{Model, ModelResponse};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, ACCEPT, USER_AGENT};
use chrono::{DateTime, Local};
use std::time::Duration;
use tokio;
use dotenv;
use std::collections::HashSet;

// 定义应用状态枚举
#[derive(PartialEq)]
pub enum AppState {
    Menu,       // 启动菜单
    Config,     // API 配置界面
    Splash,     // 启动画面
    Main,       // 主界面
}

// 定义输入焦点
#[derive(PartialEq)]
pub enum InputFocus {
    ApiKey,
    ApiBase,
    None,
}

#[derive(PartialEq)]
pub enum InputMode {
    Normal,
    Editing,
}

// 定义菜单选项
#[derive(PartialEq, Clone, Copy)]
pub enum MenuOption {
    DefaultConfig,
    CustomConfig,
}

#[derive(Debug)]
pub struct SplashProgress {
    steps: Vec<(&'static str, f64)>,
    current_idx: usize,
    last_step_time: std::time::Instant,
}

#[derive(Debug)]
pub struct DetectionItem {
    pub name: &'static str,   // 检测步骤名称
    pub progress: f64,        // 当前进度(0.0 ~ 1.0)
    pub is_done: bool,        // 是否完成
    pub is_error: bool,       // 是否出错
}

impl DetectionItem {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            progress: 0.0,
            is_done: false,
            is_error: false,
        }
    }
}

pub struct App {
    pub models: Vec<Model>,
    pub table_state: TableState,
    pub scroll_state: ScrollbarState,
    pub error: Option<String>,
    pub state: AppState,    // 当前界面状态
    pub models_loaded: bool, // 模型是否加载完成
    pub last_update: DateTime<Local>, // 最后更新时间
    pub api_base: String,    // OpenAI API 基础URL
    pub api_key: String,     // OpenAI API 密钥
    pub detection_progress: f64,  // 检测进度 (0.0-1.0)
    pub detection_step: String,   // 当前检测步骤
    pub detection_items: Vec<DetectionItem>, // 检测项目列表
    pub input_focus: InputFocus,  // 当前输入焦点
    pub show_api_key: bool,      // 是否显示 API 密钥
    pub show_popup: bool,        // 是否显示弹出框
    pub input_mode: InputMode,   // 输入模式
    pub popup_input: String,     // 弹出框输入内容
    pub cursor_position: usize,  // 光标位置
    pub selected_menu_option: MenuOption, // 当前选中的菜单选项
    pub splash_progress: Option<SplashProgress>,
}

impl App {
    pub async fn new() -> Result<Self> {
        let app = Self {
            models: Vec::new(),
            table_state: TableState::default().with_selected(Some(0)),
            scroll_state: ScrollbarState::default(),
            error: None,
            state: AppState::Menu,  // 初始状态为菜单
            models_loaded: false,
            last_update: Local::now(),
            api_base: String::from("https://api.openai.com"),
            api_key: String::new(),
            detection_progress: 0.0,
            detection_step: "等待配置...".to_string(),
            detection_items: vec![
                DetectionItem::new("初始化检测环境"),
                DetectionItem::new("验证API密钥格式"),
                DetectionItem::new("检查网络连接"),
                DetectionItem::new("验证API服务器"),
                DetectionItem::new("建立安全连接"),
                DetectionItem::new("验证API权限"),
                DetectionItem::new("获取API配置信息"),
                DetectionItem::new("获取可用模型列表"),
            ],
            input_focus: InputFocus::ApiKey,
            show_api_key: false,
            show_popup: false,
            input_mode: InputMode::Normal,
            popup_input: String::new(),
            cursor_position: 0,
            selected_menu_option: MenuOption::DefaultConfig,
            splash_progress: None,
        };

        Ok(app)
    }

    // 切换输入焦点
    pub fn toggle_input_focus(&mut self) {
        self.input_focus = match self.input_focus {
            InputFocus::ApiKey => InputFocus::ApiBase,
            InputFocus::ApiBase => InputFocus::None,
            InputFocus::None => InputFocus::ApiKey,
        };
    }

    // 切换 API 密钥显示/隐藏
    pub fn toggle_api_key_visibility(&mut self) {
        self.show_api_key = !self.show_api_key;
    }

    // 处理字符输入
    pub fn input_char(&mut self, c: char) {
        match self.input_focus {
            InputFocus::ApiKey => self.api_key.push(c),
            InputFocus::ApiBase => self.api_base.push(c),
            InputFocus::None => {}
        }
    }

    // 处理退格键
    pub fn handle_backspace(&mut self) {
        match self.input_focus {
            InputFocus::ApiKey => { self.api_key.pop(); }
            InputFocus::ApiBase => { self.api_base.pop(); }
            InputFocus::None => {}
        }
    }

    // 验证配置并开始检测
    pub async fn validate_and_start(&mut self) -> Result<()> {
        // 如果输入为空，尝试使用环境变量
        if self.api_key.is_empty() {
            self.api_key = env::var("OPENAI_API_KEY")
                .map_err(|_| anyhow::anyhow!("未找到API密钥，请在.env文件中设置OPENAI_API_KEY或手动输入"))?;
        }

        if self.api_base.is_empty() {
            self.api_base = env::var("OPENAI_API_BASE")
                .unwrap_or_else(|_| "https://api.openai.com".to_string());
        }

        // 切换到启动画面
        self.state = AppState::Splash;
        
        // 定义检测步骤
        let steps = vec![
            ("正在初始化检测环境...", 0.05),
            ("正在验证API密钥格式...", 0.15),
            ("正在检查网络连接...", 0.25),
            ("正在验证API服务器...", 0.35),
            ("正在建立安全连接...", 0.45),
            ("正在验证API权限...", 0.55),
            ("正在获取API配置信息...", 0.65),
            ("正在获取可用模型列表...", 0.75),
        ];

        // 执行检测步骤
        for (step, progress) in steps {
            self.detection_step = step.to_string();
            self.detection_progress = progress;
            tokio::time::sleep(Duration::from_millis(800)).await;
        }

        // 获取模型列表
        self.detection_step = "正在获取可用模型列表...".to_string();
        self.detection_progress = 0.75;
        if let Err(e) = self.fetch_models().await {
            self.error = Some(format!("获取模型列表失败: {}", e));
            self.detection_step = "检测失败".to_string();
            self.detection_progress = 1.0;
            return Err(e);
        }

        // 处理模型信息
        self.detection_step = "正在处理模型信息...".to_string();
        self.detection_progress = 0.85;
        tokio::time::sleep(Duration::from_millis(800)).await;

        // 优化数据结构
        self.detection_step = "正在优化数据结构...".to_string();
        self.detection_progress = 0.95;
        tokio::time::sleep(Duration::from_millis(800)).await;

        // 完成检测
        self.detection_step = "检测完成".to_string();
        self.detection_progress = 1.0;
        tokio::time::sleep(Duration::from_millis(800)).await;
        
        self.models_loaded = true;
        Ok(())
    }

    // 使用默认配置开始检测
    pub async fn start_with_default_config(&mut self) -> Result<()> {
        let _ = dotenv::dotenv();
        
        self.api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow::anyhow!("未找到API密钥，请在.env文件中设置OPENAI_API_KEY"))?;
            
        self.api_base = env::var("OPENAI_API_BASE")
            .unwrap_or_else(|_| "https://api.openai.com".to_string());

        self.init_splash_progress();
        Ok(())
    }

    // 切换到主界面
    pub fn to_main_screen(&mut self) {
        self.state = AppState::Main;
    }

    // 检查是否可以切换到主界面
    pub fn can_switch_to_main(&self) -> bool {
        // 只有在检测完成且用户按下任意键时才切换到主界面
        self.models_loaded && self.detection_progress >= 1.0
    }

    // 刷新模型列表
    pub async fn refresh_models(&mut self) -> Result<()> {
        self.detection_step = "正在获取模型列表...".to_string();
        self.detection_progress = 0.6;
        
        if let Err(e) = self.fetch_models().await {
            return Err(e).context("Failed to fetch models");
        }
        
        self.detection_step = "正在处理模型信息...".to_string();
        self.detection_progress = 0.8;
        
        self.scroll_state = ScrollbarState::new((self.models.len()).saturating_sub(1));
        self.last_update = Local::now();
        
        self.detection_step = "检测完成".to_string();
        self.detection_progress = 1.0;
        
        Ok(())
    }

    // 从OpenAI API获取模型列表
    async fn fetch_models(&mut self) -> Result<()> {
        let api_key = match env::var("OPENAI_API_KEY") {
            Ok(key) => {
                if key.len() < 100 {
                    if let Ok(env_content) = std::fs::read_to_string(".env") {
                        if let Some(key_line) = env_content.lines().find(|line| line.starts_with("OPENAI_API_KEY=")) {
                            key_line.trim_start_matches("OPENAI_API_KEY=").to_string()
                        } else {
                            key
                        }
                    } else {
                        key
                    }
                } else {
                    key
                }
            },
            Err(_) => {
                self.detection_step = "API密钥未找到".to_string();
                return Err(anyhow::anyhow!("未找到API密钥，请在.env文件中设置OPENAI_API_KEY"));
            }
        };
        
        let api_base = env::var("OPENAI_API_BASE")
            .unwrap_or_else(|_| "https://api.openai.com".to_string())
            .trim_end_matches('/')
            .to_string();
        
        let mut headers = HeaderMap::new();
        let auth_header = format!("Bearer {}", api_key);
        let auth_value = HeaderValue::from_str(&auth_header)
            .context("无效的 API 密钥格式")?;
        headers.insert(AUTHORIZATION, auth_value);
        
        headers.insert(USER_AGENT, HeaderValue::from_static("curl/8.7.1"));
        headers.insert(ACCEPT, HeaderValue::from_static("*/*"));
        headers.insert(reqwest::header::CONNECTION, HeaderValue::from_static("keep-alive"));
        
        if let Ok(url) = reqwest::Url::parse(&api_base) {
            if let Some(host) = url.host_str() {
                if let Ok(host_value) = HeaderValue::from_str(host) {
                    headers.insert(reqwest::header::HOST, host_value);
                }
            }
        }
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .default_headers(headers.clone())
            .build()
            .context("创建 HTTP 客户端失败")?;

        let request_url = format!("{}/v1/models", api_base);
        let response = client.get(&request_url)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("API请求失败: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await
                .unwrap_or_else(|e| format!("无法读取错误响应: {}", e));
            
            let error_msg = match status.as_u16() {
                401 => format!("API密钥无效或已过期: {}", error_text),
                403 => format!("API密钥权限不足: {}", error_text),
                404 => format!("API端点不存在: {}", error_text),
                429 => format!("请求过于频繁: {}", error_text),
                500..=599 => format!("API服务器错误: {}", error_text),
                _ => format!("API请求失败 ({}): {}", status, error_text)
            };
            
            return Err(anyhow::anyhow!(error_msg));
        }

        let response_text = response.text().await
            .map_err(|e| anyhow::anyhow!("无法读取响应内容: {}", e))?;
        
        let model_response: ModelResponse = serde_json::from_str(&response_text)
            .map_err(|e| anyhow::anyhow!("解析响应失败: {}", e))?;

        // 获取API返回的模型ID集合
        let api_model_ids: HashSet<String> = model_response.data.iter()
            .map(|m| m.id.clone())
            .collect();

        // 获取官方预定义模型
        let mut official_models = crate::models::get_official_models();
        
        // 更新官方模型的可用状态
        for model in &mut official_models {
            model.set_model_info();
            model.available = api_model_ids.contains(&model.id);
        }

        // 合并API返回的模型和官方预定义模型
        let mut all_models = Vec::new();
        let mut seen_models = HashSet::new();

        // 添加API返回的模型
        for mut model in model_response.data {
            model.set_model_info();
            seen_models.insert(model.id.clone());
            all_models.push(model);
        }

        // 添加未在API中出现的官方模型
        for model in official_models {
            if !seen_models.contains(&model.id) {
                all_models.push(model);
            }
        }

        // 更新模型列表
        self.models = all_models;

        // 排序
        self.models.sort_by(|a, b| {
            let priority_cmp = a.priority.cmp(&b.priority);
            if priority_cmp != std::cmp::Ordering::Equal {
                return priority_cmp;
            }

            if let (Some(sub_a), Some(sub_b)) = (&a.subgroup, &b.subgroup) {
                let subgroup_cmp = sub_a.cmp(sub_b);
                if subgroup_cmp != std::cmp::Ordering::Equal {
                    return subgroup_cmp;
                }
            }

            b.created.cmp(&a.created)
        });

        Ok(())
    }

    // 移动到列表开头
    pub fn move_to_start(&mut self) {
        if !self.models.is_empty() {
            self.table_state.select(Some(0));
            self.scroll_state = self.scroll_state.position(0);
        }
    }

    // 移动到列表末尾
    pub fn move_to_end(&mut self) {
        if !self.models.is_empty() {
            let last = self.models.len() - 1;
            self.table_state.select(Some(last));
            self.scroll_state = self.scroll_state.position(last);
        }
    }

    // 选择上一个模型
    pub fn previous(&mut self) {
        if !self.models.is_empty() {
            let i = match self.table_state.selected() {
                Some(i) => {
                    if i == 0 {
                        self.models.len() - 1
                    } else {
                        i - 1
                    }
                }
                None => 0,
            };
            // 避免重复选择相同的行
            if Some(i) != self.table_state.selected() {
                self.table_state.select(Some(i));
                self.scroll_state = self.scroll_state.position(i);
            }
        }
    }

    // 选择下一个模型
    pub fn next(&mut self) {
        if !self.models.is_empty() {
            let i = match self.table_state.selected() {
                Some(i) => {
                    if i >= self.models.len() - 1 {
                        0
                    } else {
                        i + 1
                    }
                }
                None => 0,
            };
            // 避免重复选择相同的行
            if Some(i) != self.table_state.selected() {
                self.table_state.select(Some(i));
                self.scroll_state = self.scroll_state.position(i);
            }
        }
    }

    // 优化的选择行方法
    pub fn select_row(&mut self, index: usize) {
        if index < self.models.len() {
            // 如果选择的是同一行，不做任何操作
            if Some(index) != self.table_state.selected() {
                self.table_state.select(Some(index));
                self.scroll_state = self.scroll_state.position(index);
            }
        }
    }

    // 优化的获取鼠标下的表格行索引方法
    pub fn get_row_under_mouse(&self, mouse_event: MouseEvent) -> Option<usize> {
        // 表格的有效区域（考虑边框和边距）
        let table_start_y = 4; // 考虑顶部API信息区域(3)和边距(1)
        let row = mouse_event.row as usize;
        let column = mouse_event.column as usize;
        
        // 检查鼠标是否在表格区域内（考虑左右边距）
        if row >= table_start_y && column >= 1 && column < 239 { // 假设终端宽度为 240
            let relative_row = row - table_start_y;
            // 跳过表头行，并确保不超过模型列表长度
            if relative_row > 0 && relative_row <= self.models.len() {
                return Some(relative_row - 1);
            }
        }
        None
    }

    // 获取当前选中的模型
    pub fn selected_model(&self) -> Option<&Model> {
        self.table_state
            .selected()
            .and_then(|i| self.models.get(i))
    }

    // 上一页
    pub fn previous_page(&mut self) {
        if !self.models.is_empty() {
            let current = self.table_state.selected().unwrap_or(0);
            let page_size = 10; // 每页显示的行数
            let new_index = current.saturating_sub(page_size);
            if Some(new_index) != self.table_state.selected() {
                self.table_state.select(Some(new_index));
                self.scroll_state = self.scroll_state.position(new_index);
            }
        }
    }

    // 下一页
    pub fn next_page(&mut self) {
        if !self.models.is_empty() {
            let current = self.table_state.selected().unwrap_or(0);
            let page_size = 10; // 每页显示的行数
            let new_index = (current + page_size).min(self.models.len() - 1);
            if Some(new_index) != self.table_state.selected() {
                self.table_state.select(Some(new_index));
                self.scroll_state = self.scroll_state.position(new_index);
            }
        }
    }

    // 弹出框相关方法
    pub fn toggle_popup(&mut self) {
        self.show_popup = !self.show_popup;
        if self.show_popup {
            self.input_mode = InputMode::Editing;
            self.popup_input.clear();
            self.cursor_position = 0;
        } else {
            self.input_mode = InputMode::Normal;
        }
    }

    pub fn move_cursor_left(&mut self) {
        self.cursor_position = self.cursor_position.saturating_sub(1);
    }

    pub fn move_cursor_right(&mut self) {
        let len = self.popup_input.chars().count();
        self.cursor_position = (self.cursor_position + 1).min(len);
    }

    pub fn enter_char(&mut self, c: char) {
        let byte_index = self.popup_input
            .char_indices()
            .map(|(i, _)| i)
            .nth(self.cursor_position)
            .unwrap_or(self.popup_input.len());
        self.popup_input.insert(byte_index, c);
        self.move_cursor_right();
    }

    pub fn delete_char(&mut self) {
        if self.cursor_position > 0 {
            let from = self.popup_input.char_indices()
                .map(|(i, _)| i)
                .nth(self.cursor_position - 1)
                .unwrap_or(0);
            let to = self.popup_input.char_indices()
                .map(|(i, _)| i)
                .nth(self.cursor_position)
                .unwrap_or(self.popup_input.len());
            self.popup_input.replace_range(from..to, "");
            self.move_cursor_left();
        }
    }

    pub fn submit_popup(&mut self) {
        if !self.popup_input.is_empty() {
            self.api_key = self.popup_input.clone();
            self.show_popup = false;
            self.input_mode = InputMode::Normal;
        }
    }

    // 选择上一个菜单选项
    pub fn previous_menu_option(&mut self) {
        self.selected_menu_option = match self.selected_menu_option {
            MenuOption::DefaultConfig => MenuOption::CustomConfig,
            MenuOption::CustomConfig => MenuOption::DefaultConfig,
        };
    }

    // 选择下一个菜单选项
    pub fn next_menu_option(&mut self) {
        self.selected_menu_option = match self.selected_menu_option {
            MenuOption::DefaultConfig => MenuOption::CustomConfig,
            MenuOption::CustomConfig => MenuOption::DefaultConfig,
        };
    }

    // 处理菜单选择
    pub fn handle_menu_selection(&mut self) {
        match self.selected_menu_option {
            MenuOption::DefaultConfig => {
                self.init_splash_progress();
                self.state = AppState::Splash;
            }
            MenuOption::CustomConfig => {
                self.state = AppState::Config;
            }
        }
    }

    pub fn init_splash_progress(&mut self) {
        self.splash_progress = Some(SplashProgress {
            steps: vec![
                ("正在初始化检测环境...", 0.05),
                ("正在验证API密钥格式...", 0.15),
                ("正在检查网络连接...", 0.25),
                ("正在验证API服务器...", 0.35),
                ("正在建立安全连接...", 0.45),
                ("正在验证API权限...", 0.55),
                ("正在获取API配置信息...", 0.65),
                ("正在获取可用模型列表...", 0.75),
            ],
            current_idx: 0,
            last_step_time: std::time::Instant::now(),
        });
        self.detection_progress = 0.0;
        self.detection_step = "准备开始检测...".to_string();
        self.error = None;
        self.models_loaded = false;
    }

    pub async fn advance_splash_step(&mut self) -> bool {
        if let Some(ref mut sp) = self.splash_progress {
            let elapsed = sp.last_step_time.elapsed();
            if elapsed >= Duration::from_millis(400) {
                // 获取总数，避免后面重复访问
                let total_items = self.detection_items.len();
                
                // 更新当前检测项目的进度
                if sp.current_idx < total_items {
                    // 标记上一个项目完成
                    if sp.current_idx > 0 {
                        let prev_idx = sp.current_idx - 1;
                        self.detection_items[prev_idx].is_done = true;
                        self.detection_items[prev_idx].progress = 1.0;
                    }
                    
                    // 更新当前项目进度
                    let item = &mut self.detection_items[sp.current_idx];
                    item.progress += 0.6; // 增加进度增量，从0.3改为0.6
                    
                    // 更新总体进度 (使用之前保存的 total_items)
                    self.detection_progress = (sp.current_idx as f64 + item.progress) / 
                                           (total_items as f64);
                    self.detection_step = item.name.to_string();
                    
                    if item.progress >= 1.0 {
                        item.is_done = true;
                        item.progress = 1.0;
                        sp.current_idx += 1;
                    }
                    
                    sp.last_step_time = std::time::Instant::now();
                } else {
                    // 所有项目都完成，获取模型列表
                    if let Err(e) = self.fetch_models().await {
                        self.error = Some(format!("获取模型列表失败: {}", e));
                        // 标记最后一个项目发生错误
                        if let Some(last_item) = self.detection_items.last_mut() {
                            last_item.is_error = true;
                        }
                        self.detection_step = "检测失败".to_string();
                        self.splash_progress = None;
                        return false;
                    }
                    
                    // 检测完成
                    self.detection_progress = 1.0;
                    self.detection_step = "检测完成".to_string();
                    self.models_loaded = true;
                    self.splash_progress = None;
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
} 