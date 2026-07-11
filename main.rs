#![cfg_attr(windows, windows_subsystem = "windows")]

use eframe::egui;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    fs,
    io::{Read, Write},
    net::TcpStream,
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use walkdir::WalkDir;

const DEFAULT_AGENT_MODEL: &str = "qwen2.5-coder:7b";
const LOCAL_GPT_ROUTER_MODEL: &str = "gpt-oss:20b";
const QWEN_CODER_MODEL: &str = "qwen2.5-coder:1.5b";
const DEEPSEEK_CODER_MODEL: &str = "deepseek-coder:1.3b";
const STARCODER_MODEL: &str = "starcoder2:7b";
const DEEPSEEK_LARGE_CODER_MODEL: &str = "deepseek-coder:6.7b";
const OPENCLAW_AGENT_MODEL: &str = "openclaw-agent:latest";
const ALL_DOWNLOADABLE_AGENT_MODELS: &[&str] = &[
    LOCAL_GPT_ROUTER_MODEL,
    DEFAULT_AGENT_MODEL,
    QWEN_CODER_MODEL,
    DEEPSEEK_CODER_MODEL,
    STARCODER_MODEL,
    DEEPSEEK_LARGE_CODER_MODEL,
];

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ChatMsg {
    who: String,
    text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProjectEntry {
    id: String,
    name: String,
    path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ChatSession {
    id: String,
    title: String,
    project_id: String,
    project_dir: String,
    messages: Vec<ChatMsg>,
    updated_at: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SavedState {
    projects: Vec<ProjectEntry>,
    sessions: Vec<ChatSession>,
    active_project_id: String,
    active_session_id: String,
    #[serde(default = "default_agent_graph")]
    agent_graph: AgentGraph,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AgentGraph {
    blocks: Vec<AgentBlock>,
    connections: Vec<AgentConnection>,
    next_id: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AgentBlock {
    id: u64,
    title: String,
    kind: AgentBlockKind,
    model: String,
    prompt: String,
    task: String,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum AgentBlockKind {
    Coordinator,
    Manager,
    CodingAgent,
    PlanningAgent,
    ReviewAgent,
    Task,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AgentConnection {
    from: u64,
    to: u64,
    label: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AppView {
    Chat,
    Agents,
    Log,
}

#[derive(Clone, Debug)]
struct AgentStep {
    name: &'static str,
    role: &'static str,
}

#[derive(Clone, Debug)]
struct AgentResult {
    final_answer: String,
    log_lines: Vec<String>,
}

#[derive(Clone, Debug)]
struct AgentRunConfig {
    project_dir: String,
    memory_dir: String,
    model: String,
    ollama_path: String,
    terminal_cmd: String,
    user_request: String,
    last_file_path: Option<String>,
    show_progress: bool,
    auto_apply_actions: bool,
    run_tests_after_apply: bool,
    context_limit: usize,
    max_parallel_agents: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ActionEnvelope {
    summary: Option<String>,
    actions: Vec<FileAction>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum FileAction {
    WriteFile {
        path: String,
        content: String,
    },
    CopyFile {
        source: String,
        path: String,
    },
    PackagePythonExe {
        path: String,
    },
    ReplaceText {
        path: String,
        find: String,
        replace: String,
    },
    AppendFile {
        path: String,
        content: String,
    },
}

#[derive(Debug)]
struct ActionReport {
    log_lines: Vec<String>,
    changed_files: Vec<String>,
    had_error: bool,
}

#[derive(Debug)]
struct CoderCandidateResult {
    name: String,
    model: String,
    raw: String,
    envelope: Option<ActionEnvelope>,
    parse_error: Option<String>,
    validation_errors: Vec<String>,
}

#[derive(Debug)]
enum LocalFileTask {
    CreateFile { path: String, content: String },
    NeedFileName,
}

#[derive(Debug, Deserialize)]
struct CoordinatorDecision {
    reply_now: bool,
    needs_code: bool,
    summary: String,
    direct_reply: String,
    #[serde(default = "default_recommended_agent_count")]
    recommended_agent_count: usize,
    #[serde(default)]
    use_large_single_coder: bool,
}

fn default_recommended_agent_count() -> usize {
    1
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaTagModel>,
}

#[derive(Deserialize)]
struct OllamaTagModel {
    name: String,
}

pub struct LocalAiApp {
    project_dir: String,
    data_root: PathBuf,
    terminal_cmd: String,
    context_limit: String,
    ollama_path: String,
    model: String,
    input: String,
    projects: Vec<ProjectEntry>,
    sessions: Vec<ChatSession>,
    active_project_id: String,
    active_session_id: String,
    new_project_name: String,
    new_project_path: String,
    state_path: PathBuf,
    agent_graph: AgentGraph,
    active_view: AppView,
    selected_block_id: Option<u64>,
    connecting_from: Option<u64>,
    show_python_code: bool,
    python_code: String,
    graph_status: String,
    log: Vec<String>,
    dropped_files: Vec<String>,
    busy: bool,
    rx: Option<Receiver<(u64, AgentResult)>>,
    active_run_id: u64,
    show_agent_progress: bool,
    auto_apply_actions: bool,
    run_tests_after_apply: bool,
    max_parallel_agents: String,
}

impl Default for LocalAiApp {
    fn default() -> Self {
        let dist = app_base_dir();
        let data_root = distribution_data_root(&dist);
        ensure_distribution_data_dirs(&data_root);
        let state_path = state_file_path(&dist);
        let ollama = find_ollama(&dist)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "Not found".to_string());
        let saved_state = load_saved_state(&state_path, &dist);
        let active_project_dir = saved_state
            .projects
            .iter()
            .find(|p| p.id == saved_state.active_project_id)
            .map(|p| p.path.clone())
            .unwrap_or_else(|| dist.display().to_string());
        let agent_graph = saved_state.agent_graph.clone();
        let python_code = graph_to_python_code(&agent_graph);

        let mut app = Self {
            project_dir: active_project_dir.clone(),
            data_root,
            terminal_cmd: "cargo check".to_string(),
            context_limit: "50000".to_string(),
            ollama_path: ollama,
            model: DEFAULT_AGENT_MODEL.to_string(),
            input: String::new(),
            projects: saved_state.projects,
            sessions: saved_state.sessions,
            active_project_id: saved_state.active_project_id,
            active_session_id: saved_state.active_session_id,
            new_project_name: String::new(),
            new_project_path: String::new(),
            state_path,
            agent_graph,
            active_view: AppView::Chat,
            selected_block_id: None,
            connecting_from: None,
            show_python_code: false,
            python_code,
            graph_status: "Agent graph ready.".to_string(),
            log: vec![],
            dropped_files: vec![],
            busy: false,
            rx: None,
            active_run_id: 0,
            show_agent_progress: true,
            auto_apply_actions: true,
            run_tests_after_apply: false,
            max_parallel_agents: "6".to_string(),
        };
        app.log(format!("Ollama: {}", app.ollama_path));
        app
    }
}

fn apply_light_beige_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.visuals = egui::Visuals::light();
    style.visuals.window_fill = egui::Color32::from_rgb(255, 255, 255);
    style.visuals.panel_fill = egui::Color32::from_rgb(247, 247, 248);
    style.visuals.extreme_bg_color = egui::Color32::from_rgb(255, 255, 255);
    style.visuals.faint_bg_color = egui::Color32::from_rgb(242, 242, 244);
    style.visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(247, 247, 248);
    style.visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(255, 255, 255);
    style.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(242, 242, 244);
    style.visuals.widgets.active.bg_fill = egui::Color32::from_rgb(232, 232, 235);
    style.visuals.selection.bg_fill = egui::Color32::from_rgb(16, 163, 127);
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    ctx.set_style(style);
}

impl LocalAiApp {
    fn log(&mut self, msg: impl Into<String>) {
        let line = format!("[{}] {}", now_hhmmss(), msg.into());
        append_text_file(
            &self
                .data_root
                .join("memory")
                .join("agent_traffic")
                .join("traffic.log"),
            &format!("{}\n", line),
        );
        self.log.push(line);
    }

    fn save_state(&self) {
        let state = SavedState {
            projects: self.projects.clone(),
            sessions: self.sessions.clone(),
            active_project_id: self.active_project_id.clone(),
            active_session_id: self.active_session_id.clone(),
            agent_graph: self.agent_graph.clone(),
        };

        if let Some(parent) = self.state_path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        if let Ok(text) = serde_json::to_string_pretty(&state) {
            let _ = fs::write(&self.state_path, text);
        }
    }

    fn active_session_index(&self) -> Option<usize> {
        self.sessions
            .iter()
            .position(|session| session.id == self.active_session_id)
    }

    fn active_session(&self) -> Option<&ChatSession> {
        self.active_session_index()
            .and_then(|idx| self.sessions.get(idx))
    }

    fn ensure_active_session_index(&mut self) -> usize {
        if let Some(idx) = self.active_session_index() {
            return idx;
        }

        self.create_session_for_active_project();
        self.active_session_index().unwrap_or(0)
    }

    fn set_active_session(&mut self, session_id: &str) {
        if self.busy {
            return;
        }

        if let Some(session) = self.sessions.iter().find(|s| s.id == session_id) {
            self.active_session_id = session.id.clone();
            self.active_project_id = session.project_id.clone();
            self.project_dir = session.project_dir.clone();
            self.new_project_path.clear();
            self.save_state();
        }
    }

    fn set_active_project(&mut self, project_id: &str) {
        if self.busy {
            return;
        }

        if let Some(project) = self.projects.iter().find(|p| p.id == project_id).cloned() {
            self.active_project_id = project.id.clone();
            self.project_dir = project.path.clone();
            self.new_project_path.clear();

            if let Some(session) = self
                .sessions
                .iter()
                .find(|session| session.project_id == project.id)
                .cloned()
            {
                self.active_session_id = session.id;
            } else {
                self.create_session_for_active_project();
            }

            self.save_state();
        }
    }

    fn create_session_for_active_project(&mut self) {
        let project = self
            .projects
            .iter()
            .find(|project| project.id == self.active_project_id)
            .cloned()
            .or_else(|| self.projects.first().cloned());

        let Some(project) = project else {
            return;
        };

        let id = format!("chat-{}", now_millis());
        let session = ChatSession {
            id: id.clone(),
            title: "New task".to_string(),
            project_id: project.id.clone(),
            project_dir: project.path.clone(),
            messages: vec![welcome_message()],
            updated_at: now_secs(),
        };

        self.active_project_id = project.id;
        self.active_session_id = id;
        self.project_dir = project.path.clone();
        self.new_project_path.clear();
        self.sessions.insert(0, session);
        self.save_state();
    }

    fn create_project_from_fields(&mut self) {
        if self.busy {
            return;
        }

        let requested_name = self.new_project_name.trim().to_string();
        let requested_path = self.new_project_path.trim().trim_matches('"').to_string();

        if requested_name.is_empty() && requested_path.is_empty() {
        self.log("Project name or project folder is missing.");
            return;
        }

        let path_buf = if requested_path.is_empty() {
            let folder_name = sanitize_project_folder_name(&requested_name);
            match std::env::current_dir() {
                Ok(dir) => dir.join(folder_name),
                Err(_) => PathBuf::from(folder_name),
            }
        } else {
            PathBuf::from(&requested_path)
        };

        if path_buf.exists() {
            if !path_buf.is_dir() {
                self.log(format!(
                    "Project cannot be added because the path is a file: {}",
                    path_buf.display()
                ));
                return;
            }
        } else if let Err(e) = fs::create_dir_all(&path_buf) {
            self.log(format!(
                "Project folder could not be created: {}: {}",
                path_buf.display(),
                e
            ));
            return;
        }

        let name = if requested_name.is_empty() {
            path_buf
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Project")
                .to_string()
        } else {
            requested_name
        };

        let canonical = fs::canonicalize(&path_buf).unwrap_or(path_buf);
        let canonical_path = canonical.display().to_string();
        let id = format!("project-{}", now_millis());

        self.projects.push(ProjectEntry {
            id: id.clone(),
            name: name.clone(),
            path: canonical_path.clone(),
        });

        self.active_project_id = id;
        self.project_dir = canonical_path.clone();
        self.new_project_name.clear();
        self.new_project_path.clear();
        self.create_session_for_active_project();
        self.log(format!("Project added: {} ({})", name, canonical_path));
    }

    fn sync_python_from_graph(&mut self) {
        self.python_code = graph_to_python_code(&self.agent_graph);
        self.save_state();
    }

    fn import_graph_from_python(&mut self) {
        match graph_from_python_code(&self.python_code) {
            Ok(graph) => {
                self.agent_graph = graph;
                self.selected_block_id = None;
                self.connecting_from = None;
                self.graph_status = "Python code was imported into blocks.".to_string();
                self.save_state();
            }
            Err(e) => {
                self.graph_status = format!("Code could not be imported: {}", e);
            }
        }
    }

    fn add_agent_block(&mut self, kind: AgentBlockKind) {
        let id = self.agent_graph.next_id.max(1);
        self.agent_graph.next_id = id + 1;

        let index = self.agent_graph.blocks.len() as f32;
        let (title, model, prompt, task, w, h) = default_block_fields(&kind, id);

        self.agent_graph.blocks.push(AgentBlock {
            id,
            title,
            kind,
            model,
            prompt,
            task,
            x: 90.0 + (index % 4.0) * 220.0,
            y: 110.0 + (index / 4.0).floor() * 170.0,
            w,
            h,
        });

        self.selected_block_id = Some(id);
        self.sync_python_from_graph();
    }

    fn connect_or_select_block(&mut self, block_id: u64) {
        if let Some(from) = self.connecting_from {
            if from != block_id
                && !self
                    .agent_graph
                    .connections
                    .iter()
                    .any(|connection| connection.from == from && connection.to == block_id)
            {
                self.agent_graph.connections.push(AgentConnection {
                    from,
                    to: block_id,
                    label: "delegiert".to_string(),
                });
                self.sync_python_from_graph();
            }
            self.connecting_from = None;
        }

        self.selected_block_id = Some(block_id);
    }

    fn remove_selected_block(&mut self) {
        let Some(block_id) = self.selected_block_id else {
            return;
        };

        self.agent_graph.blocks.retain(|block| block.id != block_id);
        self.agent_graph
            .connections
            .retain(|connection| connection.from != block_id && connection.to != block_id);
        self.selected_block_id = None;
        self.connecting_from = None;
        self.sync_python_from_graph();
    }

    fn show_agents_ui(&mut self, ui: &mut egui::Ui, window_width: f32, window_height: f32) {
        ui.horizontal(|ui| {
            if ui.button("+ Verwaltungsagent").clicked() {
                self.add_agent_block(AgentBlockKind::Manager);
            }
            if ui.button("+ Coding-Agent").clicked() {
                self.add_agent_block(AgentBlockKind::CodingAgent);
            }
            if ui.button("+ Planungs-Agent").clicked() {
                self.add_agent_block(AgentBlockKind::PlanningAgent);
            }
            if ui.button("+ Task block").clicked() {
                self.add_agent_block(AgentBlockKind::Task);
            }
            if ui.button("+ Tool").clicked() {
                self.add_agent_block(AgentBlockKind::Tool);
            }
            if ui.button("Python aktualisieren").clicked() {
                self.sync_python_from_graph();
            }
        });

        ui.label(&self.graph_status);
        ui.separator();

        if self.show_python_code {
            ui.horizontal(|ui| {
                if ui.button("Generate from blocks").clicked() {
                    self.sync_python_from_graph();
                    self.graph_status = "Python code was generated from blocks.".to_string();
                }
                if ui.button("Import code").clicked() {
                    self.import_graph_from_python();
                }
            });

            ui.add_sized(
                [ui.available_width(), (window_height * 0.72).max(360.0)],
                egui::TextEdit::multiline(&mut self.python_code)
                    .font(egui::TextStyle::Monospace)
                    .desired_rows(28),
            );
            return;
        }

        ui.horizontal(|ui| {
            let inspector_width = (window_width * 0.25).clamp(260.0, 390.0);
            let canvas_width = (ui.available_width() - inspector_width - 12.0).max(420.0);
            let canvas_height = (window_height * 0.74).max(420.0);

            self.show_agent_canvas(ui, egui::vec2(canvas_width, canvas_height));

            ui.separator();
            ui.vertical(|ui| {
                ui.set_width(inspector_width);
                self.show_block_inspector(ui);
            });
        });
    }

    fn show_agent_canvas(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        let (canvas_rect, canvas_response) = ui.allocate_exact_size(size, egui::Sense::click());
        let painter = ui.painter_at(canvas_rect);

        painter.rect_filled(canvas_rect, 4.0, egui::Color32::from_rgb(24, 25, 29));
        painter.rect_stroke(
            canvas_rect,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(70)),
        );

        for connection in &self.agent_graph.connections {
            let Some(from) = self
                .agent_graph
                .blocks
                .iter()
                .find(|block| block.id == connection.from)
            else {
                continue;
            };
            let Some(to) = self
                .agent_graph
                .blocks
                .iter()
                .find(|block| block.id == connection.to)
            else {
                continue;
            };

            let from_pos = canvas_rect.min + egui::vec2(from.x + from.w, from.y + from.h * 0.5);
            let to_pos = canvas_rect.min + egui::vec2(to.x, to.y + to.h * 0.5);
            let mid_x = (from_pos.x + to_pos.x) * 0.5;
            let points = vec![
                from_pos,
                egui::pos2(mid_x, from_pos.y),
                egui::pos2(mid_x, to_pos.y),
                to_pos,
            ];
            painter.add(egui::Shape::line(
                points,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 170, 255)),
            ));

            let label_pos = egui::pos2(mid_x + 4.0, (from_pos.y + to_pos.y) * 0.5);
            painter.text(
                label_pos,
                egui::Align2::LEFT_CENTER,
                &connection.label,
                egui::TextStyle::Small.resolve(ui.style()),
                egui::Color32::from_rgb(180, 205, 255),
            );
        }

        if let Some(from_id) = self.connecting_from {
            if let Some(from) = self
                .agent_graph
                .blocks
                .iter()
                .find(|block| block.id == from_id)
            {
                if let Some(pointer) = ui.ctx().pointer_hover_pos() {
                    let from_pos =
                        canvas_rect.min + egui::vec2(from.x + from.w, from.y + from.h * 0.5);
                    painter.line_segment(
                        [from_pos, pointer],
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 210, 120)),
                    );
                }
            }
        }

        let mut graph_changed = false;
        let mut clicked_block = None;
        let mut connector_started = None;
        let mut released_over_block = None;

        for block in &mut self.agent_graph.blocks {
            let block_rect = egui::Rect::from_min_size(
                canvas_rect.min + egui::vec2(block.x, block.y),
                egui::vec2(block.w, block.h),
            );
            let response = ui.allocate_rect(block_rect, egui::Sense::click_and_drag());

            if response.dragged() {
                let delta = response.drag_delta();
                block.x = (block.x + delta.x).clamp(0.0, (canvas_rect.width() - block.w).max(0.0));
                block.y = (block.y + delta.y).clamp(0.0, (canvas_rect.height() - block.h).max(0.0));
                graph_changed = true;
            }

            if response.clicked() {
                clicked_block = Some(block.id);
            }

            if response.hovered()
                && self.connecting_from.is_some()
                && ui.input(|input| input.pointer.any_released())
            {
                released_over_block = Some(block.id);
            }

            let selected = self.selected_block_id == Some(block.id);
            let fill = block_color(&block.kind, selected);
            painter.rect_filled(block_rect, 8.0, fill);
            painter.rect_stroke(
                block_rect,
                8.0,
                egui::Stroke::new(
                    if selected { 2.0 } else { 1.0 },
                    if selected {
                        egui::Color32::WHITE
                    } else {
                        egui::Color32::from_gray(95)
                    },
                ),
            );

            painter.text(
                block_rect.left_top() + egui::vec2(12.0, 10.0),
                egui::Align2::LEFT_TOP,
                &block.title,
                egui::TextStyle::Button.resolve(ui.style()),
                egui::Color32::WHITE,
            );
            painter.text(
                block_rect.left_top() + egui::vec2(12.0, 34.0),
                egui::Align2::LEFT_TOP,
                block.kind.label(),
                egui::TextStyle::Small.resolve(ui.style()),
                egui::Color32::from_gray(225),
            );
            painter.text(
                block_rect.left_bottom() + egui::vec2(12.0, -22.0),
                egui::Align2::LEFT_BOTTOM,
                &block.model,
                egui::TextStyle::Small.resolve(ui.style()),
                egui::Color32::from_gray(215),
            );

            let handle_center = egui::pos2(block_rect.right(), block_rect.center().y);
            let handle_rect = egui::Rect::from_center_size(handle_center, egui::vec2(18.0, 18.0));
            let handle_response = ui.allocate_rect(handle_rect, egui::Sense::click_and_drag());
            painter.circle_filled(handle_center, 6.0, egui::Color32::from_rgb(255, 210, 120));

            if handle_response.clicked() || handle_response.drag_started() {
                connector_started = Some(block.id);
            }
        }

        if let Some(block_id) = connector_started {
            self.selected_block_id = Some(block_id);
            self.connecting_from = Some(block_id);
            self.graph_status = "Drag or click the target block now.".to_string();
        } else if let Some(block_id) = released_over_block.or(clicked_block) {
            self.connect_or_select_block(block_id);
        } else if canvas_response.clicked() {
            self.selected_block_id = None;
            self.connecting_from = None;
        }

        if graph_changed {
            self.sync_python_from_graph();
        }
    }

    fn show_block_inspector(&mut self, ui: &mut egui::Ui) {
        ui.heading("Block");

        let Some(block_id) = self.selected_block_id else {
            ui.label("Select a block.");
            return;
        };

        let Some(index) = self
            .agent_graph
            .blocks
            .iter()
            .position(|block| block.id == block_id)
        else {
            self.selected_block_id = None;
            return;
        };

        let mut graph_changed = false;
        {
            let block = &mut self.agent_graph.blocks[index];
            ui.label(format!("ID: {}", block.id));
            ui.label("Titel");
            graph_changed |= ui.text_edit_singleline(&mut block.title).changed();

            ui.label("Typ");
            egui::ComboBox::from_id_source("block_kind_combo")
                .selected_text(block.kind.label())
                .show_ui(ui, |ui| {
                    for kind in AgentBlockKind::all() {
                        graph_changed |= ui
                            .selectable_value(&mut block.kind, kind.clone(), kind.label())
                            .changed();
                    }
                });

            ui.label("Model");
            graph_changed |= ui.text_edit_singleline(&mut block.model).changed();

            ui.label("Task");
            graph_changed |= ui.text_edit_multiline(&mut block.task).changed();

            ui.label("System/Prompt");
            graph_changed |= ui.text_edit_multiline(&mut block.prompt).changed();
        }

        ui.separator();
        if ui.button("Linie von diesem Block starten").clicked() {
            self.connecting_from = Some(block_id);
            self.graph_status = "Select the target block for the connection.".to_string();
        }
        if ui.button("Delete block").clicked() {
            self.remove_selected_block();
            return;
        }

        ui.separator();
        ui.label("Ausgehende Verbindungen");
        let mut remove_connection = None;
        let block_titles: Vec<(u64, String)> = self
            .agent_graph
            .blocks
            .iter()
            .map(|block| (block.id, block.title.clone()))
            .collect();
        for (idx, connection) in self.agent_graph.connections.iter_mut().enumerate() {
            if connection.from == block_id {
                let target_title = block_titles
                    .iter()
                    .find(|(id, _)| *id == connection.to)
                    .map(|(_, title)| title.clone())
                    .unwrap_or_else(|| format!("Block {}", connection.to));
                ui.horizontal(|ui| {
                    ui.label(format!("→ {}", target_title));
                    graph_changed |= ui.text_edit_singleline(&mut connection.label).changed();
                    if ui.button("x").clicked() {
                        remove_connection = Some(idx);
                    }
                });
            }
        }

        if let Some(idx) = remove_connection {
            self.agent_graph.connections.remove(idx);
            graph_changed = true;
        }

        if graph_changed {
            self.sync_python_from_graph();
        }
    }

    fn send_user_message(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.busy {
            return;
        }

        self.input.clear();
        let session_idx = self.ensure_active_session_index();

        self.sessions[session_idx].messages.push(ChatMsg {
            who: "You".to_string(),
            text: text.clone(),
        });

        if self.sessions[session_idx].title == "New task" || self.sessions[session_idx].title == "Neue Task" {
            self.sessions[session_idx].title = title_from_message(&text);
        }

        self.sessions[session_idx].updated_at = now_secs();

        let project_dir = self.sessions[session_idx].project_dir.clone();
        let memory_dir = self.data_root.join("memory").display().to_string();
        let model = self.model.clone();
        let ollama_path = self.ollama_path.clone();
        let terminal_cmd = self.terminal_cmd.clone();
        let last_file_path = last_file_from_messages(&self.sessions[session_idx].messages);
        let show_agent_progress = self.show_agent_progress;
        let auto_apply_actions = self.auto_apply_actions;
        let run_tests_after_apply = self.run_tests_after_apply;
        let context_limit = parse_usize_field(&self.context_limit, 50_000, 4_000, 200_000);
        let max_parallel_agents = parse_usize_field(&self.max_parallel_agents, 2, 1, 6);

        self.busy = true;
        self.active_run_id = self.active_run_id.wrapping_add(1);
        let run_id = self.active_run_id;
        self.log(format!(
            "TRAFFIC User -> Coordinator | Request:\n{}",
            truncate_text(&text, 2_000)
        ));
        self.log(format!("Coordinator started. Run id: {}", run_id));
        self.save_state();

        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);

        thread::spawn(move || {
            let result = run_agent_chain(AgentRunConfig {
                project_dir,
                memory_dir,
                model,
                ollama_path,
                terminal_cmd,
                user_request: text,
                last_file_path,
                show_progress: show_agent_progress,
                auto_apply_actions,
                run_tests_after_apply,
                context_limit,
                max_parallel_agents,
            });
            let _ = tx.send((run_id, result));
        });
    }

    fn stop_current_task(&mut self) {
        if !self.busy {
            return;
        }

        self.active_run_id = self.active_run_id.wrapping_add(1);
        self.busy = false;
        self.rx = None;
        self.log("Stop requested. The current background result will be ignored.");
    }

    fn poll_agent_result(&mut self) {
        if let Some(rx) = &self.rx {
            if let Ok((run_id, result)) = rx.try_recv() {
                if run_id != self.active_run_id {
                    return;
                }
                self.busy = false;
                self.rx = None;

                let session_idx = self.ensure_active_session_index();
                let assistant_answer = clean_visible_answer(&result.final_answer);
                self.sessions[session_idx].messages.push(ChatMsg {
                    who: "Assistant".to_string(),
                    text: assistant_answer,
                });
                self.sessions[session_idx].updated_at = now_secs();
                let completed_session = self.sessions[session_idx].clone();

                self.persist_run_dataset(&completed_session, &result);
                self.update_memory_after_run(&completed_session, &result);
                self.start_short_training();

                for line in result.log_lines {
                    self.log(line);
                }

                self.log("Short training data and memory were updated for the next answer.");
                self.save_state();
            }
        }
    }

    fn persist_run_dataset(&self, session: &ChatSession, result: &AgentResult) {
        let had_error = result
            .log_lines
            .iter()
            .any(|line| is_actual_error_log_line(line));

        let target_dir = if had_error {
            self.data_root.join("training").join("errors")
        } else {
            self.data_root.join("training").join("success")
        };
        let _ = fs::create_dir_all(&target_dir);

        let record = json!({
            "timestamp": now_secs(),
            "active_project_id": self.active_project_id,
            "active_session_id": self.active_session_id,
            "project_dir": self.project_dir,
            "had_error": had_error,
            "collector": "training_data_collector",
            "chat_messages": session.messages,
            "final_answer": result.final_answer,
            "log_lines": result.log_lines,
        });

        let fine_tuning_messages: Vec<_> = session
            .messages
            .iter()
            .filter_map(|message| {
                let role = match message.who.as_str() {
                    "You" => "user",
                    "Assistant" => "assistant",
                    _ => return None,
                };
                Some(json!({"role": role, "content": message.text}))
            })
            .collect();

        let fine_tuning_record = json!({
            "messages": fine_tuning_messages,
            "metadata": {
                "timestamp": now_secs(),
                "project_id": self.active_project_id,
                "session_id": self.active_session_id,
                "had_error": had_error
            }
        });

        append_text_file(
            &target_dir.join("runs.jsonl"),
            &format!("{}\n", record.to_string()),
        );
        append_text_file(
            &self.data_root.join("training").join("raw").join("all_runs.jsonl"),
            &format!("{}\n", record.to_string()),
        );
        if !had_error {
            append_text_file(
                &self
                    .data_root
                    .join("training")
                    .join("fine_tuning")
                    .join("chat_messages.jsonl"),
                &format!("{}\n", fine_tuning_record),
            );
        }
    }

    fn update_memory_after_run(&self, session: &ChatSession, result: &AgentResult) {
        let user_message = session
            .messages
            .iter()
            .rev()
            .find(|message| message.who == "You")
            .map(|message| message.text.as_str())
            .unwrap_or("");
        let assistant_message = session
            .messages
            .iter()
            .rev()
            .find(|message| message.who == "Assistant")
            .map(|message| message.text.as_str())
            .unwrap_or(result.final_answer.as_str());
        let error_lines: Vec<_> = result
            .log_lines
            .iter()
            .filter(|line| is_actual_error_log_line(line))
            .map(|line| truncate_text(line, 500))
            .collect();

        let memory_record = json!({
            "timestamp": now_secs(),
            "project_id": self.active_project_id,
            "session_id": self.active_session_id,
            "user": user_message,
            "assistant": assistant_message,
            "errors": error_lines,
        });
        append_text_file(
            &self
                .data_root
                .join("memory")
                .join("conversations")
                .join("history.jsonl"),
            &format!("{}\n", memory_record),
        );

        let current_memory = format!(
            "# Current project memory\n\nUpdated: {}\nSession: {}\n\n## Last user request\n{}\n\n## Last answer\n{}\n\n## Errors to avoid\n{}\n",
            now_secs(),
            self.active_session_id,
            truncate_text(user_message, 4_000),
            truncate_text(assistant_message, 4_000),
            if error_lines.is_empty() {
                "None".to_string()
            } else {
                error_lines.join("\n")
            }
        );
        let memory_path = self
            .data_root
            .join("memory")
            .join("projects")
            .join("current.md");
        if let Some(parent) = memory_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(memory_path, current_memory);
    }

    fn start_short_training(&mut self) {
        let dataset = self
            .data_root
            .join("training")
            .join("fine_tuning")
            .join("chat_messages.jsonl");
        if !dataset.exists() {
            self.log("Short training skipped: no successful training example exists yet.");
            return;
        }

        let script = self
            .data_root
            .join("training_tools")
            .join("online_train.py");
        let script = if script.exists() {
            script
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("training_tools")
                .join("online_train.py")
        };
        if !script.exists() {
            self.log(format!(
                "Short training could not start: trainer missing at {}",
                script.display()
            ));
            return;
        }

        let output_dir = self.data_root.join("training").join("online_adapter");
        let status_file = self
            .data_root
            .join("training")
            .join("short_training_status.log");
        self.log("Short LoRA training started in the background (1 step).");

        thread::spawn(move || {
            let started = now_secs();
            append_text_file(
                &status_file,
                &format!("[{}] starting one-step LoRA training\n", started),
            );
            let result = hidden_command("python")
                .arg(&script)
                .arg("--dataset")
                .arg(&dataset)
                .arg("--output")
                .arg(&output_dir)
                .arg("--steps")
                .arg("1")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();

            let message = match result {
                Ok(status) if status.success() => "training completed".to_string(),
                Ok(status) => format!("training failed with exit code {:?}", status.code()),
                Err(error) => format!("training could not start: {}", error),
            };
            append_text_file(
                &status_file,
                &format!("[{}] {}\n", now_secs(), message),
            );
        });
    }
}

impl eframe::App for LocalAiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        apply_light_beige_theme(ctx);
        self.poll_agent_result();

        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        for f in dropped {
            if let Some(path) = f.path {
                if path.is_dir() {
                    self.new_project_path = path.display().to_string();
                } else {
                    let text = format!("Attached file: {}", path.display());
                    if !self.input.trim().is_empty() {
                        self.input.push('\n');
                    }
                    self.input.push_str(&text);
                }
                self.dropped_files.push(path.display().to_string());
                self.log(format!("File dropped: {}", path.display()));
            }
        }

        let window_rect = ctx.available_rect();
        let window_width = window_rect.width().max(720.0);
        let window_height = window_rect.height().max(520.0);
        let sidebar_width = (window_width * 0.23).clamp(220.0, 360.0);
        let top_height = (window_height * 0.075).clamp(48.0, 72.0);
        let bottom_height = (window_height * 0.13).clamp(76.0, 118.0);
        let chat_started = self
            .active_session()
            .map(|session| {
                session
                    .messages
                    .iter()
                    .any(|msg| msg.who == "You" || msg.who == "Du")
            })
            .unwrap_or(false);
        egui::TopBottomPanel::top("top")
            .resizable(false)
            .exact_height(top_height)
            .show(ctx, |ui| {
                ui.add_space(top_height * 0.16);
                ui.horizontal(|ui| {
                    ui.heading("Local AI");
                    ui.separator();

                    if self.busy {
                        ui.label("Coordinator is working...");
                        if ui.button("Stop").clicked() {
                            self.stop_current_task();
                        }
                    } else {
                        ui.label("Ready");
                    }

                    let context_width = (window_width * 0.10).clamp(72.0, 120.0);
                    let agent_width = (window_width * 0.055).clamp(44.0, 72.0);
                    let spacer =
                        (ui.available_width() - context_width - agent_width - 190.0).max(12.0);
                    ui.add_space(spacer);

                    ui.label("Context");
                    ui.add_sized(
                        [context_width, 24.0],
                        egui::TextEdit::singleline(&mut self.context_limit),
                    );
                    ui.label("Agents");
                    ui.add_sized(
                        [agent_width, 24.0],
                        egui::TextEdit::singleline(&mut self.max_parallel_agents),
                    );
                });
            });

        egui::SidePanel::left("left")
            .resizable(true)
            .default_width(sidebar_width)
            .min_width(sidebar_width * 0.82)
            .max_width(sidebar_width * 1.28)
            .show(ctx, |ui| {
                ui.add_space(window_height * 0.012);

                if ui
                    .add_sized(
                        [ui.available_width(), 34.0],
                        egui::Button::new("+ New task"),
                    )
                    .clicked()
                {
                    self.create_session_for_active_project();
                    self.active_view = AppView::Chat;
                }

                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(self.active_view == AppView::Chat, "Chat")
                        .clicked()
                    {
                        self.active_view = AppView::Chat;
                    }
                    if ui
                        .selectable_label(self.active_view == AppView::Log, "Log")
                        .clicked()
                    {
                        self.active_view = AppView::Log;
                    }
                });

                ui.separator();

                ui.heading("Projects");

                let project_items: Vec<(String, String)> = self
                    .projects
                    .iter()
                    .map(|project| (project.id.clone(), project.name.clone()))
                    .collect();

                egui::ScrollArea::vertical()
                    .max_height(window_height * 0.24)
                    .show(ui, |ui| {
                        for (project_id, name) in project_items {
                            let selected = project_id == self.active_project_id;
                            if ui.selectable_label(selected, name).clicked() {
                                self.set_active_project(&project_id);
                            }
                        }
                });

                ui.add_space(window_height * 0.012);
                ui.collapsing("Add project", |ui| {
                    ui.label("Name");
                    ui.text_edit_singleline(&mut self.new_project_name);
                    ui.label("Folder");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.new_project_path)
                            .hint_text("empty = create a new folder with this name"),
                    );

                    if ui
                        .add_sized(
                            [ui.available_width(), 30.0],
                            egui::Button::new("Add project"),
                        )
                        .clicked()
                    {
                        self.create_project_from_fields();
                    }
                });

                ui.separator();
                ui.heading("Tasks");

                let session_items: Vec<(String, String)> = self
                    .sessions
                    .iter()
                    .filter(|session| session.project_id == self.active_project_id)
                    .map(|session| (session.id.clone(), session.title.clone()))
                    .collect();

                egui::ScrollArea::vertical()
                    .max_height(window_height * 0.30)
                    .show(ui, |ui| {
                        for (session_id, title) in session_items {
                            let selected = session_id == self.active_session_id;
                            if ui.selectable_label(selected, title).clicked() {
                                self.set_active_session(&session_id);
                                self.active_view = AppView::Chat;
                            }
                        }
                    });
            });

        if self.active_view == AppView::Chat && chat_started {
            egui::TopBottomPanel::bottom("input")
                .resizable(false)
                .exact_height(bottom_height)
                .show(ctx, |ui| {
                    ui.add_space(bottom_height * 0.16);

                    ui.horizontal(|ui| {
                        let button_width = 92.0;
                        let gap_width = 10.0;
                        let input_width = (ui.available_width() - button_width - gap_width).max(260.0);
                        let input_height = (bottom_height * 0.50).clamp(42.0, 64.0);
                        let response = ui.add_sized(
                            [input_width, input_height],
                            egui::TextEdit::multiline(&mut self.input)
                                .hint_text("Message the coordinator · drop files here")
                                .desired_rows(2),
                        );

                        let send_by_enter =
                            response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

                        ui.add_space(gap_width);
                        ui.vertical(|ui| {
                            ui.set_width(button_width);
                            ui.set_min_height(input_height);
                            let send_clicked = ui
                                .add_sized(
                                    [button_width, 34.0],
                                    egui::Button::new(if self.busy { "..." } else { "Send" }),
                                )
                                .clicked()
                                && !self.busy;

                            if send_by_enter || send_clicked {
                                self.send_user_message();
                            }

                            if self.busy
                                && ui
                                    .add_sized([button_width, 28.0], egui::Button::new("Stop"))
                                    .clicked()
                            {
                                self.stop_current_task();
                            }
                        });
                    });

                    if !self.dropped_files.is_empty() {
                        ui.small(format!(
                            "Last dropped: {}",
                            self.dropped_files
                                .last()
                                .map(String::as_str)
                                .unwrap_or("")
                        ));
                    } else {
                        ui.small("Tip: drop files directly into the chat window.");
                    }
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.active_view == AppView::Log {
                ui.add_space(window_height * 0.012);
                ui.horizontal(|ui| {
                    ui.heading("Agent log");
                    ui.separator();
                    ui.label(format!("{} entries", self.log.len()));
                });
                ui.label("Shows system status, agent roles, model routing, shortened prompts, responses, validation, and manager decisions.");
                ui.add_space(window_height * 0.012);

                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        for line in &self.log {
                            ui.monospace(line);
                            ui.add_space(3.0);
                        }
                    });
                return;
            }

            if self.active_view == AppView::Agents {
                self.show_agents_ui(ui, window_width, window_height);
                return;
            }

            let messages = self
                .active_session()
                .map(|session| session.messages.clone())
                .unwrap_or_default();

            if chat_started {
                ui.add_space(window_height * 0.02);

                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        let content_width = (ui.available_width() * 0.82).max(360.0);
                        for msg in messages {
                            ui.horizontal(|ui| {
                                ui.add_space((ui.available_width() - content_width).max(0.0) * 0.5);
                                ui.vertical(|ui| {
                                    ui.set_width(content_width);
                                    ui.group(|ui| {
                                        ui.label(egui::RichText::new(&msg.who).strong());
                                        render_chat_message(ui, &msg.text);
                                    });
                                });
                            });
                            ui.add_space(window_height * 0.012);
                        }
                    });
            } else {
                let center_space = (ui.available_height() * 0.24).max(48.0);
                ui.add_space(center_space);
                ui.vertical_centered(|ui| {
                    ui.heading("What can I help with?");
                    ui.add_space(window_height * 0.018);

                    let composer_width = (ui.available_width() * 0.68).clamp(420.0, 920.0);
                    let composer_height = (window_height * 0.17).clamp(104.0, 180.0);

                    ui.horizontal(|ui| {
                        let send_width = 92.0;
                        let gap_width = 10.0;
                        let input_width = (composer_width - send_width - gap_width).max(300.0);

                        ui.add_sized(
                            [input_width, composer_height],
                            egui::TextEdit::multiline(&mut self.input)
                                .hint_text("Tell me what to do in the project · drop files here")
                                .desired_rows(4),
                        );

                        ui.add_space(gap_width);
                        ui.vertical(|ui| {
                            ui.set_width(send_width);
                            ui.set_min_height(composer_height);
                            if ui
                                .add_enabled(
                                    !self.busy,
                                    egui::Button::new(if self.busy {
                                        "Working..."
                                    } else {
                                        "Send"
                                    }),
                                )
                                .clicked()
                            {
                                self.send_user_message();
                            }
                        });
                    });
                    ui.small("You can drop files into the chat window; I will attach their path to the message.");
                });
            }
        });

        if self.busy {
            ctx.request_repaint_after(Duration::from_millis(250));
        }
    }
}

fn run_agent_chain(config: AgentRunConfig) -> AgentResult {
    let mut log_lines = vec![];

    if let Some(answer) = direct_chat_answer(&config.user_request) {
        return AgentResult {
            final_answer: answer,
            log_lines,
        };
    }

    if config.show_progress {
        log_lines.push("Coordinator checks the request and project folder...".to_string());
    }

    let project_root = match canonical_project_root(&config.project_dir) {
        Ok(root) => root,
        Err(e) => {
            return AgentResult {
                final_answer: format!("I cannot open the project folder.\n\n{}", e),
                log_lines,
            };
        }
    };

    if let Some(task) =
        parse_local_file_task(&config.user_request, config.last_file_path.as_deref())
    {
        return run_local_file_task(task, &project_root, log_lines);
    }

    let ollama_models_dir = project_ollama_models_dir(&project_root);

    if config.show_progress {
        log_lines.push(format!("Project folder: {}", project_root.display()));
        log_lines.push(format!("Modelordner: {}", ollama_models_dir.display()));
        log_lines.push("Checking Ollama...".to_string());
    }

    if let Err(e) = ensure_ollama_available(&config.ollama_path, &project_root) {
        log_lines.push(format!("Ollama error: {}", e));
        return AgentResult {
            final_answer: format!(
                "Ollama is not reachable or could not be started.\n\n{}",
                e
            ),
            log_lines,
        };
    }

    log_lines.push("Checking models...".to_string());
    if let Err(e) = ensure_all_agent_models(&config.ollama_path, &project_root, &config.model) {
        log_lines.push(format!("Model check failed: {}", e));
        return AgentResult {
            final_answer: format!(
                "I could not prepare the required local models.\n\n{}",
                e
            ),
            log_lines,
        };
    }

    let project_snapshot = read_project_snapshot(
        project_root.to_string_lossy().as_ref(),
        config.context_limit,
    );
    let memory_context = read_memory_context(&config.memory_dir, config.context_limit.min(12_000));
    if !memory_context.trim().is_empty() {
        log_lines.push(format!(
            "Loaded memory context: {} chars.",
            memory_context.chars().count()
        ));
    }

    if config.show_progress {
        log_lines.push("Coordinator decides whether code changes are needed...".to_string());
    }

    let coordinator =
        run_coordinator_decision(&config, &project_root, &project_snapshot, &memory_context, &mut log_lines);
    if config.show_progress {
        log_lines.push(format!(
            "Coordinator: {}",
            truncate_text(&coordinator.summary, 220)
        ));
    }

    if coordinator.reply_now && !coordinator.needs_code {
        return AgentResult {
            final_answer: clean_visible_answer(&coordinator.direct_reply),
            log_lines,
        };
    }

    if config.show_progress {
        log_lines.push("Coder agents are working on candidate solutions...".to_string());
    }

    let effective_agent_count =
        recommended_agent_count(&config, &coordinator).clamp(1, config.max_parallel_agents.max(1));
    log_lines.push(format!(
        "Router decision: {} agent(s), large single coder: {}.",
        effective_agent_count, coordinator.use_large_single_coder
    ));

    let primary_coder_model = if coordinator.use_large_single_coder && effective_agent_count == 1 {
        LOCAL_GPT_ROUTER_MODEL
    } else {
        config.model.as_str()
    };
    let required_agent_models = agent_models_for_count(primary_coder_model, effective_agent_count);
    log_lines.push(format!(
        "Checking agent models: {}",
        required_agent_models.join(", ")
    ));
    if let Err(e) = ensure_required_models(&config.ollama_path, &project_root, &required_agent_models) {
        log_lines.push(format!("Agent model check failed: {}", e));
        return AgentResult {
            final_answer: format!(
                "I could not prepare the required local agent model.\n\n{}",
                e
            ),
            log_lines,
        };
    }

    let coder_prompt = format!(
        r#"You are the coder agent in a local Codex-like system.

You do NOT talk to the user. The coordinator handles user communication.
You only create internal file actions as JSON.

Strict output rules:
- Work like a careful coding assistant: smallest useful change, respect existing patterns, do not invent files.
- Reply only with a single JSON object.
- No Markdown code fences.
- No explanations outside JSON.
- Allowed operations:
  1. {{"op":"replace_text","path":"relative/path","find":"exact old text","replace":"new text"}}
  2. {{"op":"write_file","path":"relative/path","content":"complete file content"}}
  3. {{"op":"append_file","path":"relative/path","content":"text to append"}}
  4. {{"op":"package_python_exe","path":"relative/path.py"}}
  5. {{"op":"copy_file","source":"absolute/or/external/source/path","path":"relative/target/path/inside/project"}}
- Paths must be relative to the project folder.
- No absolute paths.
- No paths with '..'.
- copy_file may use an absolute source path outside the project, but its target path must stay inside the project.
- Use write_file for new files, or only when the full target content is safely clear.
- Use replace_text only with text that occurs exactly once in the snapshot.
- For general coding tasks, write the required source files with write_file, append_file, or replace_text.
- If the user asks for a Windows EXE/app from Python, write or copy the suitable Python file first, then apply package_python_exe to that file.
- If no file change is needed, use "actions":[].
- If unsure, prefer "actions":[] and a clear summary.

JSON schema:
{{
  "summary": "short internal summary",
  "actions": []
}}

Project folder:
{project_dir}

User request:
{user_request}

Coordinator goal:
{coordinator_summary}

Project snapshot:
{project_snapshot}

JSON:"#,
        project_dir = project_root.display(),
        user_request = &config.user_request,
        coordinator_summary = &coordinator.summary,
        project_snapshot = project_snapshot,
    );
    let coder_prompt = format!(
        "{}\n\nUniversal Python EXE rule:\n- If the user wants an EXE/app from Python, inspect the snapshot.\n- If the matching .py file already exists, use package_python_exe on that file.\n- If the Python file is attached or outside the project, first use copy_file to bring it into the project, then package_python_exe the copied file.\n- If it does not exist or is empty/unusable, first write a runnable .py file with write_file, then use package_python_exe.\n- Packaging automatically uses a temporary workspace, syntax check, short test run, diagnostics, and then copies the EXE back to dist.\n- This must work for any Python file, not only Hello World.\n\nExternal file import rule:\n- If the user wants to copy/import a file from outside the project, or mentions an \"Attached file:\" path, use this action:\n  {{\"op\":\"copy_file\",\"source\":\"absolute/or/external/source/path\",\"path\":\"relative/target/path/inside/project\"}}\n- source may be absolute and outside the project.\n- path must be relative and inside the project.\n- If no target name is specified, use the source file name at the project root or in a sensible relative folder.\n\nReply now only with the JSON object:",
        coder_prompt
    );

    let mut coder_config = config.clone();
    coder_config.model = primary_coder_model.to_string();
    let first_attempt = run_coder_candidates(
        &coder_config,
        &project_root,
        &coder_prompt,
        effective_agent_count,
    );
    let coder_result = match first_attempt {
        Ok(result) => Ok(result),
        Err(first_error) if coder_config.model != LOCAL_GPT_ROUTER_MODEL => {
            log_lines.push(format!(
                "Primary coder failed; GPT Coding Agent retries: {}",
                truncate_text(&first_error, 300)
            ));
            coder_config.model = LOCAL_GPT_ROUTER_MODEL.to_string();
            run_coder_candidates(&coder_config, &project_root, &coder_prompt, 1).map_err(
                |gpt_error| format!("{}; GPT retry also failed: {}", first_error, gpt_error),
            )
        }
        Err(error) => Err(error),
    };

    let mut envelope = match coder_result {
        Ok((envelope, candidate_logs)) => {
            log_lines.extend(candidate_logs);
            envelope
        }
        Err(e) => {
            log_lines.push(format!(
                "Agent manager could not create actions: {}",
                e
            ));
            match run_emergency_coder(
                &config,
                &project_root,
                &project_snapshot,
                &coordinator.summary,
                &mut log_lines,
            ) {
                Ok(envelope) => envelope,
                Err(emergency_error) => {
                    log_lines.push(format!("Emergency coding fallback failed: {}", emergency_error));
                    return AgentResult {
                        final_answer: coordinator_direct_fallback(
                            &config.user_request,
                            "The GPT and Qwen coding attempts did not produce a safe applicable change. The project folder was left unchanged.",
                        ),
                        log_lines,
                    };
                }
            }
        }
    };

    if coordinator.needs_code && envelope.actions.is_empty() {
        log_lines.push(
            "Empty actions are not accepted for this requested file change; GPT Coding Agent retries."
                .to_string(),
        );
        let retry_prompt = format!(
            "{}\n\nCRITICAL RETRY RULE:\nThe user explicitly requested a project change. The previous coder returned no actions. You MUST inspect the snapshot and return at least one safe write_file or replace_text action that implements the request. Do not claim it is already addressed unless the requested functionality is visibly present in the snapshot.",
            coder_prompt
        );
        let mut retry_config = config.clone();
        retry_config.model = LOCAL_GPT_ROUTER_MODEL.to_string();
        match run_coder_candidates(&retry_config, &project_root, &retry_prompt, 1) {
            Ok((retry_envelope, retry_logs)) if !retry_envelope.actions.is_empty() => {
                log_lines.extend(retry_logs);
                envelope = retry_envelope;
            }
            Ok((_, retry_logs)) => {
                log_lines.extend(retry_logs);
                log_lines.push("GPT Coding Agent also returned no file actions.".to_string());
                return AgentResult {
                    final_answer: coordinator_direct_fallback(
                        &config.user_request,
                        "The coding models returned no applicable file change. The project folder was left unchanged.",
                    ),
                    log_lines,
                };
            }
            Err(error) => {
                log_lines.push(format!("GPT Coding Agent retry failed: {}", error));
                return AgentResult {
                    final_answer: coordinator_direct_fallback(
                        &config.user_request,
                        "The GPT coding retry failed. The project folder was left unchanged.",
                    ),
                    log_lines,
                };
            }
        }
    }

    if let Some(summary) = envelope.summary.as_ref().filter(|s| !s.trim().is_empty()) {
        log_lines.push(format!(
            "Internal summary: {}",
            truncate_text(summary, 240)
        ));
    }

    let mut changed_files = Vec::new();
    let mut action_had_error = false;
    let mut test_output = String::new();

    if envelope.actions.is_empty() {
        log_lines.push("Coordinator applies no file changes.".to_string());
    } else if config.auto_apply_actions {
        log_lines.push(format!(
            "Applying {} file action(s)...",
            envelope.actions.len()
        ));
        let mut report = apply_file_actions(&project_root, &envelope.actions);
        if report.had_error && report.changed_files.is_empty() {
            let first_report_lines = report.log_lines.clone();
            log_lines.extend(first_report_lines.clone());
            log_lines.push("Repair agent checks blocked file actions...".to_string());

            let raw_actions = serde_json::to_string(&envelope).unwrap_or_else(|_| {
                envelope
                    .summary
                    .clone()
                    .unwrap_or_else(|| "Actions could not be serialized.".to_string())
            });

            match repair_action_envelope(
                &config,
                &project_root,
                &raw_actions,
                &first_report_lines.join("\n"),
                &mut log_lines,
            ) {
                Ok(repaired) => {
                    let validation_errors = validate_action_envelope(&project_root, &repaired);
                    if validation_errors.is_empty() && !repaired.actions.is_empty() {
                        log_lines
                            .push("Repair agent liefert sichere Ersatz-Actions.".to_string());
                        envelope = repaired;
                        report = apply_file_actions(&project_root, &envelope.actions);
                        log_lines.extend(report.log_lines.clone());
                    } else {
                        log_lines.push(format!(
                            "Repair agent verworfen: {}",
                            if validation_errors.is_empty() {
                                "no file actions".to_string()
                            } else {
                                validation_errors.join("; ")
                            }
                        ));
                    }
                }
                Err(e) => {
                    log_lines.push(format!("Repair agent failed: {}", e));
                }
            }
        } else {
            log_lines.extend(report.log_lines.clone());
        }

        action_had_error = report.had_error;
        changed_files = report.changed_files;
    } else {
        log_lines.push(format!(
            "{} file action(s) created, but auto-apply is off.",
            envelope.actions.len()
        ));
    }

    if config.run_tests_after_apply
        && !changed_files.is_empty()
        && !config.terminal_cmd.trim().is_empty()
    {
        log_lines.push(format!("Tester runs: {}", config.terminal_cmd));
        test_output = run_shell(
            project_root.to_string_lossy().as_ref(),
            &config.terminal_cmd,
        );
        log_lines.push("Tester ist fertig.".to_string());
    }

    if config.show_progress {
        log_lines.push("Coordinator writes the final answer...".to_string());
    }

    let deterministic = fallback_final_answer(
        envelope.summary.as_deref(),
        &changed_files,
        envelope.actions.len(),
        action_had_error,
        config.auto_apply_actions,
        &test_output,
    );

    let final_prompt = format!(
        r#"You are the coordinator agent. You are the only agent that talks to the user.

Speak naturally like a helpful coding assistant.
No JSON blocks, no internal actions, no raw API data.
Use the same language as the user.
When the user asks for commands or code, include the executable result in fenced Markdown with the correct language tag.
Never describe a command without also printing the command itself.
Do not mention coder-agent errors. If no safe change was applied, simply say what you understood and what the next step is.
No lectures about permissions and no invented risks. Say only what actually happened.

User request:
{user_request}

Coordinator goal:
{coordinator_summary}

Internal coder summary:
{summary}

Changed files:
{changed_files}

Auto-apply:
{auto_apply}

Action errors:
{action_had_error}

Shortened test output:
{test_output}

Fallback answer, if you only need to polish it:
{deterministic}

Final answer:"#,
        user_request = &config.user_request,
        coordinator_summary = truncate_text(&coordinator.summary, 1200),
        summary = envelope.summary.as_deref().unwrap_or(""),
        changed_files = if changed_files.is_empty() {
            "(none)".to_string()
        } else {
            changed_files.join(", ")
        },
        auto_apply = config.auto_apply_actions,
        action_had_error = action_had_error,
        test_output = truncate_text(&test_output, 2400),
        deterministic = deterministic,
    );

    let use_deterministic_final =
        !envelope.actions.is_empty() || !changed_files.is_empty() || action_had_error;

    let final_answer = if use_deterministic_final {
        deterministic
    } else {
        ollama_generate(&config.ollama_path, &config.model, &final_prompt)
            .map(|answer| clean_visible_answer(&answer))
            .unwrap_or(deterministic)
    };

    AgentResult {
        final_answer,
        log_lines,
    }
}

fn direct_chat_answer(user_request: &str) -> Option<String> {
    let normalized = normalize_for_intent(user_request);

    if matches!(
        normalized.as_str(),
        "hallo" | "hi" | "hey" | "servus" | "moin" | "guten tag" | "guten morgen" | "guten abend"
    ) {
        return Some(
            "Hello. I am ready. Tell me what to do in the selected project."
                .to_string(),
        );
    }

    if normalized == "danke" || normalized == "dankeschon" || normalized == "danke dir" {
        return Some("You're welcome.".to_string());
    }

    None
}

fn parse_local_file_task(user_request: &str, fallback_path: Option<&str>) -> Option<LocalFileTask> {
    let normalized = normalize_for_intent(user_request);
    let explicit_path = extract_requested_file_path(user_request);
    let fallback_path = fallback_path.filter(|path| looks_like_file_path(&path.replace('\\', "/")));
    let code_or_pack_task = normalized.contains("exe")
        || normalized.contains("app")
        || normalized.contains("programm")
        || normalized.contains("code")
        || normalized.contains("python")
        || normalized.contains("pyinstaller")
        || explicit_path
            .as_deref()
            .is_some_and(|path| path.to_ascii_lowercase().ends_with(".py"));

    if code_or_pack_task {
        return None;
    }

    let mentions_file = normalized.contains("datei")
        || normalized.contains("file")
        || normalized.contains(".txt")
        || normalized.contains(".md")
        || normalized.contains(".json")
        || normalized.contains(".rs")
        || explicit_path.is_some();

    let create_like = normalized.contains("erstell")
        || normalized.contains("anleg")
        || normalized.contains("lege")
        || normalized.contains("mach")
        || normalized.contains("schreib")
        || normalized.contains("speicher")
        || normalized.contains("create")
        || normalized.contains("make")
        || normalized.contains("write")
        || normalized.contains("save");

    let follow_up_write =
        fallback_path.is_some()
            && (normalized.contains("schreib")
                || normalized.contains("rein")
                || normalized.contains("write")
                || normalized.contains("add"));

    if !create_like && !follow_up_write {
        return None;
    }

    if !mentions_file && !follow_up_write {
        return None;
    }

    let Some(path) = explicit_path.or_else(|| fallback_path.map(str::to_string)) else {
        return Some(LocalFileTask::NeedFileName);
    };

    let content = extract_requested_file_content(user_request)
        .or_else(|| extract_followup_write_content(user_request))
        .unwrap_or_default();
    Some(LocalFileTask::CreateFile { path, content })
}

fn last_file_from_messages(messages: &[ChatMsg]) -> Option<String> {
    messages
        .iter()
        .rev()
        .filter_map(|message| extract_requested_file_path(&message.text))
        .next()
}

fn run_local_file_task(
    task: LocalFileTask,
    project_root: &Path,
    mut log_lines: Vec<String>,
) -> AgentResult {
    match task {
        LocalFileTask::NeedFileName => AgentResult {
            final_answer:
                "Sure. What should the file be called? For example: `create text.txt`."
                    .to_string(),
            log_lines,
        },
        LocalFileTask::CreateFile { path, content } => {
            log_lines.push("Local tool: create file.".to_string());
            let action = FileAction::WriteFile {
                path: path.clone(),
                content,
            };
            let report = apply_file_actions(project_root, &[action]);
            log_lines.extend(report.log_lines);

            if report.had_error || report.changed_files.is_empty() {
                AgentResult {
                    final_answer: format!(
                        "I could not safely create `{}`. Please check that the project folder is correct and the filename is relative.",
                        path
                    ),
                    log_lines,
                }
            } else {
                AgentResult {
                    final_answer: format!(
                        "Done. I created `{}`.",
                        report.changed_files[0]
                    ),
                    log_lines,
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct PythonRuntime {
    program: String,
    prefix_args: Vec<String>,
}

fn run_local_python_pack_task(
    script_path: &str,
    bootstrap_content: Option<&str>,
    project_root: &Path,
    mut log_lines: Vec<String>,
) -> AgentResult {
    log_lines.push("Local tool: package Python as EXE.".to_string());

    let script = match safe_project_path(project_root, script_path) {
        Ok(path) => path,
        Err(e) => {
            log_lines.push(format!("Python packaging blocked: {}", e));
            return AgentResult {
                final_answer: format!("I could not safely package `{}`: {}", script_path, e),
                log_lines,
            };
        }
    };

    if !script
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("py"))
    {
        return AgentResult {
            final_answer: format!("`{}` is not a Python file.", script_path),
            log_lines,
        };
    }

    if let Some(content) = bootstrap_content {
        if let Some(parent) = script.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return AgentResult {
                    final_answer: format!(
                        "I could not create the folder for `{}`: {}",
                        script_path, e
                    ),
                    log_lines,
                };
            }
        }

        if let Err(e) = fs::write(&script, content) {
            return AgentResult {
                final_answer: format!(
                    "I could not write `{}` as the start file: {}",
                    script_path, e
                ),
                log_lines,
            };
        }
        log_lines.push(format!(
            "Python file created/updated: {}",
            script_path
        ));
    }

    if !script.exists() {
        return AgentResult {
            final_answer: format!("I cannot find `{}` in the project folder.", script_path),
            log_lines,
        };
    }

    let python = match find_python_runtime() {
        Ok(runtime) => runtime,
        Err(e) => {
            log_lines.push(format!("Python is not usable: {}", e));
            return AgentResult {
                final_answer: format!(
                    "I could not find a usable Python installation.\n\n{}",
                    e
                ),
                log_lines,
            };
        }
    };

    let tools_dir = match ensure_local_pyinstaller(&python, project_root, &mut log_lines) {
        Ok(dir) => dir,
        Err(e) => {
            return AgentResult {
                final_answer: format!("I could not prepare PyInstaller.\n\n{}", e),
                log_lines,
            };
        }
    };

    let exe_name = script
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("app")
        .to_string();
    let work_root = project_root
        .join("build")
        .join("package_work")
        .join(format!("{}_{}", sanitize_file_stem(&exe_name), now_millis()));
    let source_root = work_root.join("source");
    let temp_dist_dir = work_root.join("dist");
    let temp_build_dir = work_root.join("pyinstaller_build");
    let final_dist_dir = project_root.join("dist");
    let final_exe_path = final_dist_dir.join(format!("{}.exe", exe_name));
    let relative_script = script
        .strip_prefix(project_root)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| PathBuf::from(script.file_name().unwrap_or_default()));

    if let Err(e) = copy_packaging_workspace(project_root, &source_root, &mut log_lines) {
        return AgentResult {
            final_answer: format!(
                "I could not create the temporary packaging workspace.\n\n{}",
                e
            ),
            log_lines,
        };
    }

    let temp_script = source_root.join(&relative_script);
    if !temp_script.exists() {
        return AgentResult {
            final_answer: format!(
                "The temporary packaging workspace was created, but the script was not copied: {}",
                relative_script.display()
            ),
            log_lines,
        };
    }

    if let Err(e) = run_python_packaging_preflight(&python, &temp_script, &source_root, &mut log_lines) {
        return AgentResult {
            final_answer: format!(
                "Packaging stopped during the temporary test run.\n\n{}",
                e
            ),
            log_lines,
        };
    }

    let mut command = python_command(&python);
    command
        .arg("-m")
        .arg("PyInstaller")
        .arg("--onefile")
        .arg("--clean")
        .arg("--noconfirm")
        .arg("--distpath")
        .arg(&temp_dist_dir)
        .arg("--workpath")
        .arg(&temp_build_dir)
        .arg("--specpath")
        .arg(&temp_build_dir)
        .arg("--name")
        .arg(&exe_name)
        .arg(&temp_script)
        .current_dir(&source_root)
        .env("PYTHONPATH", &tools_dir);

    log_lines.push(format!(
        "PyInstaller builds from temporary workspace: {}",
        temp_script.display()
    ));
    let output = match run_command_with_timeout(command, Duration::from_secs(240)) {
        Ok(output) => output,
        Err(e) => {
            return AgentResult {
                final_answer: format!("PyInstaller could not be started: {}", e),
                log_lines,
            };
        }
    };

    let build_output = command_output_text(&output);
    if !output.status.success() {
        log_lines.push(format!(
            "PyInstaller error: {}",
            truncate_text(&build_output, 1200)
        ));
        return AgentResult {
            final_answer: format!(
                "Packaging failed.\n\n{}",
                truncate_text(&build_output, 1800)
            ),
            log_lines,
        };
    }

    let temp_exe_path = temp_dist_dir.join(format!("{}.exe", exe_name));
    if !temp_exe_path.exists() {
        log_lines.push("PyInstaller finished, but the EXE was not found.".to_string());
        return AgentResult {
            final_answer: format!(
                "PyInstaller completed, but I cannot find the expected EXE: {}",
                temp_exe_path.display()
            ),
            log_lines,
        };
    }

    if let Err(e) = fs::create_dir_all(&final_dist_dir) {
        return AgentResult {
            final_answer: format!("I could not create the final dist folder: {}", e),
            log_lines,
        };
    }

    if let Err(e) = fs::copy(&temp_exe_path, &final_exe_path) {
        return AgentResult {
            final_answer: format!(
                "The EXE was built, but I could not copy it back to `dist`: {}",
                e
            ),
            log_lines,
        };
    }

    log_lines.push(format!("EXE created: {}", final_exe_path.display()));
    let smoke = run_exe_smoke_test(&final_exe_path, project_root, Duration::from_secs(5));
    log_lines.push(format!("EXE smoke test: {}", truncate_text(&smoke, 300)));

    AgentResult {
        final_answer: format!(
            "Done. I packaged `{}` as an EXE.\n\nEXE: `{}`\n\nSmoke test:\n{}",
            script_path,
            final_exe_path.display(),
            smoke.trim()
        ),
        log_lines,
    }
}

fn copy_packaging_workspace(
    project_root: &Path,
    source_root: &Path,
    log_lines: &mut Vec<String>,
) -> Result<(), String> {
    fs::create_dir_all(source_root)
        .map_err(|e| format!("Temporary source folder could not be created: {}", e))?;

    let mut copied = 0usize;
    let mut skipped_large = 0usize;
    let max_file_bytes = 50 * 1024 * 1024;

    for entry in WalkDir::new(project_root).into_iter().filter_entry(|entry| {
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        !matches!(
            name.as_str(),
            ".git"
                | "target"
                | "build"
                | "dist"
                | "__pycache__"
                | ".pytest_cache"
                | ".mypy_cache"
                | ".ruff_cache"
                | ".venv"
                | "venv"
                | "env"
                | "node_modules"
        )
    }) {
        let entry = entry.map_err(|e| format!("Project scan failed: {}", e))?;
        let path = entry.path();
        if path == project_root || entry.file_type().is_dir() {
            continue;
        }

        let rel = path
            .strip_prefix(project_root)
            .map_err(|e| format!("Could not resolve relative path: {}", e))?;
        let metadata = entry
            .metadata()
            .map_err(|e| format!("Could not read file metadata for {}: {}", rel.display(), e))?;

        if metadata.len() > max_file_bytes {
            skipped_large += 1;
            continue;
        }

        let target = source_root.join(rel);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Could not create temp folder {}: {}", parent.display(), e))?;
        }
        fs::copy(path, &target).map_err(|e| {
            format!(
                "Could not copy {} to temporary workspace: {}",
                rel.display(),
                e
            )
        })?;
        copied += 1;
    }

    log_lines.push(format!(
        "Temporary packaging workspace prepared: {} file(s) copied, {} large file(s) skipped.",
        copied, skipped_large
    ));
    Ok(())
}

fn run_python_packaging_preflight(
    python: &PythonRuntime,
    script: &Path,
    source_root: &Path,
    log_lines: &mut Vec<String>,
) -> Result<(), String> {
    let mut compile = python_command(python);
    compile
        .arg("-m")
        .arg("py_compile")
        .arg(script)
        .current_dir(source_root);

    let compile_output = run_command_with_timeout(compile, Duration::from_secs(30))
        .map_err(|e| format!("Python syntax check could not finish:\n{}", e))?;
    let compile_text = command_output_text(&compile_output);
    if !compile_output.status.success() {
        return Err(format!(
            "Python syntax check failed:\n{}",
            truncate_text(&compile_text, 1800)
        ));
    }
    log_lines.push("Temporary test: Python syntax check passed.".to_string());

    let mut smoke = python_command(python);
    smoke.arg(script).current_dir(source_root);
    match run_command_with_timeout(smoke, Duration::from_secs(10)) {
        Ok(output) if output.status.success() => {
            log_lines.push(format!(
                "Temporary test run passed: {}",
                truncate_text(command_output_text(&output).trim(), 300)
            ));
            Ok(())
        }
        Ok(output) => Err(format!(
            "Temporary test run failed:\n{}",
            truncate_text(&command_output_text(&output), 1800)
        )),
        Err(e) => {
            log_lines.push(format!(
                "Temporary test run did not finish quickly; continuing because GUI apps often keep running. Diagnostic: {}",
                truncate_text(&e, 500)
            ));
            Ok(())
        }
    }
}

fn find_python_runtime() -> Result<PythonRuntime, String> {
    let candidates = [
        PythonRuntime {
            program: "python".to_string(),
            prefix_args: Vec::new(),
        },
        PythonRuntime {
            program: "py".to_string(),
            prefix_args: vec!["-3".to_string()],
        },
        PythonRuntime {
            program: r"C:\Users\tarek\miniconda3\python.exe".to_string(),
            prefix_args: Vec::new(),
        },
    ];

    let mut errors = Vec::new();
    for candidate in candidates {
        let mut command = python_command(&candidate);
        command.arg("--version");
        match run_command_with_timeout(command, Duration::from_secs(10)) {
            Ok(output) if output.status.success() => return Ok(candidate),
            Ok(output) => errors.push(command_output_text(&output)),
            Err(e) => errors.push(e.to_string()),
        }
    }

    Err(errors.join("\n"))
}

fn python_command(runtime: &PythonRuntime) -> Command {
    let mut command = hidden_command(&runtime.program);
    command.args(&runtime.prefix_args);
    command
}

fn sanitize_file_stem(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if sanitized.trim_matches('_').is_empty() {
        "app".to_string()
    } else {
        sanitized
    }
}

fn ensure_local_pyinstaller(
    python: &PythonRuntime,
    project_root: &Path,
    log_lines: &mut Vec<String>,
) -> Result<PathBuf, String> {
    let app_dir = std::env::current_dir().unwrap_or_else(|_| project_root.to_path_buf());
    let tools_dir = app_dir.join(".local_ai_builder").join("packager_tools");

    if !tools_dir.join("PyInstaller").exists() {
        log_lines.push("PyInstaller is missing locally; trying installation...".to_string());
        fs::create_dir_all(&tools_dir)
            .map_err(|e| format!("Packager folder could not be created: {}", e))?;

        let mut install = python_command(python);
        install
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("--target")
            .arg(&tools_dir)
            .arg("pyinstaller");

        let output = run_command_with_timeout(install, Duration::from_secs(120))
            .map_err(|e| format!("pip could not be started: {}", e))?;
        if !output.status.success() {
            return Err(format!(
                "pip install pyinstaller failed.\n{}",
                truncate_text(&command_output_text(&output), 1800)
            ));
        }
    }

    let mut version = python_command(python);
    version
        .arg("-m")
        .arg("PyInstaller")
        .arg("--version")
        .env("PYTHONPATH", &tools_dir);
    let output = run_command_with_timeout(version, Duration::from_secs(20))
        .map_err(|e| format!("PyInstaller could not be checked: {}", e))?;

    if output.status.success() {
        log_lines.push(format!(
            "PyInstaller ready: {}",
            truncate_text(command_output_text(&output).trim(), 120)
        ));
        Ok(tools_dir)
    } else {
        Err(format!(
            "PyInstaller is not usable.\n{}",
            truncate_text(&command_output_text(&output), 1800)
        ))
    }
}

fn run_command_with_timeout(
    mut command: Command,
    timeout: Duration,
) -> Result<std::process::Output, String> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|e| format!("Command could not be started: {}", e))?;
    let started = Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                return child
                    .wait_with_output()
                    .map_err(|e| format!("Command output could not be read: {}", e));
            }
            Ok(None) if started.elapsed() < timeout => thread::sleep(Duration::from_millis(100)),
            Ok(None) => {
                let _ = child.kill();
                let output = child.wait_with_output().map_err(|e| {
                    format!("Command killed after timeout; output could not be read: {}", e)
                })?;
                let mut text = command_output_text(&output);
                if !text.trim().is_empty() {
                    text = format!(
                        "Timeout nach {} Sekunden.\n{}",
                        timeout.as_secs(),
                        truncate_text(&text, 1200)
                    );
                } else {
                    text = format!("Timeout nach {} Sekunden.", timeout.as_secs());
                }
                return Err(text);
            }
            Err(e) => return Err(format!("Command could not be monitored: {}", e)),
        }
    }
}

fn run_exe_smoke_test(exe_path: &Path, project_root: &Path, timeout: Duration) -> String {
    let mut child = match hidden_command(exe_path)
        .current_dir(project_root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => return format!("Smoke test could not be started: {}", e),
    };

    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                return child
                    .wait_with_output()
                    .map(|output| command_output_text(&output))
                    .unwrap_or_else(|e| {
                        format!("Smoke test output could not be read: {}", e)
                    });
            }
            Ok(None) if started.elapsed() < timeout => thread::sleep(Duration::from_millis(100)),
            Ok(None) => {
                let _ = child.kill();
                let output = child.wait_with_output();
                return output
                    .map(|output| {
                        let text = command_output_text(&output);
                        if text.trim().is_empty() {
                            "Smoke test ended after timeout.".to_string()
                        } else {
                            format!("Smoke test timeout. Output so far:\n{}", text)
                        }
                    })
                    .unwrap_or_else(|_| "Smoke test ended after timeout.".to_string());
            }
            Err(e) => return format!("Smoke test error: {}", e),
        }
    }
}

fn command_output_text(output: &std::process::Output) -> String {
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(&output.stdout));
    text.push_str(&String::from_utf8_lossy(&output.stderr));

    if text.trim().is_empty() {
        format!("Exit-Code: {:?}", output.status.code())
    } else {
        text
    }
}

fn normalize_for_intent(text: &str) -> String {
    text.trim()
        .to_lowercase()
        .replace('ä', "ae")
        .replace('ö', "oe")
        .replace('ü', "ue")
        .replace('ß', "ss")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn extract_requested_file_path(user_request: &str) -> Option<String> {
    let cleaned = user_request
        .replace('`', " ")
        .replace('"', " ")
        .replace('\'', " ");

    for raw in cleaned.split_whitespace() {
        let token = raw
            .trim_matches(|c: char| {
                matches!(
                    c,
                    ',' | ';' | ':' | ')' | '(' | '[' | ']' | '{' | '}' | '!' | '?'
                )
            })
            .replace('\\', "/");

        if looks_like_file_path(&token) {
            return Some(token);
        }
    }

    let normalized = normalize_for_intent(user_request);
    if let Some(rest) = normalized.split("with the name ").nth(1) {
        return fallback_name_to_txt(rest);
    }
    if let Some(rest) = normalized.split("with name ").nth(1) {
        return fallback_name_to_txt(rest);
    }
    if let Some(rest) = normalized.split("named ").nth(1) {
        return fallback_name_to_txt(rest);
    }
    if let Some(rest) = normalized.split("mit namen ").nth(1) {
        return fallback_name_to_txt(rest);
    }
    if let Some(rest) = normalized.split("namens ").nth(1) {
        return fallback_name_to_txt(rest);
    }
    if let Some(rest) = normalized.split("mit ").nth(1) {
        return fallback_name_to_txt(rest);
    }

    None
}

fn fallback_name_to_txt(rest: &str) -> Option<String> {
    let name = rest
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_');

    if name.is_empty() || matches!(name, "inhalt" | "text" | "dem" | "der" | "den") {
        return None;
    }

    if name.contains('.') {
        Some(name.to_string())
    } else {
        Some(format!("{}.txt", name))
    }
}

fn looks_like_file_path(token: &str) -> bool {
    let token_lc = token.to_lowercase();
    if token_lc.contains("://") || token_lc.starts_with("http") {
        return false;
    }

    let Some(file_name) = token_lc.rsplit('/').next() else {
        return false;
    };

    let Some((stem, ext)) = file_name.rsplit_once('.') else {
        return false;
    };

    !stem.is_empty()
        && matches!(
            ext,
            "txt"
                | "md"
                | "json"
                | "toml"
                | "yaml"
                | "yml"
                | "rs"
                | "js"
                | "ts"
                | "tsx"
                | "jsx"
                | "html"
                | "css"
                | "py"
                | "bat"
                | "ps1"
                | "csv"
                | "xml"
        )
}

fn extract_requested_file_content(user_request: &str) -> Option<String> {
    let lower = user_request.to_lowercase();

    for marker in [
        " with content ",
        " with the content ",
        " with text ",
        " with the text ",
        " content: ",
        " mit inhalt ",
        " mit dem inhalt ",
        " mit text ",
        " mit dem text ",
        " inhalt: ",
        " text: ",
    ] {
        if let Some(pos) = lower.find(marker) {
            let start = pos + marker.len();
            return clean_content_fragment(&user_request[start..]);
        }
    }

    None
}

fn extract_followup_write_content(user_request: &str) -> Option<String> {
    let lower = user_request.to_lowercase();

    for marker in ["write ", "add "] {
        if let Some(pos) = lower.rfind(marker) {
            let start = pos + marker.len();
            let mut content = user_request[start..].trim();
            for prefix in ["please ", "the text ", "the content "] {
                if content.to_lowercase().starts_with(prefix) {
                    content = content[prefix.len()..].trim_start();
                }
            }

            if let Some(end) = content.to_lowercase().find(" rein") {
                return clean_content_fragment(&content[..end]);
            }

            return clean_content_fragment(content);
        }
    }

    None
}

fn clean_content_fragment(fragment: &str) -> Option<String> {
    let mut content = fragment.trim();

    if let Some(cut) = first_content_boundary(content) {
        content = &content[..cut];
    }

    let content = content.trim().trim_matches(|c: char| {
        matches!(
            c,
            '`' | '"' | '\'' | ',' | ';' | ':' | '.' | '!' | '?' | ')' | '(' | '[' | ']'
        )
    });

    if content.is_empty() {
        None
    } else {
        Some(content.to_string())
    }
}

fn first_content_boundary(content: &str) -> Option<usize> {
    let lower = content.to_lowercase();
    [
        " and the name ",
        " and name ",
        " and filename ",
        " with the name ",
        " with name ",
        " named ",
        " into the file ",
        " into file ",
        " und dem namen ",
        " und den namen ",
        " und namen ",
        " und name ",
        " und dem dateinamen ",
        " und den dateinamen ",
        " mit dem namen ",
        " mit namen ",
        " namens ",
        " in die datei ",
        " in datei ",
    ]
    .iter()
    .filter_map(|boundary| lower.find(boundary))
    .min()
}

fn run_coordinator_decision(
    config: &AgentRunConfig,
    project_root: &Path,
    project_snapshot: &str,
    memory_context: &str,
    log_lines: &mut Vec<String>,
) -> CoordinatorDecision {
    let mut prompt = format!(
        r#"You are the coordinator of a local coding assistant.

You are the only agent that may talk to the user later.
You decide whether to answer directly or whether other agents should plan and change files.

Reply only with JSON, no Markdown.

Schema:
{{
  "reply_now": false,
  "needs_code": true,
  "summary": "what the user wants and how you route it",
  "direct_reply": "only use when no code work is needed"
}}

Rules:
- For small talk, explanation, or advice without file changes: reply_now=true, needs_code=false.
- Questions asking for PowerShell, CMD, Bash, terminal commands, code snippets, examples, or scripts are direct answers: reply_now=true, needs_code=false, unless the user explicitly asks to save or apply them to project files.
- For commands or code, direct_reply must contain executable code in fenced Markdown with the correct language tag, for example ```powershell, ```bash, ```python, or ```rust.
- Put each command on its own line. Do not replace code with a prose description.
- Briefly explain potentially destructive commands before showing them.
- For requested project changes, repair, build work, or UI changes: reply_now=false, needs_code=true.
- direct_reply must be natural and use the same language as the user.
- No internal agent names in direct_reply.

Project folder:
{project_dir}

User request:
{user_request}

Project snapshot:
{project_snapshot}

Persistent memory:
{memory_context}

JSON:"#,
        project_dir = project_root.display(),
        user_request = &config.user_request,
        project_snapshot = truncate_text(project_snapshot, config.context_limit.min(80_000)),
        memory_context = truncate_text(memory_context, 12_000),
    );
    prompt.push_str(
        r#"

IMPORTANT ROUTING OVERRIDE:
Return JSON with these fields:
{
  "reply_now": false,
  "needs_code": true,
  "summary": "short routing summary",
  "direct_reply": "",
  "recommended_agent_count": 1,
  "use_large_single_coder": true
}

Use recommended_agent_count=1 for clear/simple coding tasks, single-file work, or straightforward Python EXE packaging.
Use recommended_agent_count=2 for medium uncertainty.
Use recommended_agent_count=3..6 only for complex multi-file refactors, failing builds, repeated repair loops, or unclear architecture work.
Prefer speed. Use fewer agents unless more agents are genuinely needed.
All visible user-facing text must use the same language as the user.
"#,
    );

    log_lines.push(format!(
        "TRAFFIC GPT Router -> {} | Prompt:\n{}",
        LOCAL_GPT_ROUTER_MODEL,
        truncate_text(&prompt, 4_000)
    ));

    match ollama_generate_with_fallback(
        &config.ollama_path,
        LOCAL_GPT_ROUTER_MODEL,
        &config.model,
        &prompt,
    ) {
        Ok(answer) => {
            log_lines.push(format!(
                "TRAFFIC {} -> GPT Router | Response:\n{}",
                LOCAL_GPT_ROUTER_MODEL,
                truncate_text(&answer, 4_000)
            ));
            match parse_coordinator_decision(&answer) {
                Ok(decision) => decision,
                Err(e) => {
                    log_lines.push(format!(
                        "Coordinator parser uses fallback: {} | raw answer: {}",
                        e,
                        truncate_text(&answer, 1_000)
                    ));
                    CoordinatorDecision {
                        reply_now: false,
                        needs_code: true,
                        summary: config.user_request.clone(),
                        direct_reply: String::new(),
                        recommended_agent_count: 1,
                        use_large_single_coder: true,
                    }
                }
            }
        }
        Err(e) => {
            log_lines.push(format!(
                "TRAFFIC GPT Router <- {} (fallback: {}) failed: {}",
                LOCAL_GPT_ROUTER_MODEL, config.model, e
            ));
            CoordinatorDecision {
            reply_now: false,
            needs_code: true,
            summary: config.user_request.clone(),
            direct_reply: String::new(),
            recommended_agent_count: 1,
            use_large_single_coder: true,
            }
        }
    }
}

fn parse_coordinator_decision(raw: &str) -> Result<CoordinatorDecision, String> {
    let trimmed = raw.trim();

    if let Ok(decision) = serde_json::from_str::<CoordinatorDecision>(trimmed) {
        return Ok(decision);
    }

    if let Some(json_text) = extract_first_json_object(trimmed) {
        return serde_json::from_str::<CoordinatorDecision>(&json_text)
            .map_err(|e| format!("Coordinator JSON parser error: {}", e));
    }

    Err("no coordinator decision found".to_string())
}

fn repair_action_envelope(
    config: &AgentRunConfig,
    project_root: &Path,
    coder_answer: &str,
    error_context: &str,
    log_lines: &mut Vec<String>,
) -> Result<ActionEnvelope, String> {
    let prompt = format!(
        r#"You are an internal repair agent.

Convert the following messy coder output into safe file actions.
If you cannot infer an absolutely safe change, return actions:[].

Respond only with one JSON object and no Markdown.

Schema:
{{
  "summary": "short internal summary",
  "actions": []
}}

Allowed operations:
- {{"op":"replace_text","path":"relative/path","find":"exact old text","replace":"new text"}}
- {{"op":"write_file","path":"relative/path","content":"complete file content"}}
- {{"op":"append_file","path":"relative/path","content":"text to append"}}
- {{"op":"package_python_exe","path":"relative/path.py"}}

Rules:
- Only relative paths below this project folder: {project_dir}
- No absolute paths.
- No paths with '..'.
- Correct known errors from the error context.
- When uncertain, use actions:[].

Messy coder output:
{coder_answer}

Error context:
{error_context}

JSON:"#,
        project_dir = project_root.display(),
        coder_answer = truncate_text(coder_answer, 30_000),
        error_context = truncate_text(error_context, 10_000),
    );
    let prompt = format!(
        "{}\n\nAdditional supported action:\n- {{\"op\":\"copy_file\",\"source\":\"absolute/or/external/source/path\",\"path\":\"relative/target/path/inside/project\"}}\nUse copy_file when the user's intent is to import/copy an external or attached file into the project. source may be outside the project, path must stay inside the project.\n",
        prompt
    );

    log_lines.push(format!(
        "TRAFFIC Agent manager -> Repair agent ({}) | Prompt:\n{}",
        config.model,
        truncate_text(&prompt, 4_000)
    ));

    let answer = ollama_generate(&config.ollama_path, &config.model, &prompt)?;
    log_lines.push(format!(
        "TRAFFIC Repair agent ({}) -> Agent manager | Response:\n{}",
        config.model,
        truncate_text(&answer, 4_000)
    ));
    parse_action_envelope(&answer)
}

fn run_coder_candidates(
    config: &AgentRunConfig,
    project_root: &Path,
    coder_prompt: &str,
    requested_agent_count: usize,
) -> Result<(ActionEnvelope, Vec<String>), String> {
    let specs = [
        AgentStep {
            name: "Primary Coder Agent",
            role: "Create a robust general solution and use existing project files when they fit.",
        },
        AgentStep {
            name: "GPT Coding Agent",
            role: "Use GPT reasoning to create a complete, precise coding solution, including shell and PowerShell commands when useful. Return safe file actions as JSON.",
        },
        AgentStep {
            name: "OpenClaw Orchestration Agent",
            role: "Think like an autonomous routing agent: split larger tasks into planning, file actions, build/package steps, and verification. Use tools only through safe actions.",
        },
        AgentStep {
            name: "Qwen-Coder-Agent",
            role: "Create the smallest safe patch as JSON.",
        },
        AgentStep {
            name: "DeepSeek-Coder-Agent",
            role: "Create an independent second solution as JSON and pay attention to runnable programs.",
        },
        AgentStep {
            name: "StarCoder-Agent",
            role: "Check multiple programming languages, build files, and project structure. Good for repository understanding and alternative patches.",
        },
        AgentStep {
            name: "DeepSeek-Gross-Coder-Agent",
            role: "Use stronger coding reasoning for complex refactors, build problems, EXE packaging, and error chains.",
        },
    ];
    let count = requested_agent_count.clamp(1, specs.len());

    let (tx, rx) = mpsc::channel();
    let mut logs = Vec::new();

    for (idx, spec) in specs.iter().take(count).enumerate() {
        let tx = tx.clone();
        let name = spec.name.to_string();
        let role = spec.role.to_string();
        let model = agent_model_for_index(idx, &config.model);
        let fallback_model = config.model.clone();
        let ollama_path = config.ollama_path.clone();
        let prompt = format!(
            "{coder_prompt}\n\nSpecial role for {name}:\n{role}\n\nReply now only with the JSON object:"
        );

        logs.push(format!(
            "TRAFFIC Agent manager -> {} ({}) | Role: {}\nPrompt:\n{}",
            name,
            model,
            role,
            truncate_text(&prompt, 4_000)
        ));

        thread::spawn(move || {
            let answer = ollama_generate(&ollama_path, &model, &prompt).or_else(|first_error| {
                if model == fallback_model {
                    Err(first_error)
                } else {
                    ollama_generate(&ollama_path, &fallback_model, &prompt).map_err(
                        |fallback_error| {
                            format!(
                                "{}; fallback {} also failed: {}",
                                first_error, fallback_model, fallback_error
                            )
                        },
                    )
                }
            });

            let _ = tx.send((name, model, answer));
        });
    }

    drop(tx);

    let mut results = Vec::new();
    let mut first_error = None;

    for _ in 0..count {
        if let Ok((name, model, answer)) = rx.recv_timeout(Duration::from_secs(270)) {
            match answer {
                Ok(raw) => {
                    logs.push(format!(
                        "TRAFFIC {} ({}) -> Agent manager | raw response:\n{}",
                        name,
                        model,
                        truncate_text(&raw, 4_000)
                    ));
                    let (envelope, parse_error, validation_errors) =
                        match parse_action_envelope(&raw) {
                            Ok(envelope) => {
                                let validation_errors =
                                    validate_action_envelope(project_root, &envelope);
                                (Some(envelope), None, validation_errors)
                            }
                            Err(e) => (None, Some(e), Vec::new()),
                        };

                    let action_count = envelope
                        .as_ref()
                        .map(|envelope| envelope.actions.len())
                        .unwrap_or(0);

                    if let Some(parse_error) = &parse_error {
                        logs.push(format!(
                            "{} ({}) returned no usable JSON: {}",
                            name,
                            model,
                            truncate_text(parse_error, 160)
                        ));
                    } else if validation_errors.is_empty() {
                        logs.push(format!(
                            "{} ({}) lieferte {} validierte Action(s).",
                            name, model, action_count
                        ));
                    } else {
                        logs.push(format!(
                            "{} ({}) lieferte {} Action(s), aber die Validierung stoppte: {}",
                            name,
                            model,
                            action_count,
                            truncate_text(&validation_errors.join("; "), 220)
                        ));
                    }

                    results.push(CoderCandidateResult {
                        name,
                        model,
                        raw,
                        envelope,
                        parse_error,
                        validation_errors,
                    });
                }
                Err(e) => {
                    logs.push(format!(
                        "TRAFFIC {} ({}) -> Agent manager | Error: {}",
                        name,
                        model,
                        truncate_text(&e, 1_000)
                    ));
                    logs.push(format!("{} ({}) not usable: {}", name, model, e));
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        } else {
            logs.push("Agent manager: A coder agent took too long to respond and was skipped.".to_string());
            if first_error.is_none() {
                first_error = Some("Mindestens ein Coder-Agent lief in einen Timeout.".to_string());
            }
        }
    }

    if let Some(best) = select_best_valid_coder_candidate(&results) {
        let envelope = best.envelope.clone().unwrap_or_else(empty_action_envelope);
        logs.push(format!(
            "Agent manager selects {} ({}) mit {} Action(s).",
            best.name,
            best.model,
            envelope.actions.len()
        ));
        return Ok((envelope, logs));
    }

    for candidate in &results {
        if candidate.raw.trim().is_empty() {
            continue;
        }

        let mut error_context = String::new();
        if let Some(parse_error) = &candidate.parse_error {
            error_context.push_str(parse_error);
        }
        if !candidate.validation_errors.is_empty() {
            if !error_context.is_empty() {
                error_context.push('\n');
            }
            error_context.push_str(&candidate.validation_errors.join("\n"));
        }

        logs.push(format!(
            "Repair agent checks candidate {} ({}).",
            candidate.name, candidate.model
        ));

        match repair_action_envelope(config, project_root, &candidate.raw, &error_context, &mut logs) {
            Ok(repaired) => {
                let validation_errors = validate_action_envelope(project_root, &repaired);
                if validation_errors.is_empty() && !repaired.actions.is_empty() {
                    logs.push(format!(
                        "Repair agent created {} sichere Action(s).",
                        repaired.actions.len()
                    ));
                    return Ok((repaired, logs));
                }

                logs.push(format!(
                    "Repair agent verworfen: {}",
                    if validation_errors.is_empty() {
                        "no file actions".to_string()
                    } else {
                        truncate_text(&validation_errors.join("; "), 220)
                    }
                ));
            }
            Err(e) => {
                logs.push(format!("Repair agent could not repair: {}", e));
            }
        }
    }

    if let Some(empty) = results
        .iter()
        .filter_map(|result| result.envelope.as_ref())
        .find(|envelope| envelope.actions.is_empty())
        .cloned()
    {
        logs.push("Agent manager uses a safe empty action list.".to_string());
        return Ok((empty, logs));
    }

    Err(first_error.unwrap_or_else(|| "Kein Coding-Agent lieferte sichere Actions.".to_string()))
}

fn run_emergency_coder(
    config: &AgentRunConfig,
    project_root: &Path,
    project_snapshot: &str,
    goal: &str,
    logs: &mut Vec<String>,
) -> Result<ActionEnvelope, String> {
    let prompt = format!(
        r#"Implement the requested project change now.
Return only one JSON object with this exact shape:
{{"summary":"short summary","actions":[{{"op":"write_file","path":"relative/file","content":"complete updated file"}}]}}

Rules:
- At least one action is required.
- Prefer write_file with the complete updated content when the project is small.
- Paths must be relative to the project folder.
- No Markdown and no prose outside JSON.

User request:
{request}

Goal:
{goal}

Project snapshot:
{snapshot}

JSON:"#,
        request = config.user_request,
        goal = goal,
        snapshot = truncate_text(project_snapshot, config.context_limit.min(40_000)),
    );

    let mut errors = Vec::new();
    for model in [LOCAL_GPT_ROUTER_MODEL, QWEN_CODER_MODEL] {
        logs.push(format!("Emergency coder -> {}", model));
        match ollama_generate(&config.ollama_path, model, &prompt) {
            Ok(raw) => {
                logs.push(format!(
                    "Emergency coder {} response:\n{}",
                    model,
                    truncate_text(&raw, 4_000)
                ));
                match parse_action_envelope(&raw) {
                    Ok(envelope) => {
                        let validation = validate_action_envelope(project_root, &envelope);
                        if validation.is_empty() && !envelope.actions.is_empty() {
                            return Ok(envelope);
                        }
                        errors.push(format!(
                            "{} returned unusable actions: {}",
                            model,
                            if validation.is_empty() {
                                "empty action list".to_string()
                            } else {
                                validation.join("; ")
                            }
                        ));
                    }
                    Err(error) => errors.push(format!("{} JSON parse: {}", model, error)),
                }
            }
            Err(error) => errors.push(format!("{} request: {}", model, error)),
        }
    }
    Err(errors.join(" | "))
}

fn empty_action_envelope() -> ActionEnvelope {
    ActionEnvelope {
        summary: Some("No safe file change created.".to_string()),
        actions: Vec::new(),
    }
}

fn select_best_valid_coder_candidate(
    results: &[CoderCandidateResult],
) -> Option<&CoderCandidateResult> {
    results
        .iter()
        .filter(|result| {
            result.validation_errors.is_empty()
                && result
                    .envelope
                    .as_ref()
                    .is_some_and(|envelope| !envelope.actions.is_empty())
        })
        .max_by_key(|result| coder_candidate_score(result))
        .or_else(|| {
            results.iter().find(|result| {
                result.validation_errors.is_empty()
                    && result
                        .envelope
                        .as_ref()
                        .is_some_and(|envelope| envelope.actions.is_empty())
            })
        })
}

fn coder_candidate_score(result: &CoderCandidateResult) -> i32 {
    let Some(envelope) = &result.envelope else {
        return -1000;
    };

    let action_count = envelope.actions.len() as i32;
    let replace_bonus = envelope
        .actions
        .iter()
        .filter(|action| matches!(action, FileAction::ReplaceText { .. }))
        .count() as i32;
    let write_count = envelope
        .actions
        .iter()
        .filter(|action| matches!(action, FileAction::WriteFile { .. }))
        .count() as i32;
    let package_bonus = envelope
        .actions
        .iter()
        .filter(|action| matches!(action, FileAction::PackagePythonExe { .. }))
        .count() as i32
        * 4;
    let model_diversity_bonus =
        if result.model == QWEN_CODER_MODEL || result.model == DEEPSEEK_CODER_MODEL {
            2
        } else {
            0
        };

    100 + action_count * 10 + replace_bonus * 3 + package_bonus - write_count
        + model_diversity_bonus
}

fn coordinator_direct_fallback(user_request: &str, reason: &str) -> String {
    format!(
        "Ich habe verstanden, worum es geht: {}\n\n{}",
        user_request.trim(),
        reason
    )
}

fn parse_action_envelope(raw: &str) -> Result<ActionEnvelope, String> {
    let trimmed = raw.trim();

    if let Ok(envelope) = serde_json::from_str::<ActionEnvelope>(trimmed) {
        return Ok(envelope);
    }

    let slash_fixed = trimmed.replace('\\', "/");
    if slash_fixed != trimmed {
        if let Ok(envelope) = serde_json::from_str::<ActionEnvelope>(&slash_fixed) {
            return Ok(envelope);
        }
    }

    if let Some(json_text) = extract_first_json_object(trimmed) {
        let slash_fixed = json_text.replace('\\', "/");
        if let Ok(envelope) = serde_json::from_str::<ActionEnvelope>(&slash_fixed) {
            return Ok(envelope);
        }

        return serde_json::from_str::<ActionEnvelope>(&json_text).map_err(|e| {
            let preview = truncate_text(&json_text, 600);
            format!("JSON parser error: {}\nJSON start:\n{}", e, preview)
        });
    }

    Err("no JSON block found".to_string())
}

fn extract_first_json_object(text: &str) -> Option<String> {
    let mut start = None;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (idx, ch) in text.char_indices() {
        if start.is_none() {
            if ch == '{' {
                start = Some(idx);
                depth = 1;
            }
            continue;
        }

        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let start_idx = start.unwrap_or(0);
                    let end_idx = idx + ch.len_utf8();
                    return Some(text[start_idx..end_idx].to_string());
                }
            }
            _ => {}
        }
    }

    None
}

fn canonical_project_root(project_dir: &str) -> Result<PathBuf, String> {
    let trimmed = project_dir.trim().trim_matches('"');
    if trimmed.is_empty() {
        return Err("The project folder is empty.".to_string());
    }

    let root = PathBuf::from(trimmed);
    let canonical =
        fs::canonicalize(&root).map_err(|e| format!("Path could not be read: {}", e))?;

    if !canonical.is_dir() {
        return Err(format!("{} is not a folder.", canonical.display()));
    }

    Ok(canonical)
}

fn safe_project_path(root: &Path, rel_path: &str) -> Result<PathBuf, String> {
    let root = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let cleaned = rel_path
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .replace('\\', "/")
        .replace('\t', "/t")
        .replace('\r', "")
        .replace('\n', "");

    let rel_path = cleaned.trim();
    if rel_path.is_empty() {
        return Err("Empty file path is not allowed.".to_string());
    }

    let rel = Path::new(rel_path);
    if rel.is_absolute() {
        let candidate = PathBuf::from(rel);
        let mut check_parent = candidate.parent().unwrap_or(&root).to_path_buf();
        while !check_parent.exists() {
            if !check_parent.pop() {
                check_parent = root.clone();
                break;
            }
        }

        let check_parent = fs::canonicalize(&check_parent)
            .map_err(|e| format!("Target path could not be checked: {}", e))?;

        if check_parent.starts_with(&root) {
            return Ok(candidate);
        }

        return Err(format!(
            "Absolute path is outside the project folder: {}",
            rel_path
        ));
    }

    for component in rel.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => {
                return Err(format!("Paths with '..' are blocked: {}", rel_path));
            }
            Component::Prefix(_) | Component::RootDir => {
                return Err(format!("Non-relative path is blocked: {}", rel_path));
            }
        }
    }

    let candidate = root.join(rel);
    let mut check_parent = candidate.parent().unwrap_or(&root).to_path_buf();
    while !check_parent.exists() {
        if !check_parent.pop() {
            check_parent = root.clone();
            break;
        }
    }

    let check_parent = fs::canonicalize(&check_parent)
        .map_err(|e| format!("Target path could not be checked: {}", e))?;

    if !check_parent.starts_with(&root) {
        return Err(format!(
            "Target is outside the project folder: {}",
            rel_path
        ));
    }

    Ok(candidate)
}

fn validate_action_envelope(root: &Path, envelope: &ActionEnvelope) -> Vec<String> {
    let mut errors = Vec::new();
    let mut available_after_previous_actions: Vec<PathBuf> = Vec::new();

    for (idx, action) in envelope.actions.iter().enumerate() {
        let action_no = idx + 1;

        match action {
            FileAction::WriteFile { path, .. } => match safe_project_path(root, path) {
                Ok(target) => available_after_previous_actions.push(target),
                Err(e) => errors.push(format!("Action {} write_file: {}", action_no, e)),
            },
            FileAction::CopyFile { source, path } => {
                let source_path = PathBuf::from(source.trim().trim_matches('"').trim_matches('\''));
                if !source_path.exists() {
                    errors.push(format!(
                        "Action {} copy_file: source does not exist: {}",
                        action_no, source
                    ));
                    continue;
                }
                if !source_path.is_file() {
                    errors.push(format!(
                        "Action {} copy_file: source is not a file: {}",
                        action_no, source
                    ));
                    continue;
                }
                match safe_project_path(root, path) {
                    Ok(target) => available_after_previous_actions.push(target),
                    Err(e) => errors.push(format!("Action {} copy_file target: {}", action_no, e)),
                }
            }
            FileAction::AppendFile { path, .. } => {
                if let Err(e) = safe_project_path(root, path) {
                    errors.push(format!("Action {} append_file: {}", action_no, e));
                }
            }
            FileAction::PackagePythonExe { path } => {
                let target = match safe_project_path(root, path) {
                    Ok(path) => path,
                    Err(e) => {
                        errors.push(format!("Action {} package_python_exe: {}", action_no, e));
                        continue;
                    }
                };

                if !target
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("py"))
                {
                    errors.push(format!(
                        "Action {} package_python_exe: {} is not a .py file",
                        action_no, path
                    ));
                    continue;
                }

                if !target.exists()
                    && !available_after_previous_actions
                        .iter()
                        .any(|p| p == &target)
                {
                    errors.push(format!(
                        "Action {} package_python_exe: {} does not exist and is not written beforehand",
                        action_no, path
                    ));
                }
            }
            FileAction::ReplaceText {
                path,
                find,
                replace: _,
            } => {
                if find.is_empty() {
                    errors.push(format!(
                        "Action {} replace_text: empty search text",
                        action_no
                    ));
                    continue;
                }

                let target = match safe_project_path(root, path) {
                    Ok(path) => path,
                    Err(e) => {
                        errors.push(format!("Action {} replace_text: {}", action_no, e));
                        continue;
                    }
                };

                let current = match fs::read_to_string(&target) {
                    Ok(text) => text,
                    Err(e) => {
                        errors.push(format!(
                            "Action {} replace_text: file not readable {}: {}",
                            action_no, path, e
                        ));
                        continue;
                    }
                };

                let matches = current.matches(find).count();
                if matches != 1 {
                    errors.push(format!(
                        "Action {} replace_text: {} matches in {} (expected: 1)",
                        action_no, matches, path
                    ));
                }
            }
        }
    }

    errors
}

fn apply_file_actions(root: &Path, actions: &[FileAction]) -> ActionReport {
    let mut report = ActionReport {
        log_lines: Vec::new(),
        changed_files: Vec::new(),
        had_error: false,
    };

    for action in actions {
        match action {
            FileAction::WriteFile { path, content } => {
                let target = match safe_project_path(root, path) {
                    Ok(path) => path,
                    Err(e) => {
                        report.had_error = true;
                        report
                            .log_lines
                            .push(format!("write_file blocked: {}", e));
                        continue;
                    }
                };

                if let Some(parent) = target.parent() {
                    if let Err(e) = fs::create_dir_all(parent) {
                        report.had_error = true;
                        report.log_lines.push(format!(
                            "write_file failed (folder): {}: {}",
                            path, e
                        ));
                        continue;
                    }
                }

                match fs::write(&target, content) {
                    Ok(_) => {
                        report
                            .log_lines
                            .push(format!("File written: {}", path));
                        report.changed_files.push(path.clone());
                    }
                    Err(e) => {
                        report.had_error = true;
                        report
                            .log_lines
                            .push(format!("write_file failed: {}: {}", path, e));
                    }
                }
            }
            FileAction::CopyFile { source, path } => {
                let source_path = PathBuf::from(source.trim().trim_matches('"').trim_matches('\''));
                if !source_path.exists() || !source_path.is_file() {
                    report.had_error = true;
                    report.log_lines.push(format!(
                        "copy_file blocked: source is not available: {}",
                        source
                    ));
                    continue;
                }

                let target = match safe_project_path(root, path) {
                    Ok(path) => path,
                    Err(e) => {
                        report.had_error = true;
                        report
                            .log_lines
                            .push(format!("copy_file target blocked: {}", e));
                        continue;
                    }
                };

                if let Some(parent) = target.parent() {
                    if let Err(e) = fs::create_dir_all(parent) {
                        report.had_error = true;
                        report.log_lines.push(format!(
                            "copy_file failed creating folder for {}: {}",
                            path, e
                        ));
                        continue;
                    }
                }

                match fs::copy(&source_path, &target) {
                    Ok(_) => {
                        report.log_lines.push(format!(
                            "File copied: {} -> {}",
                            source_path.display(),
                            path
                        ));
                        report.changed_files.push(path.clone());
                    }
                    Err(e) => {
                        report.had_error = true;
                        report.log_lines.push(format!(
                            "copy_file failed: {} -> {}: {}",
                            source_path.display(),
                            path,
                            e
                        ));
                    }
                }
            }
            FileAction::PackagePythonExe { path } => {
                let result = run_local_python_pack_task(path, None, root, Vec::new());
                report.log_lines.extend(result.log_lines);

                let exe_name = Path::new(path)
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("app")
                    .to_string();
                let exe_rel = format!("dist/{}.exe", exe_name);

                if root.join(&exe_rel).exists() {
                    report
                        .log_lines
                        .push(format!("Python-EXE gepackt: {}", exe_rel));
                    report.changed_files.push(exe_rel);
                } else {
                    report.had_error = true;
                    report.log_lines.push(format!(
                        "package_python_exe failed: {}",
                        truncate_text(&result.final_answer, 500)
                    ));
                }
            }
            FileAction::ReplaceText {
                path,
                find,
                replace,
            } => {
                if find.is_empty() {
                    report.had_error = true;
                    report.log_lines.push(format!(
                        "replace_text blocked: empty search text in {}",
                        path
                    ));
                    continue;
                }

                let target = match safe_project_path(root, path) {
                    Ok(path) => path,
                    Err(e) => {
                        report.had_error = true;
                        report
                            .log_lines
                            .push(format!("replace_text blocked: {}", e));
                        continue;
                    }
                };

                let current = match fs::read_to_string(&target) {
                    Ok(text) => text,
                    Err(e) => {
                        report.had_error = true;
                        report.log_lines.push(format!(
                            "replace_text could not read file: {}: {}",
                            path, e
                        ));
                        continue;
                    }
                };

                let matches = current.matches(find).count();
                if matches != 1 {
                    report.had_error = true;
                    report.log_lines.push(format!(
                        "replace_text blocked: {} matches in {} (expected: 1)",
                        matches, path
                    ));
                    continue;
                }

                let updated = current.replacen(find, replace, 1);
                match fs::write(&target, updated) {
                    Ok(_) => {
                        report.log_lines.push(format!("Text replaced: {}", path));
                        report.changed_files.push(path.clone());
                    }
                    Err(e) => {
                        report.had_error = true;
                        report
                            .log_lines
                            .push(format!("replace_text failed: {}: {}", path, e));
                    }
                }
            }
            FileAction::AppendFile { path, content } => {
                let target = match safe_project_path(root, path) {
                    Ok(path) => path,
                    Err(e) => {
                        report.had_error = true;
                        report
                            .log_lines
                            .push(format!("append_file blocked: {}", e));
                        continue;
                    }
                };

                if let Some(parent) = target.parent() {
                    if let Err(e) = fs::create_dir_all(parent) {
                        report.had_error = true;
                        report.log_lines.push(format!(
                            "append_file failed (folder): {}: {}",
                            path, e
                        ));
                        continue;
                    }
                }

                let mut file = match fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&target)
                {
                    Ok(file) => file,
                    Err(e) => {
                        report.had_error = true;
                        report
                            .log_lines
                            .push(format!("append_file failed: {}: {}", path, e));
                        continue;
                    }
                };

                match file.write_all(content.as_bytes()) {
                    Ok(_) => {
                        report.log_lines.push(format!("Text appended: {}", path));
                        report.changed_files.push(path.clone());
                    }
                    Err(e) => {
                        report.had_error = true;
                        report.log_lines.push(format!(
                            "append_file write failed: {}: {}",
                            path, e
                        ));
                    }
                }
            }
        }
    }

    report.changed_files.sort();
    report.changed_files.dedup();
    report
}

fn fallback_final_answer(
    summary: Option<&str>,
    changed_files: &[String],
    action_count: usize,
    had_error: bool,
    auto_apply: bool,
    test_output: &str,
) -> String {
    let mut out = String::new();

    if let Some(summary) = summary.filter(|s| !s.trim().is_empty()) {
        out.push_str(summary.trim());
        out.push_str("\n\n");
    }

    if action_count == 0 {
        out.push_str(
            "I understood the request. No file was changed in this run.",
        );
    } else if auto_apply {
        if changed_files.is_empty() {
            out.push_str("I checked, but no file was changed.");
        } else {
            out.push_str("Changed: ");
            out.push_str(&changed_files.join(", "));
            out.push('.');
        }
    } else {
        out.push_str("The file actions were created, but auto-apply is off.");
    }

    if had_error {
        out.push_str(
            "\n\nSome actions were blocked for safety or match-count reasons.",
        );
    }

    if !test_output.trim().is_empty() {
        out.push_str("\n\nTest/build was run.");
    }

    out
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (idx, ch) in text.chars().enumerate() {
        if idx >= max_chars {
            out.push_str("\n[shortened]");
            return out;
        }
        out.push(ch);
    }
    out
}

fn parse_usize_field(raw: &str, default: usize, min: usize, max: usize) -> usize {
    raw.trim()
        .parse::<usize>()
        .map(|value| value.clamp(min, max))
        .unwrap_or(default)
}

fn agent_model_for_index(index: usize, primary_model: &str) -> String {
    match index {
        0 => primary_model.to_string(),
        1 => LOCAL_GPT_ROUTER_MODEL.to_string(),
        2 => OPENCLAW_AGENT_MODEL.to_string(),
        3 => QWEN_CODER_MODEL.to_string(),
        4 => DEEPSEEK_CODER_MODEL.to_string(),
        5 => STARCODER_MODEL.to_string(),
        _ => DEEPSEEK_LARGE_CODER_MODEL.to_string(),
    }
}

/// Returns all models required by the agents.
fn agent_models_for_count<'a>(primary_model: &'a str, count: usize) -> Vec<&'a str> {
    let mut models = vec![primary_model];
    for idx in 0..count {
        match idx {
            0 => models.push(primary_model),
            1 => models.push(LOCAL_GPT_ROUTER_MODEL),
            2 => {
                // local Modelfile model; do not pull here
            }
            3 => models.push(QWEN_CODER_MODEL),
            4 => models.push(DEEPSEEK_CODER_MODEL),
            5 => models.push(STARCODER_MODEL),
            _ => models.push(DEEPSEEK_LARGE_CODER_MODEL),
        }
    }
    models.sort();
    models.dedup();
    models
}

fn recommended_agent_count(config: &AgentRunConfig, decision: &CoordinatorDecision) -> usize {
    if decision.needs_code {
        return config.max_parallel_agents.clamp(1, 6);
    }

    let requested = decision.recommended_agent_count.clamp(1, 6);
    requested.min(config.max_parallel_agents.max(1))
}

fn ollama_generate_with_fallback(
    ollama_path: &str,
    model: &str,
    fallback_model: &str,
    prompt: &str,
) -> Result<String, String> {
    ollama_generate(ollama_path, model, prompt).or_else(|first_error| {
        if model == fallback_model {
            Err(first_error)
        } else {
            ollama_generate(ollama_path, fallback_model, prompt).map_err(|fallback_error| {
                format!(
                    "{}; fallback {} also failed: {}",
                    first_error, fallback_model, fallback_error
                )
            })
        }
    })
}

fn ensure_ollama_available(ollama_path: &str, project_root: &Path) -> Result<(), String> {
    if ollama_is_reachable() {
        return Ok(());
    }

    let trimmed = ollama_path.trim().trim_matches('"');
    if trimmed.is_empty() || trimmed == "Not found" || trimmed == "Nicht gefunden" {
        return Err("Ollama path is not set.".to_string());
    }

    let path = PathBuf::from(trimmed);
    if !path.exists() {
        return Err(format!("Ollama was not found: {}", path.display()));
    }

    let models_dir = project_ollama_models_dir(project_root);
    fs::create_dir_all(&models_dir)
        .map_err(|e| format!("Model folder could not be created: {}", e))?;

    hidden_command(&path)
        .arg("serve")
        .env("OLLAMA_NUM_GPU", "0")
        .env("OLLAMA_MODELS", &models_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Ollama could not be started: {}", e))?;

    for _ in 0..40 {
        thread::sleep(Duration::from_millis(250));
        if ollama_is_reachable() {
            return Ok(());
        }
    }

    Err("Ollama was started, but the API is not responding yet.".to_string())
}

fn ensure_required_models(
    ollama_path: &str,
    project_root: &Path,
    models: &[&str],
) -> Result<(), String> {
    let available = ollama_list_models().unwrap_or_default();

    for model in models {
        if model.starts_with("openclaw-agent") {
            continue;
        }
        if available.iter().any(|name| name == model) {
            continue;
        }
        ollama_pull_model(ollama_path, project_root, model)?;
    }

    Ok(())
}

fn ensure_all_agent_models(
    ollama_path: &str,
    project_root: &Path,
    selected_model: &str,
) -> Result<(), String> {
    let mut models = ALL_DOWNLOADABLE_AGENT_MODELS.to_vec();
    if !models.contains(&selected_model) && !selected_model.starts_with("openclaw-agent") {
        models.push(selected_model);
    }
    models.sort_unstable();
    models.dedup();

    ensure_required_models(ollama_path, project_root, &models)?;

    let available = ollama_list_models().unwrap_or_default();
    if !available.iter().any(|name| name == OPENCLAW_AGENT_MODEL) {
        create_openclaw_agent_model(ollama_path, project_root)?;
    }
    install_openclaw_application(ollama_path)?;
    Ok(())
}

fn install_openclaw_application(ollama_path: &str) -> Result<(), String> {
    let marker_dir = app_base_dir().join(".local_ai_builder");
    let marker = marker_dir.join("openclaw_install_started");
    if marker.exists() {
        return Ok(());
    }

    let path = PathBuf::from(ollama_path.trim().trim_matches('"'));
    if !path.exists() {
        return Err(format!("Ollama was not found: {}", path.display()));
    }
    fs::create_dir_all(&marker_dir)
        .map_err(|e| format!("OpenClaw installation folder could not be created: {}", e))?;

    hidden_command(&path)
        .arg("launch")
        .arg("openclaw")
        .arg("--model")
        .arg(LOCAL_GPT_ROUTER_MODEL)
        .arg("--yes")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("OpenClaw download/installation could not start: {}", e))?;

    fs::write(marker, format!("started={}\n", now_secs()))
        .map_err(|e| format!("OpenClaw installation marker could not be written: {}", e))?;
    Ok(())
}

fn create_openclaw_agent_model(ollama_path: &str, project_root: &Path) -> Result<(), String> {
    let trimmed = ollama_path.trim().trim_matches('"');
    let path = PathBuf::from(trimmed);
    if !path.exists() {
        return Err(format!("Ollama was not found: {}", path.display()));
    }

    let config_dir = app_base_dir().join(".local_ai_builder");
    fs::create_dir_all(&config_dir)
        .map_err(|e| format!("OpenClaw model folder could not be created: {}", e))?;
    let modelfile = config_dir.join("OpenClaw.Modelfile");
    fs::write(
        &modelfile,
        concat!(
            "FROM gpt-oss:20b\n",
            "SYSTEM You are OpenClaw, a local orchestration and coding agent. Split complex work into safe steps, produce precise implementation plans, and return only the output format requested by the caller.\n",
            "PARAMETER temperature 0.2\n"
        ),
    )
    .map_err(|e| format!("OpenClaw Modelfile could not be written: {}", e))?;

    let models_dir = project_ollama_models_dir(project_root);
    let status = hidden_command(&path)
        .arg("create")
        .arg(OPENCLAW_AGENT_MODEL)
        .arg("-f")
        .arg(&modelfile)
        .env("OLLAMA_MODELS", &models_dir)
        .stdin(Stdio::null())
        .status()
        .map_err(|e| format!("OpenClaw model installation could not start: {}", e))?;

    if status.success() {
        Ok(())
    } else {
        Err("OpenClaw model installation failed.".to_string())
    }
}

fn project_ollama_models_dir(_project_root: &Path) -> PathBuf {
    distribution_runtime_ollama_models_dir()
}

fn distribution_runtime_ollama_models_dir() -> PathBuf {
    let exe_dir = app_base_dir();
    let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let mut candidates = vec![exe_dir.join("runtime").join(".ollama").join("models")];
    candidates.push(
        exe_dir
            .join("DISTRIBUTION")
            .join("runtime")
            .join(".ollama")
            .join("models"),
    );
    candidates.push(
        current_dir
            .join("DISTRIBUTION")
            .join("runtime")
            .join(".ollama")
            .join("models"),
    );
    candidates.push(current_dir.join("runtime").join(".ollama").join("models"));

    candidates
        .iter()
        .find(|path| path.exists())
        .cloned()
        .unwrap_or_else(|| exe_dir.join("runtime").join(".ollama").join("models"))
}

fn ollama_pull_model(ollama_path: &str, project_root: &Path, model: &str) -> Result<(), String> {
    let trimmed = ollama_path.trim().trim_matches('"');
    if trimmed.is_empty() || trimmed == "Not found" || trimmed == "Nicht gefunden" {
        return Err(format!(
            "Model {} is missing, but the Ollama path is not set.",
            model
        ));
    }

    let path = PathBuf::from(trimmed);
    if !path.exists() {
        return Err(format!(
            "Model {} is missing, but Ollama was not found: {}",
            model,
            path.display()
        ));
    }

    let models_dir = project_ollama_models_dir(project_root);
    fs::create_dir_all(&models_dir)
        .map_err(|e| format!("Model folder could not be created: {}", e))?;

    let status = hidden_command(&path)
        .arg("pull")
        .arg(model)
        .env("OLLAMA_MODELS", &models_dir)
        .stdin(Stdio::null())
        .status()
        .map_err(|e| format!("ollama pull {} could not be started: {}", model, e))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("ollama pull {} failed.", model))
    }
}

fn ollama_list_models() -> Result<Vec<String>, String> {
    let request = "GET /api/tags HTTP/1.1\r\n\
         Host: 127.0.0.1:11434\r\n\
         Accept: application/json\r\n\
         Connection: close\r\n\
         \r\n";

    let mut stream = TcpStream::connect("127.0.0.1:11434")
        .map_err(|e| format!("Ollama API not reachable: {}", e))?;

    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| e.to_string())?;

    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    let mut bytes = Vec::new();
    stream
        .read_to_end(&mut bytes)
        .map_err(|e| format!("HTTP response failed: {}", e))?;

    let body_text = parse_http_body(bytes)?;
    let parsed: OllamaTagsResponse = serde_json::from_str(&body_text)
        .map_err(|e| format!("Ollama tags could not be read: {}", e))?;

    Ok(parsed.models.into_iter().map(|model| model.name).collect())
}

fn ollama_is_reachable() -> bool {
    TcpStream::connect("127.0.0.1:11434").is_ok()
}

fn ollama_generate(_ollama_path: &str, model: &str, prompt: &str) -> Result<String, String> {
    let body = json!({
        "model": model,
        "prompt": prompt,
        "stream": false
    })
    .to_string();

    let request = format!(
        "POST /api/generate HTTP/1.1\r\n\
         Host: 127.0.0.1:11434\r\n\
         Content-Type: application/json\r\n\
         Accept: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {}",
        body.as_bytes().len(),
        body
    );

    let mut stream = TcpStream::connect("127.0.0.1:11434")
        .map_err(|e| format!("Ollama API not reachable: {}", e))?;

    stream
        .set_read_timeout(Some(Duration::from_secs(240)))
        .map_err(|e| e.to_string())?;

    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| e.to_string())?;

    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    let mut bytes = Vec::new();
    stream
        .read_to_end(&mut bytes)
        .map_err(|e| format!("HTTP response failed: {}", e))?;

    let header_end = find_header_end(&bytes).ok_or_else(|| {
        "HTTP response could not be read: no header end found".to_string()
    })?;

    let header = String::from_utf8_lossy(&bytes[..header_end]).to_string();
    let mut body_bytes = bytes[header_end + 4..].to_vec();

    if !header.starts_with("HTTP/1.1 200") && !header.starts_with("HTTP/1.0 200") {
        let text = String::from_utf8_lossy(&body_bytes).to_string();
        return Err(format!("Ollama HTTP error:\n{}\n{}", header, text));
    }

    let header_lc = header.to_lowercase();
    if header_lc.contains("transfer-encoding: chunked") {
        body_bytes = decode_chunked_body(&body_bytes)?;
    }

    let body_text = String::from_utf8_lossy(&body_bytes).trim().to_string();

    if body_text.is_empty() {
        return Err("Ollama returned an empty body.".to_string());
    }

    let parsed: OllamaGenerateResponse = serde_json::from_str(&body_text).map_err(|e| {
        let preview: String = body_text.chars().take(600).collect();
        format!("JSON parser error: {}\nresponse start:\n{}", e, preview)
    })?;

    if let Some(err) = parsed.error {
        return Err(err);
    }

    Ok(parsed.response.unwrap_or_default())
}

fn parse_http_body(bytes: Vec<u8>) -> Result<String, String> {
    let header_end = find_header_end(&bytes).ok_or_else(|| {
        "HTTP response could not be read: no header end found".to_string()
    })?;

    let header = String::from_utf8_lossy(&bytes[..header_end]).to_string();
    let mut body_bytes = bytes[header_end + 4..].to_vec();

    if !header.starts_with("HTTP/1.1 200") && !header.starts_with("HTTP/1.0 200") {
        let text = String::from_utf8_lossy(&body_bytes).to_string();
        return Err(format!("Ollama HTTP error:\n{}\n{}", header, text));
    }

    let header_lc = header.to_lowercase();
    if header_lc.contains("transfer-encoding: chunked") {
        body_bytes = decode_chunked_body(&body_bytes)?;
    }

    let body_text = String::from_utf8_lossy(&body_bytes).trim().to_string();
    if body_text.is_empty() {
        return Err("Ollama returned an empty body.".to_string());
    }

    Ok(body_text)
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|w| w == b"\r\n\r\n")
}

fn decode_chunked_body(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut pos = 0usize;

    loop {
        let line_end = find_crlf(bytes, pos)
            .ok_or_else(|| "Chunked HTTP response is defective: chunk size not found".to_string())?;

        let size_line = String::from_utf8_lossy(&bytes[pos..line_end]).to_string();
        let size_hex = size_line.split(';').next().unwrap_or("").trim();

        let size = usize::from_str_radix(size_hex, 16).map_err(|e| {
            format!(
                "Chunked HTTP response is defective: invalid chunk size '{}': {}",
                size_hex, e
            )
        })?;

        pos = line_end + 2;

        if size == 0 {
            break;
        }

        if pos + size > bytes.len() {
            return Err("Chunked HTTP response is defective: chunk longer than response".to_string());
        }

        out.extend_from_slice(&bytes[pos..pos + size]);
        pos += size;

        if pos + 2 <= bytes.len() && &bytes[pos..pos + 2] == b"\r\n" {
            pos += 2;
        }
    }

    Ok(out)
}

fn find_crlf(bytes: &[u8], start: usize) -> Option<usize> {
    bytes[start..]
        .windows(2)
        .position(|w| w == b"\r\n")
        .map(|p| start + p)
}

fn clean_visible_answer(raw: &str) -> String {
    let mut s = raw.trim().to_string();

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
        if let Some(resp) = v.get("response").and_then(|x| x.as_str()) {
            s = resp.trim().to_string();
        }
    }

    if s.contains("\"model\"") && s.contains("\"created_at\"") && s.contains("\"response\"") {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            if let Some(resp) = v.get("response").and_then(|x| x.as_str()) {
                s = resp.trim().to_string();
            }
        }
    }

    if s.trim_start().starts_with("{\"actions\"")
        || s.contains("\"actions\"")
        || s.contains("\"op\":\"write\"")
        || s.contains("\"op\":\"delete\"")
    {
        return "I understood your request. I hide internal file actions in the UI and answer normally.".to_string();
    }

    s.replace("\r\n", "\n").trim().to_string()
}

fn render_chat_message(ui: &mut egui::Ui, text: &str) {
    let mut in_code = false;
    let mut code_language = String::new();
    let mut block = String::new();

    for line in text.lines() {
        if let Some(language) = line.trim().strip_prefix("```") {
            if in_code {
                render_code_block(ui, &code_language, block.trim_end());
                block.clear();
                code_language.clear();
            } else {
                code_language = language.trim().to_string();
            }
            in_code = !in_code;
            continue;
        }

        if in_code {
            block.push_str(line);
            block.push('\n');
        } else if !line.is_empty() {
            ui.add(egui::Label::new(line).selectable(true));
        } else {
            ui.add_space(4.0);
        }
    }

    if in_code && !block.is_empty() {
        render_code_block(ui, &code_language, block.trim_end());
    }
}

fn render_code_block(ui: &mut egui::Ui, language: &str, code: &str) {
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(32, 34, 37))
        .rounding(egui::Rounding::same(6.0))
        .inner_margin(egui::Margin::same(10.0))
        .show(ui, |ui| {
            if !language.is_empty() {
                ui.label(
                    egui::RichText::new(language)
                        .small()
                        .color(egui::Color32::from_rgb(170, 175, 185)),
                );
            }
            ui.add(
                egui::Label::new(
                    egui::RichText::new(code)
                        .monospace()
                        .color(egui::Color32::from_rgb(235, 235, 235)),
                )
                .selectable(true)
                .wrap(),
            );
        });
}

fn read_project_snapshot(project_dir: &str, max_chars: usize) -> String {
    let root = PathBuf::from(project_dir);
    let mut out = String::new();
    let mut files = Vec::new();

    for entry in WalkDir::new(&root)
        .max_depth(5)
        .into_iter()
        .filter_entry(|entry| !is_ignored_snapshot_entry(entry.path()))
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path().to_path_buf();
        if is_text_snapshot_file(&path) {
            files.push(path);
        }
    }

    files.sort();

    for path in files {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");

        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };

        if metadata.len() > 80_000 {
            continue;
        }

        if let Ok(txt) = fs::read_to_string(&path) {
            out.push_str("\n--- File: ");
            out.push_str(&rel);
            out.push_str(" ---\n");
            out.push_str(&txt);
            out.push('\n');

            if out.len() >= max_chars {
                out.truncate(max_chars);
                out.push_str("\n[Snapshot shortened]\n");
                break;
            }
        }
    }

    if out.trim().is_empty() {
        out.push_str("[No readable project snapshot found]");
    }

    out
}

fn is_ignored_snapshot_entry(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };

    matches!(
        name,
        ".git"
            | "target"
            | "DISTRIBUTION"
            | "runtime"
            | "PROGRAMME"
            | ".ollama"
            | "node_modules"
            | ".venv"
            | "venv"
            | "dist"
            | "build"
            | ".next"
            | ".cache"
    )
}

fn is_text_snapshot_file(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return matches!(
            path.file_name().and_then(|name| name.to_str()),
            Some("Dockerfile" | "Makefile" | "README" | "LICENSE")
        );
    };

    matches!(
        ext.to_ascii_lowercase().as_str(),
        "rs" | "toml"
            | "lock"
            | "md"
            | "txt"
            | "json"
            | "jsonl"
            | "yaml"
            | "yml"
            | "js"
            | "jsx"
            | "ts"
            | "tsx"
            | "mjs"
            | "cjs"
            | "css"
            | "scss"
            | "html"
            | "htm"
            | "py"
            | "go"
            | "java"
            | "kt"
            | "cs"
            | "cpp"
            | "c"
            | "h"
            | "hpp"
            | "sh"
            | "ps1"
            | "bat"
            | "cmd"
            | "sql"
            | "xml"
            | "vue"
            | "svelte"
            | "php"
            | "rb"
    )
}

impl AgentBlockKind {
    fn label(&self) -> &'static str {
        match self {
            AgentBlockKind::Coordinator => "Coordinator",
            AgentBlockKind::Manager => "Verwaltungsagent",
            AgentBlockKind::CodingAgent => "Coding-Agent",
            AgentBlockKind::PlanningAgent => "Planungs-Agent",
            AgentBlockKind::ReviewAgent => "Review-Agent",
            AgentBlockKind::Task => "Task",
            AgentBlockKind::Tool => "Tool",
        }
    }

    fn all() -> &'static [AgentBlockKind] {
        &[
            AgentBlockKind::Coordinator,
            AgentBlockKind::Manager,
            AgentBlockKind::CodingAgent,
            AgentBlockKind::PlanningAgent,
            AgentBlockKind::ReviewAgent,
            AgentBlockKind::Task,
            AgentBlockKind::Tool,
        ]
    }
}

fn default_agent_graph() -> AgentGraph {
    let blocks = vec![
        AgentBlock {
            id: 1,
            title: "Coordinator".to_string(),
            kind: AgentBlockKind::Coordinator,
            model: DEFAULT_AGENT_MODEL.to_string(),
            prompt: "Talk to the user, split requests, and delegate to managers or agents.".to_string(),
            task: "Steuert den gesamten Auftrag.".to_string(),
            x: 380.0,
            y: 210.0,
            w: 210.0,
            h: 112.0,
        },
        AgentBlock {
            id: 2,
            title: "Unterverwalter Code".to_string(),
            kind: AgentBlockKind::Manager,
            model: DEFAULT_AGENT_MODEL.to_string(),
            prompt: "Route coding tasks to specialized coding agents and compare their proposals.".to_string(),
            task: "Coordinates coding agents.".to_string(),
            x: 640.0,
            y: 205.0,
            w: 220.0,
            h: 118.0,
        },
        AgentBlock {
            id: 3,
            title: "CodeLlama Coder".to_string(),
            kind: AgentBlockKind::CodingAgent,
            model: DEFAULT_AGENT_MODEL.to_string(),
            prompt: "Create safe file actions and respect relative paths.".to_string(),
            task: "Haupt-Coding-Agent.".to_string(),
            x: 900.0,
            y: 110.0,
            w: 210.0,
            h: 112.0,
        },
        AgentBlock {
            id: 4,
            title: "Qwen Coder".to_string(),
            kind: AgentBlockKind::CodingAgent,
            model: QWEN_CODER_MODEL.to_string(),
            prompt: "Create alternative coding proposals and pragmatically check small patches.".to_string(),
            task: "Alternative coding answer.".to_string(),
            x: 900.0,
            y: 300.0,
            w: 210.0,
            h: 112.0,
        },
        AgentBlock {
            id: 8,
            title: "DeepSeek Coder".to_string(),
            kind: AgentBlockKind::CodingAgent,
            model: DEEPSEEK_CODER_MODEL.to_string(),
            prompt: "Create another coding solution and pay attention to runnable programs."
                .to_string(),
            task: "Third coding answer.".to_string(),
            x: 1140.0,
            y: 205.0,
            w: 220.0,
            h: 112.0,
        },
        AgentBlock {
            id: 9,
            title: "GPT Coding Agent".to_string(),
            kind: AgentBlockKind::CodingAgent,
            model: LOCAL_GPT_ROUTER_MODEL.to_string(),
            prompt: "Create precise code changes and executable command solutions as safe file actions."
                .to_string(),
            task: "GPT coding solution and command expertise.".to_string(),
            x: 1140.0,
            y: 335.0,
            w: 220.0,
            h: 112.0,
        },
        AgentBlock {
            id: 5,
            title: "Planer".to_string(),
            kind: AgentBlockKind::PlanningAgent,
            model: DEFAULT_AGENT_MODEL.to_string(),
            prompt: "Plane robuste Umsetzungsschritte und Risiken.".to_string(),
            task: "Planung und Risiko.".to_string(),
            x: 120.0,
            y: 120.0,
            w: 200.0,
            h: 108.0,
        },
        AgentBlock {
            id: 6,
            title: "Task".to_string(),
            kind: AgentBlockKind::Task,
            model: String::new(),
            prompt: "User request.".to_string(),
            task: "Aktuelle Task startet hier.".to_string(),
            x: 125.0,
            y: 330.0,
            w: 200.0,
            h: 105.0,
        },
        AgentBlock {
            id: 7,
            title: "Reviewer".to_string(),
            kind: AgentBlockKind::ReviewAgent,
            model: DEFAULT_AGENT_MODEL.to_string(),
            prompt: "Check result, errors, and final answer.".to_string(),
            task: "Final review.".to_string(),
            x: 640.0,
            y: 410.0,
            w: 210.0,
            h: 112.0,
        },
    ];

    AgentGraph {
        blocks,
        connections: vec![
            AgentConnection {
                from: 6,
                to: 1,
                label: "Auftrag".to_string(),
            },
            AgentConnection {
                from: 1,
                to: 5,
                label: "Planung".to_string(),
            },
            AgentConnection {
                from: 1,
                to: 2,
                label: "delegiert".to_string(),
            },
            AgentConnection {
                from: 2,
                to: 3,
                label: "Coding".to_string(),
            },
            AgentConnection {
                from: 2,
                to: 4,
                label: "Alternative".to_string(),
            },
            AgentConnection {
                from: 2,
                to: 8,
                label: "DeepSeek".to_string(),
            },
            AgentConnection {
                from: 2,
                to: 9,
                label: "GPT Coding".to_string(),
            },
            AgentConnection {
                from: 2,
                to: 7,
                label: "Review".to_string(),
            },
            AgentConnection {
                from: 7,
                to: 1,
                label: "Ergebnis".to_string(),
            },
        ],
        next_id: 10,
    }
}

fn default_block_fields(
    kind: &AgentBlockKind,
    id: u64,
) -> (String, String, String, String, f32, f32) {
    match kind {
        AgentBlockKind::Coordinator => (
            format!("Coordinator {}", id),
            DEFAULT_AGENT_MODEL.to_string(),
            "Talk to the user and delegate work.".to_string(),
            "Verwaltet den Auftrag.".to_string(),
            220.0,
            118.0,
        ),
        AgentBlockKind::Manager => (
            format!("Verwalter {}", id),
            DEFAULT_AGENT_MODEL.to_string(),
            "Coordinate a subgroup of agents.".to_string(),
            "Unterverwaltung.".to_string(),
            220.0,
            118.0,
        ),
        AgentBlockKind::CodingAgent => (
            format!("Coder {}", id),
            match id % 3 {
                0 => DEEPSEEK_CODER_MODEL.to_string(),
                1 => DEFAULT_AGENT_MODEL.to_string(),
                _ => QWEN_CODER_MODEL.to_string(),
            },
            "Create safe file actions with relative paths.".to_string(),
            "Coding-Task.".to_string(),
            210.0,
            112.0,
        ),
        AgentBlockKind::PlanningAgent => (
            format!("Planer {}", id),
            DEFAULT_AGENT_MODEL.to_string(),
            "Plane Umsetzung, Risiken und Tests.".to_string(),
            "Planung.".to_string(),
            205.0,
            108.0,
        ),
        AgentBlockKind::ReviewAgent => (
            format!("Reviewer {}", id),
            DEFAULT_AGENT_MODEL.to_string(),
            "Check changes and summarize the result.".to_string(),
            "Review.".to_string(),
            205.0,
            108.0,
        ),
        AgentBlockKind::Task => (
            format!("Task {}", id),
            String::new(),
            "tasksbeschreibung.".to_string(),
            "Neue Task.".to_string(),
            200.0,
            105.0,
        ),
        AgentBlockKind::Tool => (
            format!("Tool {}", id),
            String::new(),
            "Local tool or file operation.".to_string(),
            "Tool.".to_string(),
            200.0,
            105.0,
        ),
    }
}

fn block_color(kind: &AgentBlockKind, selected: bool) -> egui::Color32 {
    let color = match kind {
        AgentBlockKind::Coordinator => egui::Color32::from_rgb(64, 92, 150),
        AgentBlockKind::Manager => egui::Color32::from_rgb(78, 112, 128),
        AgentBlockKind::CodingAgent => egui::Color32::from_rgb(84, 120, 88),
        AgentBlockKind::PlanningAgent => egui::Color32::from_rgb(125, 102, 68),
        AgentBlockKind::ReviewAgent => egui::Color32::from_rgb(120, 82, 96),
        AgentBlockKind::Task => egui::Color32::from_rgb(92, 92, 105),
        AgentBlockKind::Tool => egui::Color32::from_rgb(92, 78, 120),
    };

    if selected {
        color.gamma_multiply(1.25)
    } else {
        color
    }
}

fn graph_to_python_code(graph: &AgentGraph) -> String {
    let json = serde_json::to_string_pretty(graph).unwrap_or_else(|_| "{}".to_string());
    format!(
        r#"# Local Coding AI Agent Graph
# Edit either this code or the blocks. The JSON section is the source for re-import.

AGENT_GRAPH_JSON = r'''
{}
'''

def run_agent_graph(task):
    """Sketch of the agent structure. Local AI uses the same graph in the block view."""
    graph = AGENT_GRAPH_JSON
    return {{
        "task": task,
        "graph_json": graph,
    }}
"#,
        json
    )
}

fn graph_from_python_code(code: &str) -> Result<AgentGraph, String> {
    let marker = "AGENT_GRAPH_JSON = r'''";
    let start = code
        .find(marker)
        .ok_or_else(|| "AGENT_GRAPH_JSON was not found.".to_string())?
        + marker.len();
    let rest = &code[start..];
    let end = rest
        .find("'''")
        .ok_or_else(|| "End of the AGENT_GRAPH_JSON block was not found.".to_string())?;
    let json = rest[..end].trim();

    let mut graph: AgentGraph =
        serde_json::from_str(json).map_err(|e| format!("Graph JSON is invalid: {}", e))?;
    normalize_agent_graph(&mut graph);
    Ok(graph)
}

fn normalize_agent_graph(graph: &mut AgentGraph) {
    graph
        .blocks
        .retain(|block| block.w > 40.0 && block.h > 40.0);

    for block in &mut graph.blocks {
        if block.model == "qwen2.5-coder:0.5b" {
            block.model = QWEN_CODER_MODEL.to_string();
        }
    }

    let has_deepseek = graph
        .blocks
        .iter()
        .any(|block| block.model == DEEPSEEK_CODER_MODEL || block.title.contains("DeepSeek"));
    let looks_like_old_default = graph
        .blocks
        .iter()
        .any(|block| block.id == 2 && block.title == "Unterverwalter Code")
        && graph
            .blocks
            .iter()
            .any(|block| block.id == 3 && block.title == "CodeLlama Coder")
        && graph
            .blocks
            .iter()
            .any(|block| block.id == 4 && block.title == "Qwen Coder");

    if !has_deepseek && looks_like_old_default {
        let id = graph.next_id.max(8);
        graph.blocks.push(AgentBlock {
            id,
            title: "DeepSeek Coder".to_string(),
            kind: AgentBlockKind::CodingAgent,
            model: DEEPSEEK_CODER_MODEL.to_string(),
            prompt: "Create another coding solution and pay attention to runnable programs."
                .to_string(),
            task: "Third coding answer.".to_string(),
            x: 1140.0,
            y: 205.0,
            w: 220.0,
            h: 112.0,
        });

        if !graph
            .connections
            .iter()
            .any(|connection| connection.from == 2 && connection.to == id)
        {
            graph.connections.push(AgentConnection {
                from: 2,
                to: id,
                label: "DeepSeek".to_string(),
            });
        }
    }

    let has_gpt_coder = graph.blocks.iter().any(|block| {
        block.kind == AgentBlockKind::CodingAgent
            && (block.title == "GPT Coding Agent" || block.model == LOCAL_GPT_ROUTER_MODEL)
    });
    if !has_gpt_coder && looks_like_old_default {
        let id = graph
            .blocks
            .iter()
            .map(|block| block.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        graph.blocks.push(AgentBlock {
            id,
            title: "GPT Coding Agent".to_string(),
            kind: AgentBlockKind::CodingAgent,
            model: LOCAL_GPT_ROUTER_MODEL.to_string(),
            prompt: "Create precise code changes and executable command solutions as safe file actions."
                .to_string(),
            task: "GPT coding solution and command expertise.".to_string(),
            x: 1140.0,
            y: 335.0,
            w: 220.0,
            h: 112.0,
        });
        graph.connections.push(AgentConnection {
            from: 2,
            to: id,
            label: "GPT Coding".to_string(),
        });
    }

    let max_id = graph.blocks.iter().map(|block| block.id).max().unwrap_or(0);
    graph.connections.retain(|connection| {
        graph.blocks.iter().any(|block| block.id == connection.from)
            && graph.blocks.iter().any(|block| block.id == connection.to)
            && connection.from != connection.to
    });
    graph.next_id = graph.next_id.max(max_id + 1).max(1);
}

fn state_file_path(base: &Path) -> PathBuf {
    base.join(".local_ai_builder").join("state.json")
}

fn load_saved_state(path: &Path, default_project_dir: &Path) -> SavedState {
    if let Ok(text) = fs::read_to_string(path) {
        if let Ok(state) = serde_json::from_str::<SavedState>(&text) {
            return normalize_saved_state(state, default_project_dir);
        }
    }

    default_saved_state(default_project_dir)
}

fn normalize_saved_state(mut state: SavedState, default_project_dir: &Path) -> SavedState {
    if state.projects.is_empty() {
        state.projects.push(default_project(default_project_dir));
    }

    if !state
        .projects
        .iter()
        .any(|project| project.id == state.active_project_id)
    {
        state.active_project_id = state
            .projects
            .first()
            .map(|project| project.id.clone())
            .unwrap_or_else(|| "project-default".to_string());
    }

    if state.sessions.is_empty() {
        let project = state
            .projects
            .iter()
            .find(|project| project.id == state.active_project_id)
            .cloned()
            .unwrap_or_else(|| default_project(default_project_dir));

        state.sessions.push(default_session(&project));
    }

    if state.agent_graph.blocks.is_empty() {
        state.agent_graph = default_agent_graph();
    }
    normalize_agent_graph(&mut state.agent_graph);

    for session in &mut state.sessions {
        if session.messages.is_empty() {
            session.messages.push(welcome_message());
        }
    }

    state
        .sessions
        .sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

    if !state
        .sessions
        .iter()
        .any(|session| session.id == state.active_session_id)
    {
        state.active_session_id = state
            .sessions
            .first()
            .map(|session| session.id.clone())
            .unwrap_or_else(|| "chat-default".to_string());
    }

    state
}

fn default_saved_state(default_project_dir: &Path) -> SavedState {
    let project = default_project(default_project_dir);
    let session = default_session(&project);

    SavedState {
        active_project_id: project.id.clone(),
        active_session_id: session.id.clone(),
        projects: vec![project],
        sessions: vec![session],
        agent_graph: default_agent_graph(),
    }
}

fn default_project(default_project_dir: &Path) -> ProjectEntry {
    let path = fs::canonicalize(default_project_dir).unwrap_or_else(|_| default_project_dir.into());
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("Project")
        .to_string();

    ProjectEntry {
        id: "project-default".to_string(),
        name,
        path: path.display().to_string(),
    }
}

fn default_session(project: &ProjectEntry) -> ChatSession {
    ChatSession {
        id: "chat-default".to_string(),
        title: "New task".to_string(),
        project_id: project.id.clone(),
        project_dir: project.path.clone(),
        messages: vec![welcome_message()],
        updated_at: now_secs(),
    }
}

fn welcome_message() -> ChatMsg {
    ChatMsg {
        who: "Assistant".to_string(),
        text: "Ready. You are talking to the coordinator. It decides internally which planning, coding, and review agents are needed, compares their results, and answers normally.".to_string(),
    }
}

fn title_from_message(message: &str) -> String {
    let trimmed = message
        .trim()
        .lines()
        .next()
        .unwrap_or("New task")
        .trim();
    if trimmed.is_empty() {
        return "New task".to_string();
    }

    let mut title = String::new();
    for (idx, ch) in trimmed.chars().enumerate() {
        if idx >= 42 {
            title.push_str("...");
            break;
        }
        title.push(ch);
    }
    title
}

fn sanitize_project_folder_name(name: &str) -> String {
    let mut out = String::new();

    for ch in name.trim().chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | ' ') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }

    let out = out.trim().replace(' ', "_");
    if out.is_empty() {
        "Project".to_string()
    } else {
        out
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn hidden_command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    let mut command = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    command
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lokal_ai_test_{}_{}", name, now_millis()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn smoke_config(root: &Path, request: &str, last_file_path: Option<&str>) -> AgentRunConfig {
        AgentRunConfig {
            project_dir: root.display().to_string(),
            memory_dir: root.join("memory").display().to_string(),
            model: DEFAULT_AGENT_MODEL.to_string(),
            ollama_path: "Not found".to_string(),
            terminal_cmd: String::new(),
            user_request: request.to_string(),
            last_file_path: last_file_path.map(str::to_string),
            show_progress: true,
            auto_apply_actions: true,
            run_tests_after_apply: false,
            context_limit: 50_000,
            max_parallel_agents: 2,
        }
    }

    #[test]
    fn local_file_task_creates_requested_txt_file() {
        let root = temp_test_dir("create_txt");
        let task = parse_local_file_task("Please create the text.txt file", None).unwrap();
        let result = run_local_file_task(task, &root, Vec::new());

        assert!(root.join("text.txt").exists());
        assert!(result.final_answer.contains("text.txt"));
    }

    #[test]
    fn local_file_task_uses_named_content_before_file_name_clause() {
        let root = temp_test_dir("create_txt_with_content");
        let task = parse_local_file_task(
            "Please create a .txt file with the content test and the name test.txt",
            None,
        )
        .unwrap();
        let result = run_local_file_task(task, &root, Vec::new());

        assert!(result.final_answer.contains("test.txt"));
        assert_eq!(fs::read_to_string(root.join("test.txt")).unwrap(), "test");
    }

    #[test]
    fn run_agent_chain_smoke_creates_test_file_without_ollama() {
        let root = temp_test_dir("chain_smoke_create");
        let result = run_agent_chain(smoke_config(
            &root,
            "Please create a .txt file with the content test and the name test.txt",
            None,
        ));

        assert_eq!(fs::read_to_string(root.join("test.txt")).unwrap(), "test");
        assert!(result.final_answer.contains("test.txt"));
        assert!(result
            .log_lines
            .iter()
            .any(|line| line.contains("Local tool")));
        assert!(!result.log_lines.iter().any(|line| line.contains("Ollama")));
    }

    #[test]
    fn followup_write_uses_last_mentioned_file() {
        let root = temp_test_dir("followup_write");
        let task = parse_local_file_task(
            "The file does not contain test. Write test into it.",
            Some("test.txt"),
        )
        .unwrap();
        let result = run_local_file_task(task, &root, Vec::new());

        assert!(result.final_answer.contains("test.txt"));
        assert_eq!(fs::read_to_string(root.join("test.txt")).unwrap(), "test");
    }

    #[test]
    fn run_agent_chain_smoke_followup_writes_last_file_without_ollama() {
        let root = temp_test_dir("chain_smoke_followup");
        let result = run_agent_chain(smoke_config(
            &root,
            "The file does not contain test. Write test into it.",
            Some("test.txt"),
        ));

        assert_eq!(fs::read_to_string(root.join("test.txt")).unwrap(), "test");
        assert!(result.final_answer.contains("test.txt"));
        assert!(!result.log_lines.iter().any(|line| line.contains("Ollama")));
    }

    #[test]
    fn package_python_exe_action_can_follow_write_file() {
        let root = temp_test_dir("package_action_validation");
        let envelope = ActionEnvelope {
            summary: None,
            actions: vec![
                FileAction::WriteFile {
                    path: "sample_app.py".to_string(),
                    content: "print('OK')\n".to_string(),
                },
                FileAction::PackagePythonExe {
                    path: "sample_app.py".to_string(),
                },
            ],
        };

        let errors = validate_action_envelope(&root, &envelope);

        assert!(errors.is_empty(), "{:?}", errors);
    }

    #[test]
    fn package_python_exe_action_rejects_missing_script() {
        let root = temp_test_dir("package_action_missing");
        let envelope = ActionEnvelope {
            summary: None,
            actions: vec![FileAction::PackagePythonExe {
                path: "missing.py".to_string(),
            }],
        };

        let errors = validate_action_envelope(&root, &envelope);

        assert!(!errors.is_empty());
    }

    #[test]
    fn parse_action_envelope_supports_package_python_exe() {
        let raw = r#"{
            "summary":"pack",
            "actions":[
                {"op":"write_file","path":"app.py","content":"print('OK')\n"},
                {"op":"package_python_exe","path":"app.py"}
            ]
        }"#;

        let envelope = parse_action_envelope(raw).unwrap();

        assert_eq!(envelope.actions.len(), 2);
        assert!(matches!(
            envelope.actions[1],
            FileAction::PackagePythonExe { .. }
        ));
    }

    #[test]
    fn ollama_models_dir_is_inside_distribution_runtime() {
        let root = temp_test_dir("models_dir");

        assert_eq!(
            project_ollama_models_dir(&root),
            distribution_runtime_ollama_models_dir()
        );
    }

    #[test]
    fn saved_old_default_graph_is_migrated_to_current_coder_models() {
        let mut graph = AgentGraph {
            blocks: vec![
                AgentBlock {
                    id: 2,
                    title: "Unterverwalter Code".to_string(),
                    kind: AgentBlockKind::Manager,
                    model: DEFAULT_AGENT_MODEL.to_string(),
                    prompt: String::new(),
                    task: String::new(),
                    x: 0.0,
                    y: 0.0,
                    w: 220.0,
                    h: 118.0,
                },
                AgentBlock {
                    id: 3,
                    title: "CodeLlama Coder".to_string(),
                    kind: AgentBlockKind::CodingAgent,
                    model: DEFAULT_AGENT_MODEL.to_string(),
                    prompt: String::new(),
                    task: String::new(),
                    x: 0.0,
                    y: 0.0,
                    w: 210.0,
                    h: 112.0,
                },
                AgentBlock {
                    id: 4,
                    title: "Qwen Coder".to_string(),
                    kind: AgentBlockKind::CodingAgent,
                    model: "qwen2.5-coder:0.5b".to_string(),
                    prompt: String::new(),
                    task: String::new(),
                    x: 0.0,
                    y: 0.0,
                    w: 210.0,
                    h: 112.0,
                },
            ],
            connections: Vec::new(),
            next_id: 8,
        };

        normalize_agent_graph(&mut graph);

        assert!(graph
            .blocks
            .iter()
            .any(|block| block.model == QWEN_CODER_MODEL));
        assert!(graph
            .blocks
            .iter()
            .any(|block| block.model == DEEPSEEK_CODER_MODEL));
        assert!(graph
            .connections
            .iter()
            .any(|connection| connection.from == 2 && connection.label == "DeepSeek"));
    }

    #[test]
    fn last_file_is_found_from_recent_chat_messages() {
        let messages = vec![
            ChatMsg {
                who: "Assistant".to_string(),
                text: "Done. I created `alt.txt`.".to_string(),
            },
            ChatMsg {
                who: "Assistant".to_string(),
                text: "Done. I created `test.txt`.".to_string(),
            },
        ];

        assert_eq!(last_file_from_messages(&messages).unwrap(), "test.txt");
    }

    #[test]
    fn smalltalk_does_not_need_agents() {
        let answer = direct_chat_answer("Hallo").unwrap();
        assert!(answer.contains("ready") || answer.contains("Ready"));
    }

    #[test]
    fn absolute_path_inside_project_is_allowed_after_normalization() {
        let root = temp_test_dir("absolute_inside");
        let target = root.join("inside.txt");
        let action = FileAction::WriteFile {
            path: target.display().to_string(),
            content: "ok".to_string(),
        };

        let report = apply_file_actions(&root, &[action]);

        assert!(!report.had_error);
        assert_eq!(fs::read_to_string(target).unwrap(), "ok");
    }

    #[test]
    fn agent_manager_validation_rejects_absolute_path_outside_project() {
        let root = temp_test_dir("validation_outside");
        let outside = root
            .parent()
            .unwrap_or_else(|| Path::new("C:\\"))
            .join("outside.txt");
        let envelope = ActionEnvelope {
            summary: None,
            actions: vec![FileAction::WriteFile {
                path: outside.display().to_string(),
                content: "nope".to_string(),
            }],
        };

        let errors = validate_action_envelope(&root, &envelope);

        assert!(!errors.is_empty());
    }

    #[test]
    fn agent_manager_validation_requires_single_replace_match() {
        let root = temp_test_dir("validation_replace");
        fs::write(root.join("sample.txt"), "eins\neins\n").unwrap();
        let envelope = ActionEnvelope {
            summary: None,
            actions: vec![FileAction::ReplaceText {
                path: "sample.txt".to_string(),
                find: "eins".to_string(),
                replace: "zwei".to_string(),
            }],
        };

        let errors = validate_action_envelope(&root, &envelope);

        assert!(!errors.is_empty());
    }

    #[test]
    fn bad_windows_backslashes_in_action_json_are_repaired() {
        let raw = r#"{
            "summary":"x",
            "actions":[{"op":"write_file","path":"C:\Users\tarek\Desktop\file.txt","content":"ok"}]
        }"#;

        let envelope = parse_action_envelope(raw).unwrap();

        assert_eq!(envelope.actions.len(), 1);
    }
}

fn run_shell(dir: &str, cmd: &str) -> String {
    let mut command = if cfg!(target_os = "windows") {
        let mut command = hidden_command("cmd");
        command.arg("/C").arg(cmd);
        command
    } else {
        let mut command = hidden_command("sh");
        command.arg("-c").arg(cmd);
        command
    };

    command.current_dir(dir);

    match run_command_with_timeout(command, Duration::from_secs(180)) {
        Ok(o) => {
            let mut s = String::new();
            s.push_str(&String::from_utf8_lossy(&o.stdout));
            s.push_str(&String::from_utf8_lossy(&o.stderr));

            if s.trim().is_empty() {
                format!("Command finished. Exit code: {:?}", o.status.code())
            } else {
                s
            }
        }
        Err(e) => format!("Error starting command: {}", e),
    }
}

fn app_base_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn distribution_data_root(base: &Path) -> PathBuf {
    if base.join("runtime").exists()
        || base.join("Local Coding AI.exe").exists()
        || base.join("Lokal_AI.exe").exists()
    {
        return base.to_path_buf();
    }

    let nested = base.join("DISTRIBUTION");
    if nested.exists() {
        return nested;
    }

    std::env::current_dir()
        .map(|dir| dir.join("DISTRIBUTION"))
        .unwrap_or_else(|_| base.join("DISTRIBUTION"))
}

fn ensure_distribution_data_dirs(root: &Path) {
    let dirs = [
        root.join("memory"),
        root.join("memory").join("conversations"),
        root.join("memory").join("agent_traffic"),
        root.join("memory").join("summaries"),
        root.join("memory").join("projects"),
        root.join("training"),
        root.join("training").join("raw"),
        root.join("training").join("errors"),
        root.join("training").join("success"),
        root.join("training").join("fine_tuning"),
        root.join("training").join("exports"),
        root.join("training").join("datasets"),
    ];

    for dir in dirs {
        let _ = fs::create_dir_all(dir);
    }

    let readme = root.join("training").join("README_TRAINING_DATA.txt");
    if !readme.exists() {
        let text = concat!(
            "Local AI training and memory data\n\n",
            "raw/all_runs.jsonl: every completed run with final answer and logs.\n",
            "errors/runs.jsonl: failed runs with chat history and diagnostic logs for later correction.\n",
            "success/runs.jsonl: runs without failed/error markers.\n",
            "fine_tuning/chat_messages.jsonl: successful chats in role/content message format.\n",
            "The event-driven training collector writes after every completed chat run.\n",
            "Failed answers are intentionally excluded from automatic fine-tuning data.\n",
            "fine_tuning/: place curated training files here before actual fine-tuning.\n",
            "datasets/: exported or cleaned datasets can be written here.\n",
            "memory/: persistent memory files read by the coordinator.\n"
        );
        append_text_file(&readme, text);
    }
}

fn append_text_file(path: &Path, text: &str) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(text.as_bytes());
    }
}

fn is_actual_error_log_line(line: &str) -> bool {
    if line.contains('\n') || line.starts_with("TRAFFIC ") {
        return false;
    }
    let lower = line.to_ascii_lowercase();
    lower.starts_with("ollama error:")
        || lower.contains(" failed:")
        || lower.contains(" failed.")
        || lower.contains("could not")
        || lower.contains("fehlgeschlagen")
        || lower.contains("blocked:")
        || lower.contains("timeout")
        || lower.contains("keine sicheren actions")
}

fn read_memory_context(memory_dir: &str, max_chars: usize) -> String {
    let root = PathBuf::from(memory_dir);
    let candidates = [
        root.join("global.md"),
        root.join("summaries").join("global.md"),
        root.join("projects").join("current.md"),
    ];

    let mut out = String::new();
    for path in candidates {
        if let Ok(text) = fs::read_to_string(&path) {
            out.push_str("\n--- Memory: ");
            out.push_str(&path.display().to_string());
            out.push_str(" ---\n");
            out.push_str(&text);
            out.push('\n');
        }
        if out.chars().count() >= max_chars {
            return truncate_text(&out, max_chars);
        }
    }

    truncate_text(&out, max_chars)
}

fn find_ollama(base: &Path) -> Option<PathBuf> {
    let candidates = [
        base.join("runtime").join("Ollama").join("ollama.exe"),
        base.join("DISTRIBUTION")
            .join("runtime")
            .join("Ollama")
            .join("ollama.exe"),
        base.join("PROGRAMME").join("Ollama").join("ollama.exe"),
        PathBuf::from(r"C:\Users\tarek\AppData\Local\Programs\Ollama\ollama.exe"),
    ];

    for c in candidates {
        if c.exists() {
            return Some(c);
        }
    }

    None
}

fn now_hhmmss() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        % 86400;

    let h = (secs / 3600 + 2) % 24;
    let m = (secs % 3600) / 60;
    let s = secs % 60;

    format!("{:02}:{:02}:{:02}", h, m, s)
}

fn load_app_icon() -> egui::IconData {
    eframe::icon_data::from_png_bytes(include_bytes!("../assets/app_icon256.png"))
        .expect("assets/app_icon256.png is not a valid PNG")
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Local Coding AI")
            .with_icon(load_app_icon()),
        ..Default::default()
    };

    eframe::run_native(
        "Local Coding AI",
        options,
        Box::new(|_cc| Ok(Box::new(LocalAiApp::default()))),
    )
}


