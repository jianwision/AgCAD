use crate::{
    document::{Scene, Source},
    materials::{self, Finish, Overrides},
    media::Peer,
};
use agcad_harness::Requirements;
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::StatusCode,
    response::Html,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    env, fs,
    path::{Path as FsPath, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
type Shared = Arc<Mutex<Service>>;
type Api = Result<Json<Value>, (StatusCode, String)>;
type Document = Arc<Mutex<Model>>;
fn bad(e: impl ToString) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, e.to_string())
}
fn expired() -> (StatusCode, String) {
    (StatusCode::GONE, "Viewport session expired".into())
}
fn token() -> Result<String, String> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).map_err(|e| e.to_string())?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}
#[derive(Clone, Serialize, Deserialize)]
struct Saved {
    name: String,
    #[serde(default)]
    created: bool,
    revision: u64,
    finishes: Overrides,
    edits: Vec<Value>,
}
struct Model {
    dir: PathBuf,
    saved: Saved,
    source: Source,
    scene: Arc<Scene>,
}
impl Model {
    fn save(&self, next: &Saved) -> Result<(), String> {
        let tmp = self.dir.join("changes.pending");
        let mut f = fs::File::create(&tmp).map_err(|e| e.to_string())?;
        serde_json::to_writer(&mut f, next).map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
        drop(f);
        fs::rename(tmp, self.dir.join("changes.json")).map_err(|e| e.to_string())
    }
    fn load(dir: PathBuf, saved: Saved, geometry: Arc<Scene>) -> Result<Self, String> {
        let kind = if geometry.description["kind"] == "Sheet" {
            "SurfaceBody"
        } else {
            "SolidBody"
        };
        let req = Requirements::parse(
            &json!({"id":"viewport","rev":1,"deliver":{"body_type":kind}}).to_string(),
        )
        .map_err(|e| e.to_string())?;
        let source = Source {
            base: if saved.created {
                Value::Null
            } else {
                json!({"requirements_rev":1,"expected_revision":0,"program":[{"id":"imported","op":"file.import_step","args":{"path":dir.join("model.step")}}]})
            },
            requirements: req,
            edits: saved.edits.clone(),
        };
        let scene = if source.edits.is_empty() {
            geometry
        } else {
            source.build(saved.revision + 1, Some(&geometry))?
        };
        Ok(Self {
            dir,
            saved,
            source,
            scene,
        })
    }
}
fn solid_requirements() -> Result<Requirements, String> {
    Requirements::parse(
        &json!({"id":"viewport","rev":1,"deliver":{"body_type":"SolidBody"}}).to_string(),
    )
    .map_err(|e| e.to_string())
}
struct Candidate {
    base: u64,
    saved: Saved,
    scene: Arc<Scene>,
    label: String,
    separate_part: bool,
}
#[derive(Clone, Copy, Default, serde::Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
#[repr(u32)]
enum Skybox {
    #[default]
    Dark = 0,
    Studio,
    Daylight,
    Sunset,
}
struct Session {
    skybox: Skybox,
    model_id: String,
    context_version: u64,
    preview_id: u64,
    media: Option<Peer>,
    image_frames: std::collections::VecDeque<crate::media::Frame>,
    showing_proposal: bool,
    model: Document,
    display: Arc<Scene>,
    selections: Vec<Value>,
    highlight: crate::media::Highlight,
    candidate: Option<Candidate>,
    last: Instant,
    seen: u64,
    active_requests: Arc<()>,
}
struct Service {
    backdrops: [Arc<tokio::sync::OnceCell<Bytes>>; 4],
    root: PathBuf,
    models: HashMap<String, Document>,
    sessions: HashMap<String, Session>,
    capacity: Arc<tokio::sync::Semaphore>,
    images: Arc<Mutex<Option<crate::media::ImageRenderer>>>,
}
fn show(session: &mut Session, scene: Arc<Scene>, values: &Overrides) {
    session.highlight = Default::default();
    if let Some(peer) = &mut session.media {
        peer.scene(scene.id, &scene.vertices);
        peer.palette(&materials::palette(&scene, values, &[]));
        peer.select(&session.highlight);
    }
    session.context_version += 1;
    session.image_frames.clear();
    session.selections.clear();
    session.display = scene;
}
fn state_info(s: &Session, m: &Model) -> Value {
    json!({"model":s.model_id,"skybox":s.skybox,"preview_id":s.candidate.as_ref().map(|_|s.preview_id),"revision":m.saved.revision,"empty":s.display.faces.is_empty(),"can_create":m.scene.faces.is_empty() && s.candidate.is_none(),"summary":s.display.description,"checks":s.display.report,"selection":s.selections.last(),"selections":s.selections,"preview":s.candidate.as_ref().map(|c|&c.label)})
}
async fn ice() -> Json<Value> {
    let mut servers = Vec::new();
    if let Ok(s) = env::var("AGCAD_STUN_SERVER") {
        servers.push(json!({"urls":s.replace("stun://","stun:")}))
    }
    if let Ok(s) = env::var("AGCAD_TURN_SERVER") {
        if let Ok(u) = reqwest::Url::parse(&s) {
            if let Some(host) = u.host_str() {
                let port = u
                    .port()
                    .unwrap_or(if u.scheme() == "turns" { 5349 } else { 3478 });
                servers.push(json!({"urls":format!("{}:{}:{}",u.scheme(),host,port),"username":u.username(),"credential":u.password().unwrap_or("")}));
            }
        }
    }
    Json(json!({"iceServers":servers}))
}
async fn models(State(state): State<Shared>) -> Json<Value> {
    let g = state.lock().unwrap();
    let mut items: Vec<_> = g
        .models
        .iter()
        .map(|(id, m)| json!({"id":id,"name":m.lock().unwrap().saved.name}))
        .collect();
    items.sort_by_key(|v| v["name"].as_str().unwrap_or("").to_owned());
    Json(json!(items))
}
#[derive(Deserialize)]
struct Upload {
    name: String,
}
async fn upload(State(state): State<Shared>, Query(q): Query<Upload>, bytes: Bytes) -> Api {
    let name = q.name.trim().to_string();
    if name.len() > 180
        || !["step", "stp"].contains(
            &name
                .rsplit('.')
                .next()
                .unwrap_or("")
                .to_ascii_lowercase()
                .as_str(),
        )
    {
        return Err(bad("Choose a .step or .stp file"));
    }
    let (root, cap) = {
        let g = state.lock().unwrap();
        (g.root.clone(), g.capacity.clone())
    };
    let _permit = cap
        .try_acquire_owned()
        .map_err(|_| bad("Server is busy. Retry shortly."))?;
    let id = token().map_err(bad)?;
    let dir = root.join(&id);
    let model = tokio::task::spawn_blocking(move || -> Result<Model, String> {
        let src = std::str::from_utf8(&bytes).map_err(|_| "Expected a STEP Part 21 text file")?;
        let scene = Scene::step(src)?;
        fs::create_dir(&dir).map_err(|e| e.to_string())?;
        let saved = Saved {
            name,
            created: false,
            revision: 0,
            finishes: Overrides::new(),
            edits: vec![],
        };
        fs::write(dir.join("model.step"), bytes).map_err(|e| e.to_string())?;
        let model = Model::load(dir, saved, scene)?;
        model.save(&model.saved)?;
        Ok(model)
    })
    .await
    .map_err(bad)?
    .map_err(bad)?;
    state
        .lock()
        .unwrap()
        .models
        .insert(id.clone(), Arc::new(Mutex::new(model)));
    Ok(Json(json!({"id":id})))
}
#[derive(Deserialize)]
struct Offer {
    model: String,
    #[serde(default)]
    skybox: Skybox,
    #[serde(default)]
    session: Option<String>,
    #[serde(default)]
    sdp: String,
}
async fn offer(State(state): State<Shared>, Json(o): Json<Offer>) -> Api {
    let (doc, cap) = {
        let g = state.lock().unwrap();
        if o.session.is_none() && g.sessions.len() >= 16 {
            return Err(bad("All viewport slots are in use"));
        }
        (
            g.models
                .get(&o.model)
                .cloned()
                .ok_or_else(|| bad("Unknown model"))?,
            g.capacity.clone(),
        )
    };
    let _permit = cap.try_acquire_owned().map_err(|_| bad("Server is busy"))?;
    if o.sdp.len() > 65536 {
        return Err(bad("SDP too large"));
    }
    let (peer, sdp) = if o.sdp.is_empty() {
        (None, String::new())
    } else {
        let (peer, sdp) = tokio::task::spawn_blocking(move || Peer::open(&o.sdp))
            .await
            .map_err(bad)?
            .map_err(bad)?;
        (Some(peer), sdp)
    };
    let id = token().map_err(bad)?;
    let mut g = state.lock().unwrap();
    let m = doc.lock().unwrap();
    if let Some(existing) = o.session {
        let s = g
            .sessions
            .get_mut(&existing)
            .filter(|s| Arc::ptr_eq(&s.model, &doc))
            .ok_or_else(expired)?;
        s.media = peer;
        if let Some(p) = &mut s.media {
            p.skybox(s.skybox as u32);
            p.scene(s.display.id, &s.display.vertices);
            let values = s
                .candidate
                .as_ref()
                .filter(|_| s.showing_proposal)
                .map(|c| &c.saved.finishes)
                .unwrap_or(&m.saved.finishes);
            p.palette(&materials::palette(&s.display, values, &s.selections));
            p.select(&s.highlight);
        }
        return Ok(Json(
            json!({"id":existing,"sdp":sdp,"state":state_info(s,&m)}),
        ));
    }
    if g.sessions.len() >= 16 {
        return Err(bad("All viewport slots are in use"));
    }
    let mut s = Session {
        skybox: o.skybox,
        model_id: o.model,
        context_version: 0,
        preview_id: 0,
        media: peer,
        image_frames: Default::default(),
        showing_proposal: false,
        model: doc.clone(),
        display: m.scene.clone(),
        selections: vec![],
        highlight: Default::default(),
        candidate: None,
        last: Instant::now(),
        seen: m.saved.revision,
        active_requests: Arc::new(()),
    };
    if let Some(p) = &mut s.media {
        p.skybox(s.skybox as u32);
    }
    show(&mut s, m.scene.clone(), &m.saved.finishes);
    let info = state_info(&s, &m);
    drop(m);
    g.sessions.insert(id.clone(), s);
    Ok(Json(json!({"id":id,"sdp":sdp,"state":info})))
}
#[derive(Deserialize)]
struct Command {
    op: String,
    #[serde(default)]
    data: Value,
}
async fn command(
    State(state): State<Shared>,
    Path(id): Path<String>,
    Json(cmd): Json<Command>,
) -> Api {
    if cmd.op == "ask" {
        return propose(state, id, cmd.data, false).await;
    }
    if cmd.op == "primitive" {
        return propose(state, id, cmd.data, true).await;
    }
    let mut g = state.lock().unwrap();
    if cmd.op == "close" {
        g.sessions.remove(&id);
        return Ok(Json(json!({})));
    }
    let library_root = g.root.clone();
    let s = g.sessions.get_mut(&id).ok_or_else(expired)?;
    s.last = Instant::now();
    let doc = s.model.clone();
    let mut m = doc.lock().unwrap();
    if s.seen != m.saved.revision {
        s.showing_proposal = false;
        s.candidate = None;
        show(s, m.scene.clone(), &m.saved.finishes);
        s.seen = m.saved.revision;
        if cmd.op != "status" {
            return Err((
                StatusCode::CONFLICT,
                "Model changed. Select the face again.".into(),
            ));
        }
    }
    let d = &cmd.data;
    if matches!(
        cmd.op.as_str(),
        "apply" | "discard" | "current" | "proposed"
    ) {
        if let Some(expected) = d.get("preview_id") {
            if s.candidate.is_none() || expected.as_u64() != Some(s.preview_id) {
                return Err((
                    StatusCode::CONFLICT,
                    "The reviewed preview has changed. Review the current proposal.".into(),
                ));
            }
        }
    }
    match cmd.op.as_str() {
        "status" => {}
        "skybox" => {
            s.skybox = serde_json::from_value(d["preset"].clone()).map_err(bad)?;
            if let Some(p) = &mut s.media {
                p.skybox(s.skybox as u32);
            }
        }
        "frame" => {
            let f = s
                .media
                .as_ref()
                .ok_or_else(|| bad("No video stream"))?
                .frame(stamp(d)?)
                .map_err(bad)?;
            return Ok(Json(json!({"sequence":f.sequence})));
        }
        "pick" => {
            let timestamp = stamp(d)?;
            let f = if d["image"] == true {
                s.image_frames
                    .iter()
                    .find(|f| f.timestamp == timestamp)
                    .cloned()
                    .ok_or_else(|| bad("Displayed image expired. Click again."))?
            } else {
                s.media
                    .as_ref()
                    .ok_or_else(|| bad("No video stream"))?
                    .frame(timestamp)
                    .map_err(bad)?
            };
            if f.scene != s.display.id {
                return Err(bad("Displayed revision changed. Click again."));
            }
            let mode = d.get("mode").and_then(Value::as_str).unwrap_or("replace");
            if !matches!(mode, "replace" | "toggle" | "add" | "remove" | "touch_box") {
                return Err(bad("Invalid selection mode"));
            }
            let hits = if let Some(rect) = d.get("rect") {
                s.display
                    .box_pick(&f, serde_json::from_value(rect.clone()).map_err(bad)?)
                    .map_err(bad)?
            } else {
                let hit = s
                    .display
                    .pick(
                        &f,
                        d["x"].as_f64().ok_or_else(|| bad("Missing x"))?,
                        d["y"].as_f64().ok_or_else(|| bad("Missing y"))?,
                        d["viewport_height"].as_f64(),
                    )
                    .map_err(bad)?;
                (!hit.is_null()).then_some(hit).into_iter().collect()
            };
            let same = |a: &Value, b: &Value| {
                a["body"] == b["body"] && a["kind"] == b["kind"] && a["index"] == b["index"]
            };
            if d["inspect"]==true {
                let active=hits.first().is_some_and(|h|s.selections.iter().any(|v|same(v,h)));
                return Ok(Json(json!({"selected":active})));
            }
            let mode = if mode == "touch_box" {
                let origin = s
                    .display
                    .pick(
                        &f,
                        d["x"].as_f64().ok_or_else(|| bad("Missing x"))?,
                        d["y"].as_f64().ok_or_else(|| bad("Missing y"))?,
                        d["viewport_height"].as_f64(),
                    )
                    .map_err(bad)?;
                if s.selections.iter().any(|v| same(v, &origin)) {
                    "remove"
                } else {
                    "add"
                }
            } else {
                mode
            };
            let mut next = if mode == "replace" {
                vec![]
            } else {
                s.selections.clone()
            };
            let values = s
                .candidate
                .as_ref()
                .filter(|c| c.scene.id == s.display.id && s.showing_proposal)
                .map(|c| &c.saved.finishes)
                .unwrap_or(&m.saved.finishes);
            for mut hit in hits {
                if let Some(i) = next.iter().position(|v| same(v, &hit)) {
                    if mode == "toggle" || mode == "remove" {
                        next.remove(i);
                    }
                    continue;
                }
                if mode == "remove" {
                    continue;
                }
                if hit["kind"] == "face" {
                    hit["appearance"] = materials::appearance(&hit, values).1;
                }
                next.push(hit);
            }
            if next != s.selections {
                s.context_version += 1;
                s.selections = next;
                s.highlight = s.display.highlights(&s.selections);
                if let Some(p) = &mut s.media {
                    p.palette(&materials::palette(&s.display, values, &s.selections));
                    p.select(&s.highlight);
                }
            }
        }
        "clear_selection" => {
            if !s.selections.is_empty() {
                s.context_version += 1;
                s.selections.clear();
                s.highlight = Default::default();
                let values = s
                    .candidate
                    .as_ref()
                    .filter(|_| s.showing_proposal)
                    .map(|c| &c.saved.finishes)
                    .unwrap_or(&m.saved.finishes);
                if let Some(p) = &mut s.media {
                    p.palette(&materials::palette(&s.display, values, &[]));
                    p.select(&s.highlight);
                }
            }
        }
        "fit_bounds" => {
            if d["selection"] == true && s.selections.is_empty() {
                return Err(bad("Select geometry to fit"));
            }
            return Ok(Json(
                json!({"bounds":s.display.bounds(if d["selection"]==true{&s.selections}else{&[]})}),
            ));
        }
        "finish" => {
            if s.candidate.is_some() {
                return Err(bad("Apply or discard the current proposal first"));
            }
            let next = finish(&m.saved, &s.selections, d).map_err(bad)?;
            s.showing_proposal = true;
            let mut selected = s.selections.clone();
            show(s, m.scene.clone(), &next.finishes);
            s.highlight = s.display.highlights(&selected);
            if let Some(peer) = &mut s.media {
                peer.palette(&materials::palette(&s.display, &next.finishes, &selected));
                peer.select(&s.highlight);
            }
            for v in &mut selected {
                if v["kind"] == "face" {
                    v["appearance"] = materials::appearance(v, &next.finishes).1;
                }
            }
            s.selections = selected;
            s.preview_id += 1;
            s.candidate = Some(Candidate {
                base: m.saved.revision,
                saved: next,
                scene: m.scene.clone(),
                label: "Material change".into(),
                separate_part: false,
            });
        }
        "current" => {
            s.showing_proposal = false;
            show(s, m.scene.clone(), &m.saved.finishes)
        }
        "proposed" => {
            let c = s.candidate.as_ref().ok_or_else(|| bad("No proposal"))?;
            s.showing_proposal = true;
            let scene = c.scene.clone();
            let values = c.saved.finishes.clone();
            show(s, scene, &values);
        }
        "discard" => {
            s.showing_proposal = false;
            s.candidate = None;
            show(s, m.scene.clone(), &m.saved.finishes);
        }
        "apply" => {
            let c = s
                .candidate
                .as_ref()
                .ok_or_else(|| bad("No pending proposal"))?;
            if c.base != m.saved.revision {
                return Err(bad("Stale proposal"));
            }
            if c.separate_part {
                let model_id = token().map_err(bad)?;
                let dir = library_root.join(&model_id);
                let source = Source {
                    base: Value::Null,
                    requirements: solid_requirements().map_err(bad)?,
                    edits: c.saved.edits.clone(),
                };
                let new_model = Model {
                    dir,
                    saved: c.saved.clone(),
                    source,
                    scene: c.scene.clone(),
                };
                fs::create_dir(&new_model.dir).map_err(bad)?;
                new_model.save(&new_model.saved).map_err(bad)?;
                s.showing_proposal = false;
                s.candidate = None;
                s.seen = new_model.saved.revision;
                show(s, new_model.scene.clone(), &new_model.saved.finishes);
                s.model_id = model_id.clone();
                let info = state_info(s, &new_model);
                let new_doc = Arc::new(Mutex::new(new_model));
                s.model = new_doc.clone();
                g.models.insert(model_id, new_doc);
                return Ok(Json(info));
            }
            m.save(&c.saved).map_err(bad)?;
            let c = s.candidate.take().unwrap();
            s.showing_proposal = false;
            m.source.edits = c.saved.edits.clone();
            m.saved = c.saved;
            m.scene = c.scene;
            s.seen = m.saved.revision;
            show(s, m.scene.clone(), &m.saved.finishes);
        }
        _ => return Err(bad("Unknown operation")),
    }
    Ok(Json(state_info(s, &m)))
}
fn stamp(v: &Value) -> Result<u32, (StatusCode, String)> {
    v["timestamp"]
        .as_u64()
        .and_then(|x| x.try_into().ok())
        .ok_or_else(|| bad("Missing presented frame timestamp"))
}
fn finish(base: &Saved, selections: &[Value], d: &Value) -> Result<Saved, String> {
    let mut keys = std::collections::BTreeSet::new();
    for face in selections.iter().filter(|v| v["kind"] == "face") {
        keys.insert(materials::key(face, d["scope"].as_str().unwrap_or("face"))?);
    }
    if keys.is_empty() {
        return Err("Select faces to change the finish".into());
    }
    let finish: Finish = serde_json::from_value(d["finish"].clone()).map_err(|e| e.to_string())?;
    finish.validate()?;
    let mut next = base.clone();
    next.revision += 1;
    for key in keys {
        let f = finish.clone();
        let entry = next.finishes.entry(key).or_default();
        if let Some(value) = f.color {
            entry.color = Some(value);
        }
        if let Some(value) = f.roughness {
            entry.roughness = Some(value);
        }
        if let Some(value) = f.metalness {
            entry.metalness = Some(value);
        }
        if let Some(value) = f.designation {
            entry.designation = Some(value);
        }
        if let Some(value) = f.density_kg_m3 {
            entry.density_kg_m3 = Some(value);
        }
    }
    Ok(next)
}
fn evaluate_proposal(
    source: &Source,
    scene: &Arc<Scene>,
    saved: &Saved,
    selection: &[Value],
    manual: bool,
    reply: Value,
) -> Result<(String, Option<Candidate>), String> {
    if let Some(spec) = reply.get("sphere") {
        let sphere: crate::create::Primitive =
            serde_json::from_value(spec.clone()).map_err(|e| e.to_string())?;
        let mut program = sphere.program()?;
        let occupied = !scene.faces.is_empty();
        let separate = occupied && reply["join"] != true;
        if separate {
            let source = Source {
                base: Value::Null,
                requirements: solid_requirements()?,
                edits: vec![program.clone()],
            };
            let geometry = source.build(2, None)?;
            let new = Saved {
                name: "Sphere".into(),
                created: true,
                revision: 1,
                finishes: Default::default(),
                edits: vec![program],
            };
            return Ok((
                format!(
                    "{} Apply to save as a separate part.",
                    reply["answer"].as_str().unwrap_or("Sphere preview.")
                ),
                Some(Candidate {
                    base: saved.revision,
                    saved: new,
                    scene: geometry,
                    label: "New sphere part".into(),
                    separate_part: true,
                }),
            ));
        }
        program["expected_revision"] = json!(scene.revision);
        let result = evaluate_proposal(
            source,
            scene,
            saved,
            selection,
            manual,
            json!({"answer":reply["answer"],"program":program}),
        );
        if occupied
            && let Err(error) = &result
            && error.contains("general B-rep boolean")
        {
            return Ok(("Joining a sphere to this body requires a general curved-solid boolean, which the native kernel does not yet support.".into(),None));
        }
        return result;
    }
    let reply = if let Some(spec) = reply.get("primitive") {
        if !scene.faces.is_empty() {
            return Err("Start a new part to create a first solid".into());
        }
        let shape: crate::create::Primitive =
            serde_json::from_value(spec.clone()).map_err(|e| e.to_string())?;
        json!({"answer":reply["answer"],"program":shape.program()?})
    } else {
        reply
    };
    let answer = reply["answer"].as_str().unwrap_or("").to_owned();
    let c = if reply.get("finish").is_some_and(Value::is_object) {
        let next = finish(saved, selection, &reply)?;
        Some(Candidate {
            base: saved.revision,
            saved: next,
            scene: scene.clone(),
            label: "Assistant material proposal".into(),
            separate_part: false,
        })
    } else if reply.get("program").is_some() {
        if scene.description["bodies"].as_u64().is_some_and(|n| n > 1) {
            return Err("Geometry editing currently requires a single STEP body".into());
        }
        let full = if reply["program"].is_array() {
            &reply
        } else {
            &reply["program"]
        };
        let edit = source.check_edit(full)?;
        for stmt in edit["program"].as_array().into_iter().flatten() {
            let targets=stmt["args"]["edges"].as_array().map(Vec::as_slice).unwrap_or_default();
            let all=if targets.len()==12 {scene.block_edges()}else{vec![]};
            let whole_block=!all.is_empty() && all.iter().all(|v|targets.iter().any(|e|e["body"]==v["body"] && e["endpoints_mm"]==v["endpoints_mm"]));
            for face in stmt["args"]["faces"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|f| f.get("plane_origin").is_some())
            {
                if !selection
                    .iter()
                    .filter(|v| v["kind"] == "face")
                    .any(|v| crate::assistant::tools::plane_target(v).ok().as_ref() == Some(face))
                {
                    return Err("Face selection changed".into());
                }
            }
            for edge in stmt["args"]["edges"].as_array().into_iter().flatten() {
                if let Some(p) = edge.get("vertex_mm") {
                    if !selection.iter().any(|v| {
                        v["kind"] == "vertex"
                            && v["body"] == edge["body"]
                            && *p
                                == json!([
                                    v["point_mm"]["x"],
                                    v["point_mm"]["y"],
                                    v["point_mm"]["z"]
                                ])
                    }) {
                        return Err("Vertex selection changed".into());
                    }
                }
                if let Some(ends) = edge.get("endpoints_mm") {
                    if !whole_block && !selection.iter().any(|v| {
                        v["kind"] == "edge"
                            && edge["body"] == v["body"]
                            && ends == &v["endpoints_mm"]
                    }) {
                        return Err("The edge reference differs from the current selection. Pick the edge again.".into());
                    }
                }
            }
        }
        let mut next = saved.clone();
        next.revision += 1;
        next.edits.push(edit);
        let mut proposed = source.clone();
        proposed.edits = next.edits.clone();
        let rebuilt = proposed.build(next.revision + 1, Some(scene))?;
        Some(Candidate {
            base: saved.revision,
            saved: next,
            scene: rebuilt,
            separate_part: false,
            label: if manual {
                "New solid".into()
            } else {
                "Assistant geometry proposal".into()
            },
        })
    } else if !answer.is_empty() {
        None
    } else {
        return Err("The bridge returned no answer or proposal".into());
    };
    Ok((answer, c))
}
async fn propose(state: Shared, id: String, data: Value, manual: bool) -> Api {
    let prompt = if manual {
        ""
    } else {
        data["prompt"]
            .as_str()
            .filter(|s| !s.trim().is_empty() && s.len() <= 8192)
            .ok_or_else(|| bad("Enter a question or edit request"))?
    }
    .to_owned();
    let (source, scene, saved, selection, bridge, cap, _active, context_version) = {
        let mut g = state.lock().unwrap();
        let bridge =
            if manual {
                None
            } else {
                Some(crate::assistant::Assistant::configured()
            .map_err(bad)?
            .ok_or_else(|| {
                bad("Set DEEPSEEK_API_KEY on the GPU server to enable its CAD assistant")
            })?)
            };
        let cap = g.capacity.clone();
        let s = g.sessions.get_mut(&id).ok_or_else(expired)?;
        s.last = Instant::now();
        let m = s.model.lock().unwrap();
        if s.candidate.is_some() || s.seen != m.saved.revision {
            return Err(bad("Review the current revision first"));
        }
        (
            m.source.clone(),
            m.scene.clone(),
            m.saved.clone(),
            s.selections.clone(),
            bridge,
            cap,
            s.active_requests.clone(),
            s.context_version,
        )
    };
    let _permit = cap
        .try_acquire_owned()
        .map_err(|_| bad("Assistant is busy"))?;
    let base = saved.revision;
    let (answer, candidate) = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let evaluate =
            |reply| evaluate_proposal(&source, &scene, &saved, &selection, manual, reply);
        if let Some(provider) = bridge {
            let observation =
                source.observation(&scene, &prompt, selection.clone(), &data["history"]);
            let preference = match data.get("assistant") {
                Some(value) => serde_json::from_value(value.clone()).map_err(|_| "Invalid assistant selection")?,
                None => Default::default(),
            };
            provider.complete(&observation, preference, evaluate)
        } else {
            evaluate(json!({"primitive":data}))
        }
    })
    .await
    .map_err(bad)?
    .map_err(bad)?;
    let mut g = state.lock().unwrap();
    let s = g.sessions.get_mut(&id).ok_or_else(expired)?;
    let doc = s.model.clone();
    let m = doc.lock().unwrap();
    if base != m.saved.revision || s.context_version != context_version || s.candidate.is_some() {
        return Err((
            StatusCode::CONFLICT,
            "The model, selection or preview changed while the assistant was working. Retry the request.".into(),
        ));
    }
    if let Some(c) = candidate {
        s.showing_proposal = true;
        show(s, c.scene.clone(), &c.saved.finishes);
        s.preview_id += 1;
        s.candidate = Some(c);
    }
    let mut out = state_info(s, &m);
    out["answer"] = json!(answer);
    Ok(Json(out))
}
async fn new_part(State(state): State<Shared>, Json(req): Json<Upload>) -> Api {
    let name = req.name.trim();
    if name.is_empty() || name.len() > 180 {
        return Err(bad("Enter a part name of up to 180 bytes"));
    }
    let id = token().map_err(bad)?;
    let dir = state.lock().unwrap().root.join(&id);
    fs::create_dir(&dir).map_err(bad)?;
    let model = Model::load(
        dir,
        Saved {
            name: name.into(),
            created: true,
            revision: 0,
            finishes: Default::default(),
            edits: vec![],
        },
        Scene::empty(),
    )
    .map_err(bad)?;
    model.save(&model.saved).map_err(bad)?;
    state
        .lock()
        .unwrap()
        .models
        .insert(id.clone(), Arc::new(Mutex::new(model)));
    Ok(Json(json!({"id":id})))
}
pub fn run() -> Result<(), String> {
    let root =
        PathBuf::from(env::var_os("AGCAD_VIEW_DATA").unwrap_or_else(|| "agcad-models".into()));
    fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let root = fs::canonicalize(root).map_err(|e| e.to_string())?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join("server.lock"))
        .map_err(|e| e.to_string())?;
    lock.try_lock()
        .map_err(|e| format!("Model library is already in use: {e}"))?;
    let mut loaded = HashMap::new();
    for entry in fs::read_dir(&root).map_err(|e| e.to_string())?.flatten() {
        let dir = entry.path();
        if !dir.join("changes.json").is_file() {
            continue;
        }
        match restore(&dir) {
            Ok(m) => {
                loaded.insert(
                    entry.file_name().to_string_lossy().into_owned(),
                    Arc::new(Mutex::new(m)),
                );
            }
            Err(e) => eprintln!("{}: {e}", dir.display()),
        }
    }
    let shared = Arc::new(Mutex::new(Service {
        backdrops: Default::default(),
        root,
        models: loaded,
        sessions: HashMap::new(),
        capacity: Arc::new(tokio::sync::Semaphore::new(4)),
        images: Default::default(),
    }));
    let bind = env::var("AGCAD_VIEW_BIND").unwrap_or_else(|_| "0.0.0.0:8090".into());
    tokio::runtime::Runtime::new()
        .map_err(|e| e.to_string())?
        .block_on(async {
            let cleanup = shared.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(15)).await;
                    cleanup.lock().unwrap().sessions.retain(|_, p| {
                        Arc::strong_count(&p.active_requests) > 1
                            || p.last.elapsed() < Duration::from_secs(120)
                    });
                }
            });
            let app = Router::new()
                .route("/", get(|| async { Html(include_str!("index.html")) }))
                .route(
                    "/workbench.js",
                    get(|| async {
                        (
                            [
                                ("content-type", "text/javascript; charset=utf-8"),
                                ("cache-control", "no-cache"),
                            ],
                            include_str!("workbench.js"),
                        )
                    }),
                )
                .route(
                    "/workbench.css",
                    get(|| async {
                        (
                            [
                                ("content-type", "text/css; charset=utf-8"),
                                ("cache-control", "no-cache"),
                            ],
                            include_str!("workbench.css"),
                        )
                    }),
                )
                .route("/ice", get(ice))
                .route("/assistant/options", get(|| async {
                    let configured = crate::assistant::Assistant::configured().map_err(bad)?;
                    let models = configured.map(|a| a.options()).unwrap_or_default();
                    Ok::<_,(StatusCode,String)>(([("cache-control","no-store")], Json(json!({"models":models}))))
                }))
                .route("/skybox/{preset}", get(skybox_backdrop))
                .route("/models", get(models).post(upload))
                .route("/models/new", post(new_part))
                .route("/session/{id}/image", post(image))
                .route("/offer", post(offer))
                .route("/session/{id}", post(command))
                .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
                .with_state(shared);
            let listener = tokio::net::TcpListener::bind(bind)
                .await
                .map_err(|e| e.to_string())?;
            println!(
                "AgCAD GPU server listening at http://{}",
                listener.local_addr().map_err(|e| e.to_string())?
            );
            axum::serve(listener, app).await.map_err(|e| e.to_string())
        })
}
fn restore(dir: &FsPath) -> Result<Model, String> {
    let saved: Saved =
        serde_json::from_slice(&fs::read(dir.join("changes.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let scene = if saved.created {
        Scene::empty()
    } else {
        Scene::step(&fs::read_to_string(dir.join("model.step")).map_err(|e| e.to_string())?)?
    };
    Model::load(dir.to_owned(), saved, scene)
}
// The empty workbench has no model or peer. Cache its initial camera view once per preset.
async fn skybox_backdrop(
    State(state): State<Shared>,
    Path(preset): Path<Skybox>,
) -> Result<impl axum::response::IntoResponse, (StatusCode, String)> {
    let cell = state.lock().unwrap().backdrops[preset as usize].clone();
    let jpeg = cell.get_or_try_init(|| async move {
        tokio::task::spawn_blocking(move || -> Result<Bytes, String> {
            let mut renderer = crate::media::ImageRenderer::new()?;
            let (jpeg, _) = renderer.render(Scene::empty(), &[], &Default::default(), preset as u32,
                [0.64, 0.62, 1.0, 0.0, 0.0])?;
            Ok(Bytes::from(jpeg))
        }).await.map_err(bad)?.map_err(bad)
    }).await?;
    Ok(([("content-type","image/jpeg"),("cache-control","public, max-age=3600")],jpeg.clone()))
}
async fn image(
    State(state): State<Shared>,
    Path(id): Path<String>,
    Json(pose): Json<[f32; 5]>,
) -> Result<impl axum::response::IntoResponse, (StatusCode, String)> {
    if pose.iter().any(|v| !v.is_finite()) || !(0.05..=100.).contains(&pose[2]) {
        return Err(bad("Invalid camera"));
    }
    let (renderer, scene, colors, highlight, skybox, cap) = {
        let mut g = state.lock().unwrap();
        let renderer = g.images.clone();
        let cap = g.capacity.clone();
        let s = g.sessions.get_mut(&id).ok_or_else(expired)?;
        if s.display.vertices.is_empty() {
            return Err(bad("Empty part"));
        }
        s.last = Instant::now();
        let m = s.model.lock().unwrap();
        let values = s
            .candidate
            .as_ref()
            .filter(|_| s.showing_proposal)
            .map(|c| &c.saved.finishes)
            .unwrap_or(&m.saved.finishes);
        (
            renderer,
            s.display.clone(),
            materials::palette(&s.display, values, &s.selections),
            s.highlight.clone(),
            s.skybox,
            cap,
        )
    };
    let _permit = cap.try_acquire_owned().map_err(|_| bad("Server is busy"))?;
    let snapshot = scene.clone();
    let (jpeg, frame) = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let mut cached = renderer.lock().unwrap();
        if cached.is_none() {
            *cached = Some(crate::media::ImageRenderer::new()?)
        }
        let result = cached
            .as_mut()
            .unwrap()
            .render(snapshot, &colors, &highlight, skybox as u32, pose);
        if result.is_err() {
            *cached = None
        }
        result
    })
    .await
    .map_err(bad)?
    .map_err(bad)?;
    let mut g = state.lock().unwrap();
    let s = g.sessions.get_mut(&id).ok_or_else(expired)?;
    if !Arc::ptr_eq(&s.display, &scene) || s.skybox != skybox {
        return Err((StatusCode::CONFLICT, "Scene changed".into()));
    }
    let stamp = frame.timestamp.to_string();
    s.image_frames.push_back(frame);
    if s.image_frames.len() > 8 {
        s.image_frames.pop_front();
    }
    Ok((
        [
            ("content-type", "image/jpeg".into()),
            ("cache-control", "no-store".into()),
            ("x-frame-id", stamp),
        ],
        jpeg,
    ))
}
