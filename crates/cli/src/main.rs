//! `agcad` command-line tool (spec 07 § LLM path; spec 10).
//!
//! ```text
//! agcad version
//! agcad catalog [filter]
//! agcad run <program.json> --requirements <req.json> [--out <dir>] [--part <name>]
//! agcad serve --requirements <req.json> [--part <name>]      # MCP over stdio
//! agcad trace diff <a.jsonl> <b.jsonl>
//! agcad check-step <file.stp>
//! agcad bench-device [--configs N] [--rounds R] [--levels L] [--cap C] [--device cuda|cpu] [--no-aniso]
//! agcad check-program <program.json>                          # lint, all diagnostics, exit 1 on errors
//! agcad conformance [--fuzz N] [--capture <tests.rs>]       # shrunk failures → regression tests
//! agcad improve <archive-dir> <command> ...
//! ```

use agcad_harness::{HarnessSession, Requirements};
use agcad_ir::Catalog;
use std::path::PathBuf;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("agcad: {msg}");
    std::process::exit(2)
}

fn requirements(args: &[String]) -> Requirements {
    let path = flag(args, "--requirements").unwrap_or_else(|| die("--requirements <file.json> is required (requirements are caller-owned, AG-06-001)"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| die(format!("{path}: {e}")));
    Requirements::parse(&text).unwrap_or_else(|e| die(format!("{path}: {e}")))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let catalog = Catalog::global();
    match args.get(1).map(String::as_str) {
        Some("version") | None => {
            println!("agcad {} (driver ABI {}, catalog {}, {} ops, {} standard features implemented)", env!("CARGO_PKG_VERSION"), agcad_driver::ABI_VERSION, catalog.version, catalog.ops.len(), agcad_std::standard_registry().implemented().len());
        }
        Some("catalog") => {
            let f = args.get(2).map(String::as_str).unwrap_or("");
            let reg = agcad_std::standard_registry();
            for o in catalog.ops.iter().filter(|o| f.is_empty() || o.name.contains(f) || o.aliases.iter().any(|a| a.contains(f))) {
                println!("{:<40} 0x{:04X} {:<4} {}", o.name, o.opcode, if reg.get(&o.name).is_some() { "impl" } else { "-" }, o.summary);
            }
        }
        Some("run") => {
            let prog_path = args.get(2).unwrap_or_else(|| die("usage: agcad run <program.json> --requirements <req.json>"));
            let req = requirements(&args);
            let text = std::fs::read_to_string(prog_path).unwrap_or_else(|e| die(format!("{prog_path}: {e}")));
            let program = agcad_ir::json::decode_program(&text, catalog).unwrap_or_else(|e| die(e));
            let mut s = HarnessSession::new(req, &flag(&args, "--part").unwrap_or_else(|| "part".into()));
            if let Some(out) = flag(&args, "--out") { s.out_dir = PathBuf::from(out); }
            match s.run_program("main", &program, catalog) {
                Ok(out) => {
                    println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "report": out.report, "delivery": out.delivery })).unwrap());
                    let trace_path = s.out_dir.join("trace.jsonl");
                    let _ = std::fs::create_dir_all(&s.out_dir);
                    let _ = std::fs::write(&trace_path, s.trace.to_jsonl());
                    eprintln!("trace: {}", trace_path.display());
                }
                Err((st, msg)) => die(format!("{}: {msg}", st.name())),
            }
        }
        Some("serve") => {
            let req = requirements(&args);
            let mut server = agcad_server::Server::new(req, &flag(&args, "--part").unwrap_or_else(|| "part".into()));
            if let Err(e) = server.serve_stdio() { die(e); }
        }
        Some("trace") if args.get(2).map(String::as_str) == Some("diff") => {
            let (a, b) = (args.get(3).unwrap_or_else(|| die("trace diff <a> <b>")), args.get(4).unwrap_or_else(|| die("trace diff <a> <b>")));
            let load = |p: &String| agcad_trace::Trace::from_jsonl(&std::fs::read_to_string(p).unwrap_or_else(|e| die(e))).unwrap_or_else(|e| die(e));
            match load(a).diverges_from(&load(b)) {
                None => println!("identical (bit-identical replay)"),
                Some(i) => { println!("diverges at event {i}"); std::process::exit(1); }
            }
        }
        Some("bench-device") => {
            let num = |k: &str, d: u32| flag(&args, k).and_then(|s| s.parse().ok()).unwrap_or(d);
            let cfg = agcad_dops::StudioConfig { configs: num("--configs", 4096), records: 4, max_rounds: num("--rounds", 16), coarse: 8, levels: num("--levels", 5), cap: num("--cap", 4096), nodes: 8, budget: 64, tighten: 0, max_cells: 0,
                policy: if flag(&args, "--policy").as_deref() == Some("cmaes") { agcad_dops::abi::policy::CMAES } else { agcad_dops::abi::policy::MOCK }, island: num("--island", 16), aniso: !args.iter().any(|v| v == "--no-aniso") };
            let base = agcad_dops::examples::flange(60.0, 12.0, 6.0);
            let mut goal = agcad_dops::examples::goal(60_000.0, 66_000.0);
            if cfg.policy == agcad_dops::abi::policy::CMAES { goal.hole_r = 0.0; } // parameters only
            let want_cpu = flag(&args, "--device").as_deref() == Some("cpu");
            let cuda = if want_cpu { None } else { agcad_driver_cuda::CudaDevice::open(0).ok() };
            let (device, wall, summary, stages) = match cuda {
                Some(dev) => {
                    let st = agcad_dops::Studio::new(dev, &base, &goal, cfg).unwrap_or_else(|e| die(e));
                    let t0 = std::time::Instant::now();
                    st.run_graph().unwrap_or_else(|e| die(e));
                    let wall = t0.elapsed().as_secs_f64();
                    let sm = st.results(4).unwrap_or_else(|e| die(e));
                    let stg = st.stage_times(sm.rounds).unwrap_or_else(|e| die(e));
                    ("cuda (graph WHILE)", wall, sm, stg)
                }
                None => {
                    let st = agcad_dops::Studio::new(agcad_driver_cpu::CpuDevice::new(), &base, &goal, cfg).unwrap_or_else(|e| die(e));
                    let t0 = std::time::Instant::now();
                    st.run_host_loop().unwrap_or_else(|e| die(e));
                    let wall = t0.elapsed().as_secs_f64();
                    let sm = st.results(4).unwrap_or_else(|e| die(e));
                    let stg = st.stage_times(sm.rounds).unwrap_or_else(|e| die(e));
                    ("cpu reference (host loop)", wall, sm, stg)
                }
            };
            let fine = cfg.coarse << (cfg.levels - 1);
            println!("device: {device}");
            println!("configurations {} × rounds {} | adaptive {}³ root, {} levels → {}³ finest, cap {} cells/config/level | goal met {}/{}", cfg.configs, summary.rounds, cfg.coarse, cfg.levels, fine, cfg.cap, summary.done, cfg.configs);
            println!("wall {:.3} s | {:.0} config-rounds/s | in-loop readback {} B", wall, cfg.configs as f64 * summary.rounds as f64 / wall, summary.in_loop_readback_bytes);
            let n = stages.len().max(1) as f64;
            let total: f64 = stages.iter().flatten().map(|&x| x as f64).sum::<f64>().max(1.0);
            println!("{:<10} {:>12} {:>8}", "stage", "ms/round", "share");
            for (k, name) in agcad_dops::abi::STAGE_NAMES.iter().enumerate() {
                let t: f64 = stages.iter().map(|r| r[k] as f64).sum();
                println!("{:<10} {:>12.3} {:>7.1}%", name, t / n / 1e6, 100.0 * t / total);
            }
            if let Some(b) = summary.top.first() {
                println!("best: configuration {} volume ≈ {:.1} mm³ ∈ [{:.1}, {:.1}] (bounded)", b.configuration, b.obs[12], b.obs[0], b.obs[1]);
            }
        }
        Some("check-program") => {
            let p = args.get(2).unwrap_or_else(|| die("check-program <program.json>"));
            let text = std::fs::read_to_string(p).unwrap_or_else(|e| die(e));
            let r = agcad_ir::lint::check_program(&text, Catalog::global());
            println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
            if !r.ok { std::process::exit(1); }
        }
        Some("check-step") => {
            let p = args.get(2).unwrap_or_else(|| die("check-step <file.stp>"));
            let text = std::fs::read_to_string(p).unwrap_or_else(|e| die(e));
            match agcad_geo::step_read::read_step(&text, &Default::default()) {
                Ok((bodies, rep)) => {
                    for (i, b) in bodies.iter().enumerate() {
                        let v = b.validate();
                        println!("body {i}: {} faces, valid={} {:?}", b.faces.len(), v.valid, v.problems);
                    }
                    println!("schema {}, {} entities, length scale {}", rep.schema, rep.entities, rep.length_scale);
                }
                Err(e) => die(e),
            }
        }
        Some("conformance") => {
            let fuzz: usize = flag(&args, "--fuzz").and_then(|s| s.parse().ok()).unwrap_or(0);
            let r = agcad_conformance::run_all(fuzz);
            println!("{}", serde_json::to_string_pretty(&r).unwrap());
            // --capture <file>: append each shrunk failure as a regression test (the file uses
            // the `ok(...)` helper of crates/conformance/tests/fuzz_regressions.rs)
            if let Some(path) = flag(&args, "--capture") {
                let tests: String = r["fuzz"]["failures"].as_array().into_iter().flatten().filter_map(|f| f["regression_test"].as_str()).collect();
                if !tests.is_empty() {
                    use std::io::Write;
                    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).unwrap_or_else(|e| die(e));
                    f.write_all(tests.as_bytes()).unwrap_or_else(|e| die(e));
                    eprintln!("captured {} regression test(s) into {path}", r["fuzz"]["failures"].as_array().map_or(0, Vec::len));
                }
            }
            if r["failed"].as_u64().unwrap_or(0) > 0 { std::process::exit(1); }
        }
        Some("improve") => {
            let code = agcad_improve::cli(&args[2..]);
            std::process::exit(code);
        }
        Some(other) => die(format!("unknown subcommand '{other}'")),
    }
}
