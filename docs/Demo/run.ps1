param(
    [ValidateSet('run', 'car', 'mvp', 'primitives', 'suite', 'constraints', 'harness', 'diagnostics', 'session', 'test-session', 'engineering', 'test-engineering', 'resident', 'test-resident', 'perf-resident', 'train-proposals', 'train', 'train-gpu', 'build', 'test', 'lint')]
    [string]$Action = 'run',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DemoArgs
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:CARGO_HOME = Join-Path $PSScriptRoot '.cargo-home'
    $vswherePath = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if ((Test-Path -LiteralPath $vswherePath) -and -not $env:GEOMETRY_DEMO_VS_INITIALIZED) {
        $vsInstall = & $vswherePath -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsInstall) {
            Import-Module (Join-Path $vsInstall 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
            Enter-VsDevShell -VsInstallPath $vsInstall -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
            if ($env:VCToolsInstallDir) {
                $env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = Join-Path $env:VCToolsInstallDir 'bin\Hostx64\x64\link.exe'
                $env:GEOMETRY_DEMO_VS_INITIALIZED = '1'
            }
        }
    }
    if ($env:CUDA_PATH) {
        $cudaBin = Join-Path $env:CUDA_PATH 'bin'
        if ($env:PATH.Split(';') -notcontains $cudaBin) { $env:PATH = $cudaBin + ';' + $env:PATH }
    }
    switch ($Action) {
        'resident' {
            & cargo build --release --locked --bin surface_worker
            if ($LASTEXITCODE -ne 0) { throw 'Resident surface worker build failed' }
            & python scripts/resident_demo.py @DemoArgs
        }
        'test-resident' {
            & cargo build --release --locked --bin surface_worker
            if ($LASTEXITCODE -ne 0) { throw 'Resident surface worker build failed' }
            & python scripts/resident_acceptance.py
        }
        'perf-resident' {
            & cargo build --release --locked --bin surface_worker --bin class_a_suite
            if ($LASTEXITCODE -ne 0) { throw 'Surface benchmark backend build failed' }
            & python scripts/resident_benchmark.py
        }
        'engineering' {
            & cargo build --release --locked --bin class_a_suite
            if ($LASTEXITCODE -ne 0) { throw 'Engineering GPU backend build failed' }
            & python scripts/engineering_demo.py @DemoArgs
        }
        'test-engineering' {
            & cargo build --release --locked --bin class_a_suite
            if ($LASTEXITCODE -ne 0) { throw 'Engineering GPU backend build failed' }
            & python scripts/engineering_acceptance.py
        }
        'session' {
            & cargo build --release --locked --bin cad_worker
            if ($LASTEXITCODE -ne 0) { throw 'Persistent CAD worker build failed' }
            & python scripts/session_demo.py @DemoArgs
        }
        'test-session' {
            & cargo build --release --locked --bin cad_worker
            if ($LASTEXITCODE -ne 0) { throw 'Persistent CAD worker build failed' }
            & python -m unittest discover -s scripts -p test_model_session.py -v
        }
        'train-proposals' {
            $env:RAYON_NUM_THREADS = '2'
            # This diagnostic action explicitly trains on CPU. CuModel inference
            # remains on CUDA; training and execution backends are independent.
            & cargo build --manifest-path trainer\Cargo.toml --release --locked --bin probe_policy
            if ($LASTEXITCODE -ne 0) { throw 'Proposal trainer build failed' }
            & .\trainer\target\release\probe_policy.exe output/diagnostics/model
        }
        'diagnostics' {
            & python scripts/diagnostic_suite.py prepare @DemoArgs
            if ($LASTEXITCODE -ne 0) { throw 'Diagnostic preparation failed; run train-proposals first if model is absent' }
            & cargo run --release --locked --bin diagnostic_suite -- output/diagnostics
            if ($LASTEXITCODE -ne 0) { throw 'GPU diagnostic execution failed' }
            & python scripts/diagnostic_suite.py validate
        }
        'build' { & cargo build --release --locked }
        'test'  { & cargo test --release --locked }
        'lint'  { & cargo clippy --release --locked -- -D warnings }
        'run'   { & cargo run --release --locked -- @DemoArgs }
        'car'   { & cargo run --release --locked --bin surface_car -- @DemoArgs }
        'primitives' { & cargo run --release --locked --bin primitive_suite -- @DemoArgs }
        'harness' {
            & python scripts/harness_demo.py prepare
            if ($LASTEXITCODE -ne 0) { throw 'Harness template compilation failed' }
            & cargo run --release --locked --bin gpu_harness -- output/harness
            if ($LASTEXITCODE -ne 0) { throw 'GPU harness execution failed' }
            foreach ($harnessCase in @('geometry_failure', 'numerical_failure', 'invalid_baseline')) {
                & cargo run --release --locked --bin gpu_harness -- "output/harness/$harnessCase"
                if ($LASTEXITCODE -ne 0) { throw "GPU harness $harnessCase fixture failed" }
            }
            & python scripts/harness_demo.py validate
            if ($LASTEXITCODE -ne 0) { throw 'Harness independent validation failed' }
            & python scripts/harness_delivery.py
        }
        'constraints' {
            & python scripts/constraint_demo.py prepare
            if ($LASTEXITCODE -ne 0) { throw 'Constraint graph compilation failed' }
            & cargo run --release --locked --bin constraint_suite -- output/constraints
            if ($LASTEXITCODE -ne 0) { throw 'GPU surface constraint solve failed' }
            & cargo run --release --locked --bin constraint_suite -- output/constraints/random
            if ($LASTEXITCODE -ne 0) { throw 'GPU random constraint solve failed' }
            & cargo run --release --locked --bin constraint_suite -- output/constraints/iteration_limit
            if ($LASTEXITCODE -ne 0) { throw 'GPU iteration limit fixture failed' }
            & cargo run --release --locked --bin constraint_suite -- output/constraints/attachments
            if ($LASTEXITCODE -ne 0) { throw 'GPU attachment constraints failed' }
            foreach ($constraintCase in @('fully_fixed', 'zero_rhs')) {
                & cargo run --release --locked --bin constraint_suite -- "output/constraints/$constraintCase"
                if ($LASTEXITCODE -ne 0) { throw "GPU $constraintCase fixture failed" }
            }
            & python scripts/constraint_demo.py validate
        }
        'suite' {
            & cargo run --release --locked --bin primitive_suite
            if ($LASTEXITCODE -ne 0) { throw 'GPU primitive suite failed' }
            & python scripts/validate_all.py
            if ($LASTEXITCODE -ne 0) { throw 'Baseline validation failed' }
            & python scripts/validate_extensions.py
            if ($LASTEXITCODE -ne 0) { throw 'Curve/solid extension validation failed' }
            & python scripts/validate_operations.py
            if ($LASTEXITCODE -ne 0) { throw 'Modeling operation validation failed' }
            & python scripts/validate_topology.py
            if ($LASTEXITCODE -ne 0) { throw 'Topology/smoothing validation failed' }
            & python scripts/validate_advanced.py
            if ($LASTEXITCODE -ne 0) { throw 'Advanced capability validation failed' }
            & python scripts/validate_motion.py
            if ($LASTEXITCODE -ne 0) { throw 'Placement/motion validation failed' }
            & python scripts/composed_panel.py
        }
        'mvp' {
            & cargo run --release --locked --bin surface_car -- @DemoArgs
            if ($LASTEXITCODE -ne 0) { throw 'GPU policy/modeling failed; STEP export skipped' }
            $surfaceOut = 'output/car'
            for ($i = 0; $i -lt $DemoArgs.Count - 1; $i++) {
                if ($DemoArgs[$i] -eq '--out') { $surfaceOut = $DemoArgs[$i + 1] }
            }
            & python scripts/export_step.py --out $surfaceOut
        }
        'train' { & cargo run --manifest-path trainer\Cargo.toml --release --locked -- @DemoArgs }
        'train-gpu' {
            if ($env:VCToolsInstallDir) { $env:NVCC_CCBIN = Join-Path $env:VCToolsInstallDir 'bin\Hostx64\x64\cl.exe' }
            $env:RAYON_NUM_THREADS = '2'
            $env:NVCC_PREPEND_FLAGS = '--use-local-env'
            & cargo run --manifest-path trainer\Cargo.toml --release --locked --features cuda -- --cuda @DemoArgs
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "cargo $Action failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
