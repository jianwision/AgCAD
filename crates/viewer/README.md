# agcad-viewer

The server hosts a shared STEP library on a Windows workstation with an NVIDIA GPU. Each open browser has its own native D3D11 camera and H.264 WebRTC stream. Visitors can select exact CAD faces, inspect STEP colors and material designations, and preview finish or geometry proposals. Apply saves a revision for everyone using the model.

## Running the server

Install the GStreamer MSVC x86-64 runtime and development SDK with the D3D11, NVENC, WebRTC, ICE and DTLS plugins. Build from an x64 Visual Studio developer shell with the GStreamer bin directory on PATH and its SDK root in GSTREAMER_1_0_ROOT_MSVC_X86_64.

    cargo build --release -p agcad-viewer
    target\release\agcad-viewer.exe

Open http://localhost:8090 on the host or use the workstation address from another browser on the LAN. The process listens on all interfaces by default. AGCAD_VIEW_BIND changes the address and port. The executable can run under a process supervisor with the project directory as its working directory. The model library defaults to agcad-models in that directory. AGCAD_VIEW_DATA selects a different persistent directory.

The viewer opens on an empty workspace. Describe a part in chat to preview its first solid. Press Enter to send or Shift+Enter for a new line. Escape or a viewport click hides the conversation. Click the chat box to reopen it. Type `/new` or click + to clear the viewport and start a fresh conversation. Applied parts remain available at their saved URLs. Drop one `.step` or `.stp` file anywhere in the viewer to open it.

Click the orientation cube or choose a standard view from its menu. Drag to orbit, Shift-drag or right-drag to pan, and scroll or pinch to zoom. With the viewport focused, F fits the model and Shift+F fits the selection. The skybox dropdown changes the rendering environment for this browser.

Click a face, edge or vertex to select it. Ctrl-click (Cmd-click on macOS) toggles individual items. Hold that modifier and drag to add visible items in a rectangle, or also hold Shift to remove them. On touchscreens, hold an item to toggle it. Hold and drag from a selected item to remove items in a rectangle, or from an unselected item to add them. Escape clears the selection.

Each proposal offers Current, Proposed, Apply and Discard. Body and face finish overrides are stored with the source file. Uploaded assemblies use the bodies decoded by the native STEP reader. Geometry proposals operate on a newly created part or a single imported solid or sheet. Rectangular solids support fillets and chamfers on groups of edges, corner bevels and planar face offsets. The browser can also display workstation-rendered images when WebRTC is unavailable.

## DeepSeek

The two menus below the chat input select the model and thinking level for the next message. The browser remembers both choices. The built-in DeepSeek choices are Flash and V4 Pro, with Off, Low, High and Max thinking. New browsers start with the server's configured model and thinking off.

Set `AGCAD_LLM_MODELS` to a comma-separated list to customize the alternatives to the configured default. Custom chat-completions endpoints expose their configured models. A gateway implementing DeepSeek's thinking parameters can opt in with `AGCAD_LLM_API_STYLE=deepseek`.

Copy .env.example from the project root to .env and enter the workstation account's DEEPSEEK_API_KEY. The server rereads this file when a browser submits a prompt. Process environment variables take precedence. AGCAD_ENV_FILE selects an alternate file. The default model is deepseek-flash. AGCAD_LLM_MODEL and AGCAD_LLM_API_BASE can select another chat-completions API. The base URL should exclude /chat/completions.

The server sends recent chat turns and current CAD facts to the provider. Chat completions uses function calling for first solids, geometry edits, operation signatures and selected-face finishes. Inspection questions can receive a text answer. The server validates previews and returns a rejected tool diagnostic to the model for one correction. CAD rebuilds and Apply run on the workstation. A server-side localhost AgCAD bridge can also be configured with AGCAD_LLM_BRIDGE when no API key is set.

## Network access

Allow the browser to reach the listening port and WebRTC ICE candidates on the workstation. For access across networks, serve the page through HTTPS and provide an accessible TURN service in AGCAD_TURN_SERVER (turn://user:password@host:port). AGCAD_STUN_SERVER accepts stun://host:port. These are host environment settings. The browser receives the configured TURN credentials to establish the media connection. The library is shared by everyone with network access to this server, including the ability to submit requests using its configured model account.
