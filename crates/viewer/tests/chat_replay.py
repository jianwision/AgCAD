from pathlib import Path
import json, os, subprocess, tempfile, time, socket, urllib.request as http, urllib.error as errors, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from contextlib import ExitStack

live = "--live" in sys.argv
browser = "--browser" in sys.argv
root = Path(__file__).resolve().parents[3]
requests = []
provider_settings = []
provider_waiting = threading.Event()
release_provider = threading.Event()


class Fixture(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        turn = json.loads(
            next(
                m["content"] for m in reversed(data["messages"]) if m["role"] == "user"
            )
        )
        question = turn["request"]
        requests.append(question)
        provider_settings.append((data["model"], data.get("thinking"), data.get("reasoning_effort")))
        if question == "delayed preview":
            provider_waiting.set()
            if not release_provider.wait(15):
                self.send_error(504)
                return
        if question == "Describe the selection":
            cad = turn["cad"]
            facts = {k: cad[k] for k in ("selected_entity", "selected_face", "extrusion_face")}
            facts["finish_tool"] = any(t["function"]["name"] == "preview_finish" for t in data["tools"])
            raw = json.dumps({"choices": [{"message": {"role": "assistant", "content": json.dumps(facts)}}]}).encode()
        elif "truncated" in question and requests.count(question) == 1:
            raw = b'{"choices":'
        elif "empty twice" in question:
            raw = b""
        else:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "sample",
                        "type": "function",
                        "function": {"name": "preview_sphere", "arguments": "{}"},
                    }
                ],
            }
            if question in ("round this edge by 2 mm","round this edge by 100 mm","chamfer this edge by 1 mm"):
                chamfer = question.startswith("chamfer")
                msg["tool_calls"][0]["function"] = {
                    "name": "preview_edge_chamfer" if chamfer else "preview_edge_round",
                    "arguments": json.dumps({"distance_mm":1} if chamfer else {"scope":"selected","radius_mm":100 if "100" in question else 2}),
                }
            groups={
                "round picked edges":("preview_edge_round",{"scope":"selected","radius_mm":1}),
                "round all edges":("preview_edge_round",{"scope":"all","radius_mm":1}),
                "chamfer picked edges":("preview_edge_chamfer",{"distance_mm":1}),
                "bevel picked vertices":("preview_vertex_bevel",{"distance_mm":1}),
                "distance between vertices":("measure_vertices",{}),
                "offset selected faces":("preview_face_offset",{"distance_mm":2}),
                "make selected faces blue":("preview_finish",{"scope":"face","color":"#2255ff"}),
                "polished silver":("preview_finish",{"scope":"body","color":"#dddddd","roughness":0.02,"metalness":1}),
                "matte silver":("preview_finish",{"scope":"body","color":"#dddddd","roughness":1,"metalness":1}),
            }
            if question in groups:
                name,arguments=groups[question]
                msg["tool_calls"][0]["function"]={"name":name,"arguments":json.dumps(arguments)}
            if question == "extend 3 cm":
                msg["tool_calls"][0]["function"] = {
                    "name": "preview_extrusion",
                    "arguments": json.dumps({"distance": "3 cm"}),
                }
            raw = json.dumps(
                {"choices": [{"message": msg, "finish_reason": "tool_calls"}]}
            ).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(raw)


def run():
    with ExitStack() as stack:
        tmp = stack.enter_context(tempfile.TemporaryDirectory())
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {
            **os.environ,
            "AGCAD_VIEW_DATA": tmp,
            "AGCAD_VIEW_BIND": f"127.0.0.1:{port}",
        }
        if not live:
            fixture = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
            threading.Thread(target=fixture.serve_forever, daemon=True).start()
            stack.callback(fixture.server_close)
            stack.callback(fixture.shutdown)
            env.update(
                AGCAD_ENV_FILE="absent-fixture-env",
                AGCAD_LLM_API_STYLE="deepseek",
                AGCAD_LLM_MODEL="deepseek-flash",
                AGCAD_LLM_KEY_ENV="VIEW_KEY",
                VIEW_KEY="fixture",
                AGCAD_LLM_API_BASE=f"http://127.0.0.1:{fixture.server_port}",
            )
        server = subprocess.Popen(
            [root / "target/release/agcad-viewer.exe"],
            env=env,
            cwd=root,
            creationflags=subprocess.CREATE_NO_WINDOW,
            stdout=subprocess.DEVNULL,
        )
        stack.callback(server.wait, 10)
        stack.callback(server.terminate)
        for _ in range(100):
            try:
                with http.urlopen(url, timeout=1):
                    break
            except OSError:
                time.sleep(0.1)

        def raw(path, data):
            b = isinstance(data, bytes)
            return http.urlopen(
                http.Request(
                    url + path,
                    data=data if b else json.dumps(data).encode(),
                    headers={
                        "Content-Type": (
                            "application/octet-stream" if b else "application/json"
                        )
                    },
                ),
                timeout=110,
            )

        def post(path, data):
            try:
                with raw(path, data) as r:
                    return json.load(r)
            except errors.HTTPError as e:
                raise ValueError((e.code, e.read().decode()[:1000])) from e

        def new(name):
            return post("/models/new", {"name": name})["id"]

        def offer(model):
            return post("/offer", {"model": model})["id"]

        def command(s, op, data=None):
            return post("/session/" + s, {"op": op, "data": data or {}})

        def ask(s, text, hist=None):
            r = command(s, "ask", {"prompt": text, "history": hist or []})
            print(text, r["answer"][:200], r["summary"].get("bounding_box"), flush=True)
            return r

        def size(r):
            assert r["preview"] and r["checks"]["status"] == "ACCEPTED", r["answer"]
            return r["summary"]["bounding_box"]["size"]

        if "--preferences" in sys.argv:
            with http.urlopen(url + "/assistant/options") as response:
                catalog = json.load(response)["models"]
            assert {"deepseek-flash","deepseek-v4-pro"} <= {m["id"] for m in catalog}, catalog
            s = offer(new("Assistant preference test"))
            previous = []
            for model,level in [("deepseek-flash","low"),("deepseek-v4-pro","max")]:
                prompt = "Preview a 12 mm diameter sphere at the origin."
                result = command(s,"ask",{"prompt":prompt,"history":previous,"assistant":{"model":model,"thinking":level}})
                assert result["preview"] and result["checks"]["status"]=="ACCEPTED", result["answer"]
                previous = [{"role":"user","text":prompt},{"role":"assistant","text":result["answer"]}]
                command(s,"discard")
                if not live: assert provider_settings[-1]==(model,{"type":"enabled"},level)
                print("PASS model and thinking level", model,level,flush=True)
            for choice in [{"model":"unlisted-model"},{"thinking":"impossible"}]:
                before = len(requests)
                try: command(s,"ask",{"prompt":"preview sphere","assistant":choice})
                except ValueError as err: assert err.args[0][0] == 400
                else: raise AssertionError("unsupported selection accepted")
                assert len(requests) == before
            assert command(s,"status")["revision"]==0
            return
        if "--multi" in sys.argv:
            from multi_view import check_multi
            check_multi(raw,post,offer,command,browser,url)
            return
        if "--picking" in sys.argv:
            from viewport_picking import check_picking

            check_picking(raw, post, new, offer, command, browser, url)
            return
        if browser:
            from playwright.sync_api import sync_playwright

            pw = stack.enter_context(sync_playwright())
            b = pw.chromium.launch(channel="msedge")
            stack.callback(b.close)
            p = b.new_page()
            p.add_init_script(
                "Object.defineProperty(window,'RTCPeerConnection',{value:undefined})"
            )
            offers = []
            js_errors = []
            p.on(
                "response",
                lambda r: (
                    offers.append(r.json()["id"])
                    if r.url.endswith("/offer") and r.status == 200
                    else None
                ),
            )
            p.on("pageerror", lambda err: js_errors.append(str(err)))
            model = new("Browser sphere")
            p.goto(url + "/#" + model)
            picker = p.get_by_role("combobox",name="Chat model")
            effort = p.get_by_role("combobox",name="Thinking level")
            picker.select_option("deepseek-v4-pro")
            effort.select_option("high")
            p.reload()
            picker.wait_for()
            assert picker.input_value()=="deepseek-v4-pro" and effort.input_value()=="high"
            p.locator("#prompt").fill("preview sphere")
            p.locator("#prompt").press("Enter")
            p.locator("#apply").wait_for()
            if not live: assert provider_settings[-1]==("deepseek-v4-pro",{"type":"enabled"},"high")
            p.locator("#prompt").fill("my unsent draft")
            initial = p.locator(".turn.user").count()
            before_reconnect = len(offers)
            command(offers[-1], "close")
            p.locator("#apply").click()
            p.locator("#review").wait_for(state="hidden", timeout=15000)
            assert p.locator("#prompt").input_value() == "my unsent draft"
            assert p.locator(".turn.user").count() == initial and p.url.endswith(model)
            assert (
                len(offers) == before_reconnect + 1
                and not js_errors
                and command(offers[-1], "status")["revision"] == 0
            )
            p.locator("#prompt").fill("preview sphere again")
            picker.select_option("deepseek-flash")
            effort.select_option("off")
            p.locator("#prompt").press("Enter")
            p.locator("#apply").click()
            p.locator("#review").wait_for(state="hidden")
            assert command(offers[-1], "status")["revision"] == 1
            if not live: assert provider_settings[-1]==("deepseek-flash",{"type":"disabled"},None)
            print(
                "PASS Edge draft, history, expired Apply, explicit re-apply", flush=True
            )
            return
        empty = new("Start empty")
        s = offer(empty)
        if not live:
            sphere_size = size(ask(s, "truncated sphere"))
            assert (
                sphere_size == [20, 20, 20] and requests.count("truncated sphere") == 2
            )
            command(s, "discard")
            try:
                ask(s, "empty twice")
            except ValueError as e:
                assert requests.count("empty twice") == 2
            else:
                raise AssertionError("expected rejection")
            assert command(s, "status")["preview"] is None
            from concurrent.futures import ThreadPoolExecutor

            def conflict(operation):
                try:
                    operation()
                except ValueError as err:
                    assert err.args[0][0] == 409, err
                else:
                    raise AssertionError("expected a preview conflict")

            shape = {"shape": "block", "width": 40, "depth": 30, "height": 10}
            first = command(s, "primitive", shape)["preview_id"]
            command(s, "discard", {"preview_id": first})
            second = command(s, "primitive", shape)["preview_id"]
            assert first != second
            conflict(lambda: command(s, "apply", {"preview_id": first}))
            assert command(s, "status")["preview_id"] == second
            command(s, "apply", {"preview_id": second})
            conflict(lambda: command(s, "apply", {"preview_id": second}))
            assert command(s, "status")["revision"] == 1
            extended_size = size(ask(s, "extend 3 cm"))
            assert extended_size == [40, 30, 40]
            command(s, "discard")
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(command, s, "ask", {"prompt": "delayed preview"})
                try:
                    assert provider_waiting.wait(10)
                    with raw(
                        "/session/" + s + "/image", [0.64, 0.62, 1, 0, 0]
                    ) as frame:
                        stamp = int(frame.headers["x-frame-id"])
                        frame.read()
                    picked = command(
                        s,
                        "pick",
                        {"timestamp": stamp, "image": True, "x": 0.5, "y": 0.5},
                    )
                    assert picked["selection"] is not None
                finally:
                    release_provider.set()
                conflict(pending.result)
            assert command(s, "status")["preview_id"] is None
            print(
                "PASS stale selection, replacement preview, duplicate Apply", flush=True
            )
            if "--mcp" in sys.argv:
                import asyncio
                from mcp_replay import compare

                asyncio.run(compare(root, tmp, sphere_size, extended_size))
            print("PASS injected incomplete and empty responses", flush=True)
            return
        assert size(ask(s, "create a sphere")) == [20, 20, 20]
        command(s, "discard")
        hist = [
            {"role": "user", "text": "create a sphere"},
            {"role": "assistant", "text": "Which diameter do you want?"},
        ]
        assert size(ask(s, "you fill now", hist)) == [20, 20, 20]
        command(s, "discard")
        assert size(
            ask(s, "Create a 12 mm diameter sphere centered at (4, 5, 6) mm.")
        ) == [12, 12, 12]
        command(s, "discard")
        assert not ask(s, "smooth")["preview"]
        command(
            s, "primitive", {"shape": "block", "width": 40, "depth": 30, "height": 10}
        )
        command(s, "apply")
        assert size(ask(s, "extrude 3cm")) == [40, 30, 40]
        command(s, "apply")
        assert size(ask(s, "Extrude the same end face by -5 mm inward.")) == [
            40,
            30,
            35,
        ]
        command(s, "discard")
        r = ask(s, "smooth all edges by 2 mm")
        assert size(r) and r["summary"]["faces"] == 26
        command(s, "discard")
        v = ask(s, "Round just the vertical corners with a 2 mm radius.")
        assert v["preview"] and v["checks"]["status"] == "ACCEPTED"
        command(s, "discard")
        model = post(
            "/models?name=sample.step",
            (Path(__file__).parent / "fixtures/plate.step").read_bytes(),
        )["id"]
        t = offer(model)
        r = ask(t, "extrude 3cm")
        assert not r["preview"] and "face" in r["answer"].lower()
        r = ask(t, "smooth all edges by 2 mm")
        assert size(r) and r["summary"]["faces"] == 26
        command(t, "discard")
        with raw("/session/" + t + "/image", [0.64, 0.62, 1, 0, 0]) as r:
            stamp = int(r.headers["x-frame-id"])
            r.read()
        sel = command(
            t, "pick", {"timestamp": stamp, "image": True, "x": 0.5, "y": 0.5}
        )["selection"]
        print("imported face", sel["entity"], sel["can_extrude"], flush=True)
        assert sel["can_extrude"]
        assert size(ask(t, "extrude 3cm"))
        command(t, "discard")
        assert size(ask(t, "create a sphere")) == [20, 20, 20]
        command(t, "discard")
        assert command(t, "status")["revision"] == 0
        assert size(ask(t, "create a sphere")) == [20, 20, 20]
        applied = command(t, "apply")
        assert (
            applied["model"] != model
            and applied["revision"] == 1
            and command(offer(model), "status")["revision"] == 0
        )
        raised = False
        try:
            command(t, "apply")
        except ValueError:
            raised = True
        assert raised
        print("PASS live HTTP replay", flush=True)


if __name__ == "__main__":
    run()
