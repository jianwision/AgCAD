import json
import time
from math import cos, sin, hypot
from pathlib import Path


def screen_point(offset, radius, pose=(.64, .62, 1, 0, 0)):
    x, y, z = offset
    yaw, pitch, zoom, pan_x, pan_y = pose
    right = cos(yaw) * x + sin(yaw) * y
    up = (-sin(yaw) * x + cos(yaw) * y) * sin(pitch) + z * cos(pitch)
    return ((1 + right / radius * zoom * 9 / 16 + pan_x) / 2,
            (1 - up / radius * zoom - pan_y) / 2)


def check_picking(raw, post, new, offer, command, browser, url):
    plate = post("/models?name=plate.step", (Path(__file__).parent / "fixtures/plate.step").read_bytes())["id"]
    block = new("Pick a block")
    native = offer(block)
    def create(session, shape):
        preview = command(session, "primitive", shape)["preview_id"]
        command(session, "apply", {"preview_id": preview})
    create(native, {"shape": "block", "width": 40, "depth": 30, "height": 10})

    def frame(session, pose):
        with raw(f"/session/{session}/image", pose) as response:
            return int(response.headers["x-frame-id"]), response.read()
    def pick(session, stamp, xy):
        result = command(session, "pick", {"image": True, "timestamp": stamp, "x": xy[0], "y": xy[1], "viewport_height": 540})
        return result["selection"]

    pose = [.64, .62, 1, 0, 0]
    for session, half_y, origin in [(native, 15, (0, 0, 5)), (offer(plate), 14, (20, 14, 5))]:
        radius = 1.15 * hypot(20, half_y, 5)
        def xy(point):
            return screen_point(point, radius, pose)
        stamp, plain = frame(session, pose)
        for kind, target in [("vertex", (20, -half_y, 5)), ("edge", (0, -half_y, 5))]:
            selection = pick(session, stamp, xy(target))
            assert selection and selection["kind"] == kind, (target, selection)
            assert selection["body"] and isinstance(selection["index"], int)
            if kind == "edge":
                assert selection["curve"] == "Line"
            facts = json.loads(command(session, "ask", {"prompt":"Describe the selection"})["answer"])
            assert facts["selected_entity"] == selection
            assert facts["selected_face"] is None and facts["extrusion_face"] is None
            assert not facts["finish_tool"]
            actual = selection["point_mm"]
            assert all(abs(actual[k] - (delta + base)) < .02 for k, delta, base in zip("xyz", target, origin)), actual
            marked_stamp, marked = frame(session, pose)
            assert marked != plain, "selection highlight missing on GPU frame"
            assert pick(session, marked_stamp, (.02, .02)) is None
            stamp, cleared = frame(session, pose)
            assert cleared == plain, "selection leaked into cleared GPU frame"
        # The back corner projects inside the body. Its foreground face wins.
        assert pick(session, stamp, xy((-20, half_y, -5)))["kind"] == "face"
        assert pick(session, stamp, xy((0, 0, 5)))["kind"] == "face"
        print("PASS vertex, line, face, occlusion, highlight and clear", origin, flush=True)

    for session, model, half_y, volume in [(native,block,15,12000),(offer(plate),plate,14,11200)]:
        def pick_front():
            stamp,_=frame(session,pose)
            return pick(session,stamp,screen_point((0,-half_y,5),1.15*hypot(20,half_y,5)))
        original = pick_front()
        assert original["kind"]=="edge" and len(original["endpoints_mm"])==2
        try:
            command(session,"ask",{"prompt":"round this edge by 100 mm"})
        except ValueError as err:
            assert "adjacent face width" in str(err), err
        else:
            raise AssertionError("Oversized fillet accepted")
        assert command(session,"status")["preview"] is None
        for question in ("round this edge by 2 mm","chamfer this edge by 1 mm"):
            assert pick_front()["index"]==original["index"]
            preview=command(session,"ask",{"prompt":question})
            assert preview["summary"]["faces"]==7 and preview["preview_id"],preview
            if question.startswith("round"):
                command(session,"discard",{"preview_id":preview["preview_id"]})
                assert command(session,"status")["summary"].get("faces",6)==6
            else:
                command(session,"apply",{"preview_id":preview["preview_id"]})
                reopened=command(offer(model),"status")
                assert abs(reopened["summary"]["volume"]-(volume-20))<1e-4
        print("PASS preview, discard, oversize, apply and reopen",model,flush=True)
    cylinder = offer(new("Pick a circular edge"))
    create(cylinder, {"shape": "cylinder", "diameter": 30, "height": 10})
    pose2 = [.64, .62, 1.6, -.08, .07]
    stamp, _ = frame(cylinder, pose2)
    angle = -1.1
    rim = (15*cos(angle), 15*sin(angle), 5)
    curved = pick(cylinder, stamp, screen_point(rim, 1.15*hypot(15,5), pose2))
    assert curved["kind"] == "edge" and curved["curve"] == "Circle", curved
    print("PASS circular edge after pan and zoom", flush=True)
    if browser:
        from playwright.sync_api import sync_playwright, expect
        with sync_playwright() as pw:
            app = pw.chromium.launch(channel="msedge")
            for live in (False, True):
                page = app.new_page(viewport={"width": 960, "height": 720})
                if not live:
                    page.add_init_script("window.RTCPeerConnection = undefined")
                ui_plate = post("/models?name=ui.step", (Path(__file__).parent / "fixtures/plate.step").read_bytes())["id"]
                page.goto(url + "/#" + ui_plate)
                media = "#video" if live else "#still"
                page.locator(media).wait_for(timeout=25000)
                for kind, offset in [("Edge", (0,-14,5)), ("Vertex", (20,-14,5))]:
                    x,y = screen_point(offset, 1.15*hypot(20,14,5))
                    page.mouse.click(x*960, 90+y*540)
                    expect(page.locator("#context")).to_contain_text(kind)
                    # Read the decoded GPU output at the picked point in both transports.
                    page.wait_for_function("""([selector,x,y]) => {
                      const canvas=document.createElement('canvas');
                      canvas.width=canvas.height=25;
                      const ctx=canvas.getContext('2d');
                      ctx.drawImage(document.querySelector(selector),x*1920-12,y*1080-12,25,25,0,0,25,25);
                      const pixels=ctx.getImageData(0,0,25,25).data;
                      for(let i=0;i<pixels.length;i+=4)
                        if(pixels[i]>205 && pixels[i+1]>100 && pixels[i+1]<205 && pixels[i+2]<90) return true;
                      return false;
                    }""", arg=[media,x,y], timeout=12000)
                page.mouse.click(40,120)
                expect(page.locator("#context")).to_be_hidden()
                x,y=screen_point((0,-14,5),1.15*hypot(20,14,5))
                for action,question in [("discard","round this edge by 2 mm"),("apply","chamfer this edge by 1 mm")]:
                    # Discard can complete before its replacement frame reaches the browser.
                    # Retry only a stale-frame rejection, after another image is presented.
                    deadline = time.monotonic()+5
                    def is_pick(res):
                        return (res.request.method=="POST" and "/session/" in res.url and
                                isinstance(res.request.post_data_json,dict) and res.request.post_data_json.get("op")=="pick")
                    while True:
                        old_src = page.locator("#still").get_attribute("src")
                        with page.expect_response(is_pick) as picked:
                            page.mouse.click(x*960,90+y*540)
                        if picked.value.ok:
                            break
                        assert time.monotonic()<deadline and picked.value.text() in (
                            "Displayed image expired. Click again.", "Displayed revision changed. Click again.",
                            "Displayed frame expired. Click again."), picked.value.text()
                        if live:
                            page.evaluate("new Promise(resolve=>document.querySelector('#video').requestVideoFrameCallback(()=>resolve(true)))")
                        else:
                            page.wait_for_function("src=>document.querySelector('#still').src!==src",arg=old_src)
                    expect(page.locator("#context")).to_contain_text("Edge")
                    page.locator("#prompt").fill(question)
                    page.locator("#prompt").press("Enter")
                    page.locator("#"+action).click()
                    expect(page.locator("#review")).to_be_hidden()
                page.reload()
                page.locator(media).wait_for(timeout=25000)
                assert abs(command(offer(ui_plate),"status")["summary"]["volume"]-11180)<1e-4
                page.close()
                print("PASS browser picking and gold GPU pixels on", media, flush=True)
            app.close()
