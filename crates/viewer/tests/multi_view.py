from math import hypot, pi
from pathlib import Path
from viewport_picking import screen_point

def check_multi(raw,post,offer,command,browser,url):
    plate=(Path(__file__).parent/"fixtures/plate.step").read_bytes()
    def model(): return post("/models?name=Picking.step",plate)["id"]
    doc=model()
    session=offer(doc)
    radius=1.15*hypot(20,14,5)
    def cmd(op,data=None): return command(session,op,data)
    assert cmd("skybox", {"preset":"sunset"})["skybox"] == "sunset"
    try:
        cmd("skybox", {"preset":"nonsense"})
    except ValueError:
        pass
    else:
        raise AssertionError("accepted unknown skybox")
    other = offer(doc)
    assert command(other,"status")["skybox"] == "dark"
    command(other,"close")
    assert cmd("status")["skybox"] == "sunset"
    def pick(pose,p,mode="replace",rect=None):
        with raw("/session/"+session+"/image",pose) as image:
            stamp=int(image.headers["x-frame-id"]);image.read()
        x,y=screen_point(p,radius,pose)
        data={"image":True,"timestamp":stamp,"x":x,"y":y,"mode":mode,"viewport_height":540}
        if rect: data["rect"]=rect
        return cmd("pick",data)
    front=[0,0,1,0,0];back=[pi,0,1,0,0]
    def ask(prompt): return cmd("ask",{"prompt":prompt})
    def discard(proposal): cmd("discard",{"preview_id":proposal["preview_id"]})
    for prompt in ["round picked edges","chamfer picked edges"]:
        p=pick(front,(0,-14,5))
        assert p["selection"]["kind"]=="edge"
        p=pick(back,(0,14,5),"toggle")
        assert len(p["selections"])==2 and p["selections"][0]["index"]!=p["selections"][1]["index"]
        p=pick(back,(0,14,5),"toggle");assert len(p["selections"])==1
        p=pick(front,(20,-14,0),"add");assert len(p["selections"])==2
        prop=ask(prompt);assert prop["summary"]["faces"]==8,prop
        discard(prop)
    prop=ask("round all edges");assert prop["summary"].get("faces")==26,(prop["answer"],prop["summary"]);discard(prop)
    pick(front,(-20,-14,5));pick(front,(20,-14,5),"add")
    distance=ask("distance between vertices");assert "40.000000 mm" in distance["answer"] and not distance["preview_id"],distance
    prop=ask("bevel picked vertices");assert abs(prop["summary"]["volume"]-(11200-2/6))<1e-5;discard(prop)
    pick(front,(0,-14,0));pick(back,(0,14,0),"add")
    prop=ask("offset selected faces");assert prop["summary"]["bounding_box"]["size"]==[40,32,10],prop;discard(prop)
    pick(front,(0,-14,0));pick(back,(0,14,0),"add")
    prop=ask("make selected faces blue");cmd("apply",{"preview_id":prop["preview_id"]})
    assert pick(front,(0,-14,0))["selection"]["appearance"]["color"]=="#2255ff"
    assert pick(back,(0,14,0))["selection"]["appearance"]["color"]=="#2255ff"
    assert pick([0,pi/2,1,0,0],(0,0,5))["selection"]["appearance"]["color"]!="#2255ff"
    cmd("clear_selection");p=pick(front,(0,0,0),"add",[0,0,1,1])
    kinds=[v["kind"] for v in p["selections"]];assert kinds.count("face")==1 and kinds.count("vertex")==4,(kinds,p["selections"])
    p=pick(front,(0,0,0),"remove",[0,0,1,1]);assert not p["selections"]
    print("PASS group fillet, chamfer, vertex bevel and distance, opposite face offsets, finish coverage, box add/remove",flush=True)
    if not browser:return
    from playwright.sync_api import sync_playwright,expect
    with sync_playwright() as pw:
        app=pw.chromium.launch(channel="msedge")
        for live in [False,True]:
            page=app.new_page(viewport={"width":960,"height":720})
            errors=[];page.on("pageerror",lambda err:errors.append(str(err)))
            if not live:page.add_init_script("window.RTCPeerConnection=undefined")
            m=model();page.goto(url+"/#"+m);sel="#video" if live else "#still";page.locator(sel).wait_for(timeout=25000)
            if not live:
                gear = page.get_by_role("button",name="Viewport settings")
                cube = page.locator("#view-cube")
                for width in (360,960):
                    page.set_viewport_size({"width":width,"height":720})
                    c,sky,g = [item.bounding_box() for item in (cube,page.locator("#skybox"),gear)]
                    assert c["x"] < 20 and c["x"]+c["width"] < sky["x"] < g["x"]
                    assert c["y"] == sky["y"] == g["y"] and width-g["x"]-g["width"] < 20
                    gear.click()
                    panel=page.locator("#viewport-settings").bounding_box()
                    assert 0 <= panel["x"] and panel["x"]+panel["width"] <= width
                    page.keyboard.press("Escape")
                    expect(page.locator("#viewport-settings")).to_be_hidden()
                gear.click()
                page.get_by_label("Show view cube").uncheck()
                expect(cube).to_be_hidden()
                page.get_by_label("Reverse scroll zoom").check()
                page.reload()
                page.locator(sel).wait_for()
                expect(cube).to_be_hidden()
                page.mouse.move(480,360)
                with page.expect_request(lambda r:r.url.endswith("/image")) as zoom:
                    page.mouse.wheel(0,120)
                assert zoom.value.post_data_json[2] > 1
                gear.click()
                page.get_by_label("Show view cube").check()
                page.get_by_label("Reverse scroll zoom").uncheck()
                page.keyboard.press("Escape")
                page.reload();page.locator(sel).wait_for()
            def presented():
                if live:
                    page.evaluate("""new Promise(done=>{let n=0;const v=document.querySelector('#video');function tick(){if(++n===6)done(true);else v.requestVideoFrameCallback(tick)}v.requestVideoFrameCallback(tick)})""")
                else:
                    # Camera redraws may overlap; wait for the returned JPEG to decode.
                    page.wait_for_function("document.querySelector('#still').complete")
                    page.wait_for_timeout(130)
            def view(name):
                page.locator("#view-menu summary").click();page.get_by_role("button",name=name,exact=True).click();presented()
            def click_pt(x,y,mods=()):
                for key in mods:page.keyboard.down(key)
                page.mouse.click(x*960,90+y*540)
                for key in reversed(mods):page.keyboard.up(key)
            for name in ["Top","Bottom","Left","Right","Front","Back"]:
                view(name);click_pt(.5,.5);expect(page.locator("#context")).to_contain_text("Face")
            view("Front")
            x,y=screen_point((0,-14,5),radius,front);click_pt(x,y);expect(page.locator("#context")).to_contain_text("Edge")
            view("Back");click_pt(x,y,["Control"]);expect(page.locator("#context")).to_contain_text("2 edges")
            for keys in [["Control","Shift"],["Control"]]:
                for key in keys:page.keyboard.down(key)
                with page.expect_response(lambda r:r.request.method=="POST" and isinstance(r.request.post_data_json,dict) and r.request.post_data_json.get("op")=="pick",timeout=4000) as resp:
                    page.mouse.move(x*960-5,90+y*540-3);page.mouse.down();page.mouse.move(x*960+5,90+y*540+3,steps=5);page.mouse.up()
                assert resp.value.ok,resp.value.text()
                picked=resp.value.json()["selections"]
                assert sum(v["kind"]=="edge" for v in picked)==(1 if len(keys)==2 else 2),(resp.value.request.post_data_json,[(v["kind"],v["index"]) for v in picked])
                for key in reversed(keys):page.keyboard.up(key)
                expect(page.locator("#context")).to_contain_text("Edge" if len(keys)==2 else "2 edges")
            page.locator("#screen").focus();page.keyboard.press("Shift+f");presented()
            page.mouse.wheel(0,-120);presented()
            # A click on the cube changes the view and never becomes a geometry pick.
            page.locator("#view-cube").click(position={"x":54,"y":46})
            expect(page.locator("#context")).to_contain_text("2 edges")
            page.locator("#prompt").fill("round picked edges");page.locator("#prompt").press("Enter");page.locator("#apply").click();expect(page.locator("#review")).to_be_hidden()
            page.reload();page.locator(sel).wait_for(timeout=25000)
            saved = offer(m)
            assert command(saved,"status")["summary"]["faces"]==8
            command(saved,"close")
            # Sample the actual media, including H.264 decode or JPEG decode.
            sky = page.get_by_role("combobox", name="Skybox")
            pixel = """selector => {
                const el=document.querySelector(selector), c=document.createElement('canvas');
                const ctx=c.getContext('2d');
                ctx.drawImage(el,100,100,1,1,0,0,1,1);
                return Array.from(ctx.getImageData(0,0,1,1).data).slice(0,3);
            }"""
            colors = []
            for preset in ["dark","studio","daylight","sunset","dark"]:
                sky.select_option(preset)
                presented()
                if colors:
                    page.wait_for_function(
                        "([sample,selector,old])=>eval(sample)(selector).some((v,i)=>Math.abs(v-old[i])>8)",
                        arg=[pixel,sel,colors[-1]])
                colors.append(page.evaluate(pixel,sel))
            assert max(abs(a-b) for a,b in zip(colors[0],colors[-1])) < 7, colors
            assert all(max(abs(a-b) for a,b in zip(c,d))>8 for i,c in enumerate(colors[:4]) for d in colors[:i]), colors
            sky.select_option("sunset")
            expect(sky).to_be_enabled()
            page.reload();page.locator(sel).wait_for(timeout=25000)
            assert sky.input_value()=="sunset"
            page.wait_for_function(
                "([sample,selector,expected])=>eval(sample)(selector).every((v,i)=>Math.abs(v-expected[i])<7)",
                arg=[pixel,sel,colors[3]])
            backdrop = page.evaluate("""async () => {
                const image = new Image();
                image.src='/skybox/sunset';
                await image.decode();
                const ctx=document.createElement('canvas').getContext('2d');
                ctx.drawImage(image,100,100,1,1,0,0,1,1);
                return Array.from(ctx.getImageData(0,0,1,1).data).slice(0,3);
            }""")
            assert max(abs(a-b) for a,b in zip(backdrop,colors[3]))<8, (backdrop,colors[3])
            sky.select_option("daylight")
            expect(sky).to_be_enabled()
            view("Front")
            # In this view the light direction (-.41,.78,.47) projects to (676,215).
            # Compare the solar disc to a nearby patch of sky in the decoded frame.
            sun_visible="""selector => {
                const ctx=document.createElement('canvas').getContext('2d');
                const rgb=(x,y)=>{
                    ctx.drawImage(document.querySelector(selector),x,y,1,1,0,0,1,1);
                    return ctx.getImageData(0,0,1,1).data;
                };
                const disk=rgb(676,215), sky=rgb(722,215);
                return disk[2]>230 && disk[2]-sky[2]>25;
            }"""
            page.wait_for_function(sun_visible,arg=sel)
            page.mouse.move(390,360);page.keyboard.down("Shift")
            page.mouse.down();page.mouse.move(480,400,steps=5);page.mouse.up();page.keyboard.up("Shift")
            page.mouse.wheel(0,-160);presented()
            assert page.evaluate(sun_visible,sel), "sun moved during pan or zoom"
            view("Back")
            page.wait_for_function("([sample,selector])=>!eval(sample)(selector)",arg=[sun_visible,sel])
            def drop(name, content, target):
                page.locator(target).evaluate("""(el,[name,text])=>{
                    const dt=new DataTransfer();
                    dt.items.add(new File([text],name));
                    for(const type of ['dragover','drop']) el.dispatchEvent(new DragEvent(type,{dataTransfer:dt,bubbles:true,cancelable:true}));
                }""",[name,content])
            source=plate.decode()
            def reset():
                page.locator("#prompt").fill("/new")
                page.locator("#prompt").press("Enter")
                expect(page.locator("#screen video")).to_be_hidden()
                expect(page.locator("#still")).to_be_hidden()
                expect(page.locator("#thread p")).to_have_count(0)
                assert "#" not in page.url
            reset()
            drop("Dropped.STP",source,"#prompt")
            page.locator(sel).wait_for(timeout=25000)
            imported=page.url
            drop("broken.step","broken STEP content","#screen")
            expect(page.locator(".turn.error")).to_be_visible()
            assert page.url==imported and page.locator(sel).is_visible()
            drop("Replacement.step",source,"#screen")
            page.wait_for_function("old=>location.href!==old",arg=imported)
            page.locator(sel).wait_for(timeout=25000)
            # A pending geometry proposal can be abandoned with the same local command.
            page.locator("#prompt").fill("preview sphere")
            page.locator("#prompt").press("Enter")
            page.locator("#review").wait_for()
            reset()
            page.locator("#prompt").fill("preview sphere")
            page.locator("#prompt").press("Enter")
            page.locator("#apply").click()
            expect(page.locator("#review")).to_be_hidden()
            page.locator(sel).wait_for(timeout=25000)
            sky.select_option("studio");expect(sky).to_be_enabled();presented()
            highlights = []
            for finish in ["polished silver","matte silver"]:
                click_pt(.52,.48)
                expect(page.locator("#context")).to_contain_text("Face")
                page.locator("#prompt").fill(finish)
                page.locator("#prompt").press("Enter")
                page.locator("#apply").click()
                expect(page.locator("#review")).to_be_hidden()
                presented()
                # Only sample inside the sphere, excluding the background and silhouette.
                highlights.append(page.evaluate("""selector=>{
                    const c=document.createElement('canvas');c.width=c.height=128;
                    const ctx=c.getContext('2d');
                    ctx.drawImage(document.querySelector(selector),448,28,1024,1024,0,0,128,128);
                    const data=ctx.getImageData(0,0,128,128).data;
                    const values=[];
                    for(let y=0;y<128;y++)for(let x=0;x<128;x++){
                        if(Math.hypot(x-64,y-64)<30)values.push(data[4*(y*128+x)+1]);
                    }
                    values.sort((a,b)=>a-b);
                    return values[Math.floor(values.length*.97)];
                }""",sel))
            assert highlights[0]>highlights[1]+12,highlights
            assert not errors,errors
            page.close();print("PASS six views, box deselection, fit, zoom, multi-edge apply, four GPU skyboxes, STEP drop and /new",sel,flush=True)
        app.close()
