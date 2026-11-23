const $ = id => document.getElementById(id);
const video = $('video'), still = $('still'), screen = $('screen');
const prompt = $('prompt'), popup = $('popup'), thread = $('thread');
const skybox = $('skybox');
let changingSky = false;
try {
  const saved = localStorage.getItem('skybox');
  if ([...skybox.options].some(option => option.value === saved)) skybox.value = saved;
} catch {}
screen.dataset.skybox = skybox.value;
const initialPose = () => [.64, .62, 1, 0, 0];
let generation = 0, session = '', model = '', state = {empty: true}, pose = initialPose();
let peer, channel, videoFrame, imageFrame, cameraSeq = 0, drag;
let busy = false, turns = [], objectURL, imageBusy = false, imageDirty = false;
let videoAttempt = 0, videoTimer, needsVideo = true;

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw Object.assign(Error((await response.text()) || response.statusText), {status:response.status});
  return response.json();
}
function post(path, data) {
  return api(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
}
const chatModel = $('chat-model'), thinking = $('thinking');
let chatModels = [];
function setThinkingLevels(preferred) {
  const levels = chatModels.find(model => model.id === chatModel.value)?.thinking || [];
  const labels = {off:'Thinking off', low:'Low', high:'High', max:'Max'};
  thinking.replaceChildren(...levels.map(level => new Option(labels[level], level)));
  if (levels.includes(preferred)) thinking.value = preferred;
  thinking.hidden = !levels.length;
}
function chatPreference() {
  const choice = {};
  if (chatModel.value) choice.model = chatModel.value;
  if (thinking.value) choice.thinking = thinking.value;
  return choice;
}
function saveChatPreference() {
  try { localStorage.setItem('agcad.chat.preference', JSON.stringify(chatPreference())); } catch {}
}
const assistantReady = api('/assistant/options').then(({models}) => {
  chatModels = models;
  let saved;
  try { saved = JSON.parse(localStorage.getItem('agcad.chat.preference')); } catch {}
  chatModel.replaceChildren(...models.map(model => new Option(model.label, model.id)));
  if (models.some(model => model.id === saved?.model)) chatModel.value = saved.model;
  chatModel.hidden = !models.length;
  setThinkingLevels(saved?.thinking);
}).catch(() => {}); // Older servers still accept chat with their configured defaults.
chatModel.onchange = () => { setThinkingLevels(thinking.value); saveChatPreference(); };
thinking.onchange = saveChatPreference;
let reconnecting;
async function recoverSession(expired) {
  if (!model || session !== expired) return;
  if (reconnecting?.session === expired) return reconnecting.promise;
  const turn = generation;
  const promise = (async () => {
    const r = await post('/offer', {model, skybox:skybox.value});
    if (turn !== generation || session !== expired) {
      await post('/session/' + r.id, {op:'close'});
      return;
    }
    closeVideo();
    still.hidden=true;
    imageFrame=null;
    needsVideo=true;
    cameraSeq=0;
    const lostPreview=!!state.preview;
    session=r.id;
    showState(r.state);
    render();
    if (lostPreview) message('The preview expired. Request a new preview to apply the edit.');
  })();
  reconnecting={session:expired,promise};
  try { await promise; } finally { if (reconnecting?.promise===promise) reconnecting=null; }
}
async function command(op, data = {}) {
  const local=session;
  try { return await post('/session/' + local, {op, data}); }
  catch (error) {
    if (error.status !== 410) throw error;
    await recoverSession(local);
    if (op==='status') return state;
    throw Object.assign(Error('Reconnected. Review the current part and retry the action.'),{status:410});
  }
}
function appendTurn(role, text) {
  const line = document.createElement('p');
  line.className = 'turn ' + role;
  line.setAttribute('aria-label', role === 'user' ? 'You' : 'Assistant');
  line.textContent = text;
  thread.append(line);
}
function openPopup() {
  popup.hidden = !thread.childElementCount && !state.preview;
  thread.scrollTop = thread.scrollHeight;
}
function remember(role, text) {
  turns.push({role, text});
  if (turns.length > 40) {
    turns.shift();
    thread.firstElementChild?.remove();
  }
  appendTurn(role, text);
  try {
    if (model) sessionStorage.setItem('chat:' + model, JSON.stringify(turns));
  } catch {}
  openPopup();
}
function message(text) {
  if (text && thread.lastElementChild?.textContent !== text) remember('error', text);
}
function restoreHistory() {
  try {
    turns = JSON.parse(sessionStorage.getItem('chat:' + model) || '[]');
    for (const {role, text} of turns) appendTurn(role, text);
  } catch { turns = []; }
}
function controls() {
  const text = prompt.value.trim();
  $('ask').disabled = !text || (text.toLowerCase() !== '/new' && (busy || !!state.preview));
  skybox.disabled = busy || changingSky;
  chatModel.disabled = thinking.disabled = busy;
  $('new').disabled = busy;
  $('working').textContent = busy ? 'Working…' : '';
}
function showState(next) {
  if (next.model && next.model !== model) {
    model = next.model;
    window.history.replaceState(null, '', '#' + model);
  }
  state = next;
  screen.dataset.empty = String(!!next.empty);
  $('navigation').hidden=!!next.empty;
  drawCube();
  $('review').hidden = !next.preview;
  const entity = next.selection;
  $('context').hidden = !entity;
  if (next.selections?.length>1) {
    const names={edge:'edges',vertex:'vertices',face:'faces'};
    $('context').textContent=Object.entries(names).map(([k,plural])=>{
      const n=next.selections.filter(v=>v.kind===k).length;
      return n ? n+' '+(n===1?k:plural) : '';
    }).filter(Boolean).join(' \u00b7 ');
  } else if (entity) {
    const label = entity.name || entity.curve || entity.surface?.kind;
    $('context').textContent = entity.label + (label ? ' \u00b7 ' + label : '');
  } else {
    $('context').textContent = '';
  }
  if (next.empty) {
    still.hidden = true;
    video.hidden = true;
  }
  controls();
}
function closeVideo() {
  ++videoAttempt;
  clearTimeout(videoTimer);
  if (peer) {
    peer.onconnectionstatechange = null;
    peer.ontrack = null;
    peer.close();
  }
  peer = null;
  channel = null;
  videoFrame = null;
  video.hidden = true;
  video.srcObject = null;
}
function leave() {
  ++generation;
  clearTimeout(pendingClick?.timer);
  pendingClick = null;
  pickQueue = Promise.resolve();
  cancelDrag();
  touches.clear();
  pinch = cubeDrag = null;
  viewMenu.open = false;
  changingSky = false;
  screen.classList.remove('drop-target');
  closeVideo();
  if (session) navigator.sendBeacon('/session/' + session, new Blob([JSON.stringify({op:'close'})], {type:'application/json'}));
  session = ''; model = ''; pose = initialPose(); cameraSeq = 0; drag = null; needsVideo = true;
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = null;
  still.removeAttribute('src');
  still.hidden = true;
  imageFrame = null;
  turns = [];
  thread.replaceChildren();
  popup.hidden = true;
  busy = false;
  prompt.value = '';
  showState({empty:true});
}
function blank() {
  leave();
  window.history.replaceState(null, '', location.pathname);
  prompt.focus();
}
async function openModel(id, turn, replace = false) {
  const result = await post('/offer', {model:id, skybox:skybox.value});
  if (generation !== turn) {
    await post('/session/' + result.id, {op:'close'});
    return;
  }
  if (replace) leave();
  model = id;
  session = result.id;
  window.history.replaceState(null, '', '#' + id);
  restoreHistory();
  showState(result.state);
  render();
}
function render() {
  if (!session || state.empty) return;
  if (video.hidden) requestImage();
  if (needsVideo) {
    needsVideo = false;
    connectVideo();
  }
}
async function requestImage(){
  imageDirty=true;if(imageBusy)return;
  imageBusy=true;
  try{
    while(imageDirty){
      imageDirty=false;if(!session||state.empty||!video.hidden)break;
      const turn=generation, local=session;
      const result=await fetch('/session/'+local+'/image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(pose)});
      if(turn!==generation||local!==session)continue;
      if(!result.ok){if(result.status===410){await recoverSession(local);continue}if(result.status===409){imageDirty=true;continue}throw Error(await result.text())}
      const stamp=Number(result.headers.get('x-frame-id')),url=URL.createObjectURL(await result.blob());
      const decoded=new Image();decoded.src=url;
      try{await decoded.decode()}catch(err){URL.revokeObjectURL(url);throw err}
      if(turn!==generation||local!==session||state.empty||!video.hidden){URL.revokeObjectURL(url);continue}
      const previous=objectURL;objectURL=url;still.src=url;imageFrame=stamp;still.hidden=false;
      if(previous)URL.revokeObjectURL(previous);

    }
  }catch(err){if(video.hidden&&session)message(err.message)}
  finally{imageBusy=false;if(imageDirty)requestImage()}
}
function useImages(pc){
  if(peer!==pc)return;
  closeVideo();requestImage();
}
async function connectVideo(){
  closeVideo();const turn=generation,attempt=videoAttempt;
  const current=()=>turn===generation && attempt===videoAttempt;
  if(!window.RTCPeerConnection||!video.requestVideoFrameCallback)return;
  try{
    const config=await api('/ice');if(!current())return;
    const pc=new RTCPeerConnection(config);peer=pc;
    const dc=pc.createDataChannel('camera',{ordered:false,maxRetransmits:0});channel=dc;
    dc.onopen=()=>{if(current())dc.send([++cameraSeq,...pose].join(' '))};
    pc.addTransceiver('video',{direction:'recvonly'});
    pc.ontrack=e=>{if(current()){video.srcObject=new MediaStream([e.track]);video.play().catch(()=>useImages(pc))}};
    pc.onconnectionstatechange=()=>{if(current()&&['failed','disconnected'].includes(pc.connectionState))useImages(pc)};
    await pc.setLocalDescription(await pc.createOffer());
    if(pc.iceGatheringState!=='complete')await new Promise(resolve=>{
      const done=()=>{clearTimeout(timer);pc.removeEventListener('icegatheringstatechange',check);resolve()};
      const check=()=>{if(pc.iceGatheringState==='complete')done()};
      const timer=setTimeout(done,4000);pc.addEventListener('icegatheringstatechange',check);check();
    });
    if(!current())return;
    const answer=await post('/offer',{model,session,sdp:pc.localDescription.sdp});
    if(!current())return;
    await pc.setRemoteDescription({type:'answer',sdp:answer.sdp});
    videoTimer=setTimeout(()=>{if(current()&&video.hidden)useImages(pc)},8000);
  }catch{if(current()){closeVideo();requestImage()}}
}
if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(function presented(_,info){
  if(peer && !state.empty && info.rtpTimestamp!=null && video.readyState>=2){
    clearTimeout(videoTimer);videoFrame=info.rtpTimestamp;video.hidden=false;still.hidden=true;

  }
  video.requestVideoFrameCallback(presented);
});
function camera(){
  drawCube();
  if(channel?.readyState==='open')channel.send([++cameraSeq,...pose].join(' '));
  if(video.hidden)requestImage();
}
const cube=$('view-cube'), box=$('pick-box'), viewMenu=$('view-menu');
const showCube = $('show-cube'), reverseZoom = $('reverse-zoom');
try {
  const saved = JSON.parse(localStorage.getItem('agcad.viewport.settings'));
  showCube.checked = saved?.showCube !== false;
  reverseZoom.checked = saved?.reverseZoom === true;
} catch {}
cube.hidden = !showCube.checked;
$('viewport-settings').addEventListener('change', () => {
  cube.hidden = !showCube.checked;
  try {
    localStorage.setItem('agcad.viewport.settings', JSON.stringify({
      showCube:showCube.checked, reverseZoom:reverseZoom.checked
    }));
  } catch {}
});
const cubeSides=[['Left',-1,0,0],['Right',1,0,0],['Front',0,-1,0],['Back',0,1,0],['Bottom',0,0,-1],['Top',0,0,1]];
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
let cubeHits=[], cubeDrag=null, pickQueue=Promise.resolve(), pendingClick=null, holdTimer=null, pinch=null;
const touches=new Map();
function axes(){
  const [yaw,pitch]=pose, c=Math.cos(yaw), s=Math.sin(yaw), v=Math.cos(pitch), h=Math.sin(pitch);
  return [[c,s,0],[-s*h,c*h,v],[s*v,-c*v,h]];
}
const dot=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0);
function orient(x,y,z){
  pose[0]=x===0&&y===0?0:Math.atan2(x,-y);
  pose[1]=Math.atan2(z,Math.hypot(x,y)); camera();
}
function drawCube(){
  const ctx=cube.getContext('2d'), basis=axes();
  ctx.setTransform(2,0,0,2,0,0);ctx.clearRect(0,0,108,96);cubeHits=[];
  for(const [name,...n] of cubeSides){
    const facing=dot(n,basis[2]);if(facing<0.0001)continue;
    const axis=n.findIndex(x=>x!==0),a=(axis+1)%3,b=(axis+2)%3;
    const corners=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([u,v])=>{const p=n.slice();p[a]=u;p[b]=v;
      return {p,x:54+24*dot(p,basis[0]),y:46-24*dot(p,basis[1])};});
    const path=new Path2D();corners.forEach((p,i)=>i?path.lineTo(p.x,p.y):path.moveTo(p.x,p.y));path.closePath();
    const light=Math.round(50+28*facing+8*n[2]);ctx.fillStyle='rgb('+[light,light+13,light+20]+')';ctx.fill(path);
    ctx.strokeStyle='#9aaeba';ctx.lineWidth=1;ctx.stroke(path);
    if(facing>.28){ctx.fillStyle='#edf5f8';ctx.font='10px Segoe UI, sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(name,54+24*dot(n,basis[0]),46-24*dot(n,basis[1]));}
    cubeHits.push({corners,n,path});
  }
}
function orbit(dx,dy){pose[0]-=dx*.008;pose[1]=clamp(pose[1]+dy*.008,-Math.PI/2,Math.PI/2);camera();}
cube.onpointerdown=e=>{cube.setPointerCapture(e.pointerId);cubeDrag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,moved:false};};
cube.onpointermove=e=>{if(!cubeDrag)return;const g=cubeDrag;
  if(Math.hypot(e.clientX-g.startX,e.clientY-g.startY)>3)g.moved=true;
  if(g.moved)orbit(e.clientX-g.x,e.clientY-g.y);g.x=e.clientX;g.y=e.clientY;};
cube.onpointerup=e=>{const g=cubeDrag;cubeDrag=null;if(!g||g.moved)return;const b=cube.getBoundingClientRect();
  const x=(e.clientX-b.left)*108/b.width,y=(e.clientY-b.top)*96/b.height;
  for(const hit of cubeHits){const p=hit.corners.find(c=>Math.hypot(x-c.x,y-c.y)<5);if(p){orient(...p.p);return;}}
  const ctx=cube.getContext('2d');ctx.save();ctx.resetTransform();
  const hit=cubeHits.find(h=>ctx.isPointInPath(h.path,x,y));ctx.restore();if(hit)orient(...hit.n);};
cube.onpointercancel=()=>cubeDrag=null;
function menuAction(label,fn){const button=document.createElement('button');button.textContent=label;button.type='button';button.onclick=()=>{viewMenu.open=false;screen.focus();fn();};$('views').append(button);}
for(const [label,...normal] of cubeSides)menuAction(label,()=>orient(...normal));
menuAction('Fit model',()=>fit(false));menuAction('Fit selection',()=>fit(true));
function viewportPoint(x,y){
  const b=screen.getBoundingClientRect(),w=Math.min(b.width,b.height*16/9),h=w*9/16;
  return {x:(x-b.left-(b.width-w)/2)/w,y:(y-b.top-(b.height-h)/2)/h,w,h};
}
function zoomAt(x,y,factor,dx=0,dy=0){
  const v=viewportPoint(x,y),old=pose[2];pose[2]=clamp(old*factor,.05,100);const ratio=pose[2]/old;
  pose[3]=(2*v.x-1)*(1-ratio)+ratio*pose[3]+2*dx/v.w;
  pose[4]=(1-2*v.y)*(1-ratio)+ratio*pose[4]-2*dy/v.h;camera();
}
async function fit(selection){
  if(!session||state.empty)return;
  const turn=generation;try{await pickQueue;const {bounds}=await command('fit_bounds',{selection});if(turn!==generation||!bounds)return;
    const [lo,hi]=bounds,center=lo.map((v,i)=>(v+hi[i])/2),span=lo.map((v,i)=>hi[i]-v),[right,up]=axes();
    const width=dot(right.map(Math.abs),span)*9/16,height=dot(up.map(Math.abs),span);
    pose[2]=clamp(1.8/Math.max(width,height,.018),.05,100);
    pose[3]=-dot(right,center)*9/16*pose[2];pose[4]=-dot(up,center)*pose[2];camera();
  }catch(err){message(err.message);}
}
function queuePick(op,data){
  const turn=generation,local=session;
  pickQueue=pickQueue.then(async()=>{if(turn!==generation||session!==local)return;
    try{const r=await command(op,data);if(turn===generation){showState(r);render();}}catch(err){if(turn===generation)message(err.message);}
  });
}
function framePick(g,x,y){
  const start=viewportPoint(g.ox,g.oy),end=viewportPoint(x,y),image=video.hidden,timestamp=image?imageFrame:videoFrame;
  if(timestamp==null)return;
  const data={timestamp,image,x:start.x,y:start.y,viewport_height:start.h,mode:g.mode};
  if(g.box && g.moved){
    const rect=[start.x,start.y,end.x,end.y].map(v=>clamp(v,0,1));data.rect=rect;
    data.x=rect[0];data.y=rect[1];if(data.mode==='toggle')data.mode='add';
  }else if(start.x<0||start.x>1||start.y<0||start.y>1)return;
  else if(data.mode==='touch_box')data.mode='toggle';
  queuePick('pick',data);
}
function drawBox(g,x,y){box.hidden=false;box.dataset.sign=g.mode==='remove'||g.subtract?'\u2212':'+';
  Object.assign(box.style,{left:Math.min(x,g.ox)+'px',top:Math.min(y,g.oy)+'px',width:Math.abs(x-g.ox)+'px',height:Math.abs(y-g.oy)+'px'});}
function touchPair(){const [a,b]=touches.values();return {x:(a.x+b.x)/2,y:(a.y+b.y)/2,d:Math.hypot(a.x-b.x,a.y-b.y)};}
function cancelDrag(){clearTimeout(holdTimer);box.hidden=true;drag=null;}
function flushClick(){if(pendingClick){clearTimeout(pendingClick.timer);pendingClick.pick();pendingClick=null;}}
screen.oncontextmenu=e=>e.preventDefault();
screen.ondragstart=e=>e.preventDefault();
screen.onpointerdown=e=>{
  if(!session||state.empty)return;
  e.preventDefault();screen.focus({preventScroll:true});popup.hidden=true;viewMenu.open=false;screen.setPointerCapture(e.pointerId);
  if(e.pointerType==='touch'){
    touches.set(e.pointerId,{x:e.clientX,y:e.clientY});if(touches.size>=2){cancelDrag();pinch=touchPair();return;}
  }
  const multi=e.ctrlKey||e.metaKey;
  if(multi)flushClick();
  drag={id:e.pointerId,x:e.clientX,y:e.clientY,ox:e.clientX,oy:e.clientY,box:multi,mode:multi?(e.shiftKey?'remove':'toggle'):'replace',
    pan:!multi&&(e.shiftKey||e.button!==0),moved:false};
  if(e.pointerType==='touch'){const g=drag;holdTimer=setTimeout(()=>{if(drag===g&&!g.moved){
    g.box=true;g.mode='touch_box';
    const v=viewportPoint(g.ox,g.oy),image=video.hidden,timestamp=image?imageFrame:videoFrame;
    if(timestamp!=null)command('pick',{inspect:true,image,timestamp,x:v.x,y:v.y,viewport_height:v.h})
      .then(r=>{if(drag===g){g.subtract=r.selected;if(g.moved)drawBox(g,g.x,g.y);}}).catch(()=>{});
  }},450);}
};
screen.onpointermove=e=>{
  if(touches.has(e.pointerId))touches.set(e.pointerId,{x:e.clientX,y:e.clientY});
  if(pinch && touches.size>=2){const next=touchPair();if(pinch.d>1)zoomAt(pinch.x,pinch.y,next.d/pinch.d,next.x-pinch.x,next.y-pinch.y);pinch=next;return;}
  const g=drag;if(!g||g.id!==e.pointerId)return;
  if(!g.moved && Math.hypot(e.clientX-g.ox,e.clientY-g.oy)<=3)return;
  g.moved=true;clearTimeout(holdTimer);
  const dx=e.clientX-g.x,dy=e.clientY-g.y;g.x=e.clientX;g.y=e.clientY;
  if(g.box)drawBox(g,g.x,g.y);else if(g.pan){const v=viewportPoint(g.x,g.y);pose[3]+=2*dx/v.w;pose[4]-=2*dy/v.h;camera();}else orbit(dx,dy);
};
screen.onpointerup=e=>{
  touches.delete(e.pointerId);if(pinch){if(touches.size<2)pinch=null;cancelDrag();return;}
  const g=drag;cancelDrag();if(!g || g.id!==e.pointerId || g.pan || (g.moved&&!g.box))return;
  if(g.mode==='replace'){
    if(pendingClick && Math.hypot(e.clientX-pendingClick.x,e.clientY-pendingClick.y)<5){
      clearTimeout(pendingClick.timer);pendingClick=null;pose=initialPose();camera();return;}
    flushClick();const turn=generation, image=video.hidden,stamp=image?imageFrame:videoFrame;
    const v=viewportPoint(e.clientX,e.clientY);
    if(stamp==null||v.x<0||v.x>1||v.y<0||v.y>1)return;
    const pick=()=>{if(turn===generation)queuePick('pick',{timestamp:stamp,image,x:v.x,y:v.y,viewport_height:v.h});};
    pendingClick={x:e.clientX,y:e.clientY,pick,timer:setTimeout(flushClick,230)};
  }else framePick(g,e.clientX,e.clientY);
};
screen.onpointercancel=e=>{touches.delete(e.pointerId);pinch=null;cancelDrag();};
screen.addEventListener('wheel', e => {
  if (state.empty) return;
  e.preventDefault();
  const delta = reverseZoom.checked ? e.deltaY : -e.deltaY;
  zoomAt(e.clientX, e.clientY, Math.exp(delta * .001));
}, {passive:false});
screen.onkeydown=e=>{if(state.empty)return;if(e.key==='Escape'){flushClick();queuePick('clear_selection',{});}else if(e.key.toLowerCase()==='f'){e.preventDefault();flushClick();fit(e.shiftKey);}};

skybox.onchange = async () => {
  const preset = skybox.value, turn = generation;
  const previous = screen.dataset.skybox;
  changingSky = true;
  controls();
  try {
    if (session) await command('skybox', {preset});
    if (turn !== generation) return;
    screen.dataset.skybox = preset;
    try { localStorage.setItem('skybox', preset); } catch {}
    render();
  } catch (err) {
    if (turn === generation) {
      skybox.value = screen.dataset.skybox = previous;
      message(err.message);
    }
  } finally {
    if (turn === generation) { changingSky = false; controls(); }
  }
};

// File drops can land on the composer or on the cube as well as on the model.
function hasFiles(event) { return event.dataTransfer?.types.includes('Files'); }
window.addEventListener('dragover', event => {
  if (!hasFiles(event)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = busy ? 'none' : 'copy';
  screen.classList.toggle('drop-target', !busy);
});
window.addEventListener('dragleave', event => {
  if (!event.relatedTarget) screen.classList.remove('drop-target');
});
window.addEventListener('drop', async event => {
  screen.classList.remove('drop-target');
  if (!hasFiles(event)) return;
  event.preventDefault();
  if (busy) return;
  const files = event.dataTransfer.files;
  if (files.length !== 1 || !/\.(step|stp)$/i.test(files[0].name)) {
    message('Drop one .step or .stp file.');
    return;
  }
  const file = files[0];
  const turn = generation;
  busy = true;
  controls();
  try {
    const result = await api('/models?name=' + encodeURIComponent(file.name), {method:'POST', body:file});
    if (turn !== generation) return;
    await openModel(result.id, turn, true);
  } catch (err) { if (turn === generation) message(err.message); }
  finally { if (turn === generation) { busy = false; controls(); } }
});

$('new').onclick = blank;
prompt.oninput = controls;
prompt.onfocus = openPopup;
prompt.onclick = openPopup;
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('viewport-settings').matches(':popover-open')) {
    popup.hidden = true;
    screen.focus({preventScroll:true});
  }
});
async function send() {
  if (prompt.value.trim().toLowerCase() === '/new') { blank(); return; }
  const submitted = generation;
  flushClick();await Promise.all([pickQueue, assistantReady]);
  if (submitted !== generation) return;
  const text = prompt.value.trim();
  if (!text || busy || state.preview) return;
  const assistant = chatPreference();
  const turn = generation;
  busy = true;
  controls();
  try {
    if (!session) {
      const result = await post('/models/new', {name:'Untitled part'});
      if (turn !== generation) return;
      await openModel(result.id, turn);
    }
    if (turn !== generation) return;
    const history = turns.filter(t => t.role !== 'error').slice(-12);
    remember('user', text);
    prompt.value = '';
    const response = await command('ask', {prompt:text, history, assistant});
    if (turn !== generation) return;
    showState(response);
    remember('assistant', response.answer || response.preview);
    render();
  } catch (err) {
    if (turn === generation) {
      if (err.status===410 && !prompt.value) prompt.value=text;
      message(err.message);
    }
  }
  finally {
    if (turn === generation) { busy = false; controls(); }
  }
}
$('composer').onsubmit = event => { event.preventDefault(); send(); };
prompt.onkeydown = event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    send();
  }
};
for (const op of ['current','proposed','apply','discard']) {
  $(op).onclick = async () => {
    if (busy) return;
    const turn = generation;
    busy = true;
    controls();
    try {
      const result = await command(op, {preview_id:state.preview_id});
      if (turn !== generation) return;
      showState(result);
      render();
      if (op === 'apply') remember('assistant', 'Applied to the part.');
      if (op === 'discard') remember('assistant', 'Edit discarded.');
    } catch (err) { if (turn === generation) message(err.message); }
    finally { if (turn === generation) { busy = false; controls(); } }
  };
}
setInterval(async () => {
  if (!session || busy) return;
  const local = session, previous = state;
  try {
    const r = await command('status');
    if (local === session && previous === state && !busy &&
        (r.model !== model || r.revision !== state.revision || r.preview_id !== state.preview_id)) {
      showState(r);
      render();
    }
  } catch (err) { if (local === session) message(err.message); }
}, 5000);
async function start() {
  const id = location.hash.slice(1);
  leave();
  if (id) {
    busy = true;
    controls();
    const turn = generation;
    try { await openModel(id, turn); }
    catch (err) { if (turn === generation) message(err.message); }
    finally { if (turn === generation) { busy = false; controls(); } }
  }
}
window.addEventListener('pagehide', leave);
window.addEventListener('pageshow', e => { if (e.persisted) start(); });
window.addEventListener('hashchange', start);
start();
