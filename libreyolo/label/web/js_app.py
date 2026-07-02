"""JS: boot + chrome wiring, class palette/editor, sidebar list, stats, insights."""

PART = r"""// ---- API ----
async function jget(u){ const r = await fetch(u); if(!r.ok) throw new Error((await r.json()).error||r.status); return r.json(); }

async function init(){
  wireChrome();
  wireHome();
  const d = await jget("/api/dataset");
  let atHome=false; try{ atHome = sessionStorage.getItem("ll-home")==="1"; }catch(e){}
  if(!d.open || atHome){ showHome(d); return; }
  await enterLabeler(d);
}
async function enterLabeler(d){
  const gen = loadSeq;                 // re-entrancy guard: a newer open supersedes this one
  DS = d;
  CLSCOL = Array.isArray(d.colors) ? d.colors.slice() : [];   // custom per-class colors from the sidecar
  try{ sessionStorage.removeItem("ll-home"); }catch(e){}
  if(active>=(DS.names||[]).length) active=0;        // stale class index from a prior project
  $("#dsname").textContent = DS.name || (DS.root||"").split(/[\\/]/).filter(Boolean).pop() || "dataset";
  $("#dsname").title = DS.linked ? ("Linked project - images stay in " + (DS.source||"their folder") + "; nothing is written there") : (DS.root||"");
  const cb=$("#classesbtn"); if(cb) cb.style.display = (DS.writable!==false) ? "" : "none";
  renderPalette();
  const isObb = DS.task==="obb";   // OBB projects: only the oriented-box tool makes valid rows
  const isSeg = DS.task==="segment";   // segment projects: polygons only (a box row corrupts seg labels)
  { const tO=$("#toolObb"); if(tO) tO.style.display=isObb?"":"none";
    const tb=$("#toolBox"); if(tb) tb.style.display=(isObb||isSeg)?"none":"";
    const tp=$("#toolPoly"); if(tp) tp.style.display=isObb?"none":""; }
  setTool(isObb ? "obb" : (isSeg ? "poly" : "box"));
  const imgs = (await jget("/api/images")).images;
  if(gen!==loadSeq) return;            // another project opened while we were fetching
  IMAGES = imgs;
  hideHome();                          // only leave home once images are actually in hand
  renderList();
  renderStats();
  resizeCanvas();
  initAssist();
  if(IMAGES.length) await load(0);
  else { idx=-1; imgOk=false; stageMsg = "No images found - check the dataset paths"; setSave("no images"); draw(); }
  // Project instructions (Settings): shown once on open, unless a warning banner
  // (read-only reason, ...) is already up. DS===d detects a superseding open
  // (loadSeq can't: load(0) itself bumps it).
  if(DS===d && DS.instructions && $("#banner").style.display!=="flex")
    banner("Instructions: " + DS.instructions);
}
function wireChrome(){
  $("#classchip").onclick = togglePicker;
  $("#submitbtn").onclick = submitImage;
  $("#classesbtn").onclick = openClassEdit;
  $("#ceclose").onclick = closeClassEdit;
  $("#cecancel").onclick = closeClassEdit;
  $("#cesave").onclick = saveClassEdit;
  $("#ceadd").onclick = addClassRow;
  $("#classedit").onclick = (e)=>{ if(e.target.id==="classedit") closeClassEdit(); };
  $("#psearch").oninput = e=> filterClasses(e.target.value);
  $("#helpbtn").onclick = toggleHelp;
  $("#themebtn").onclick = toggleTheme;
  applyTheme(document.documentElement.classList.contains("light") ? "light" : "dark");
  $("#insbtn").onclick = openInsights;
  $("#insclose").onclick = closeInsights;
  $("#insights").onclick = (e)=>{ if(e.target.id==="insights") closeInsights(); };
  $("#radarclose").onclick = closeRadar;
  $("#radar").onclick = (e)=>{ if(e.target.id==="radar") closeRadar(); };
  $("#mapclose").onclick = closeMap;
  $("#mapmodal").onclick = (e)=>{ if(e.target.id==="mapmodal") closeMap(); };
  $("#mapbtn").onclick = openMap;
  $("#boostbtn").onclick = runBoost;
  $("#homebtn").onclick = backToHome;
  $("#brandhome").onclick = backToHome;   // clicking the logo also returns to Projects (the conventional escape hatch)
  $("#sharebtn").onclick = toggleShare;
  document.addEventListener("click", e=>{ const pop=$("#sharepop");
    if(pop && pop.classList.contains("show") && !pop.contains(e.target) && !e.target.closest("#sharebtn")) pop.classList.remove("show"); });
  wireMap();
  $("#toolBox").onclick = ()=> setTool("box");
  $("#toolSeg").onclick = ()=> setTool("seg");
  $("#toolPoly").onclick = ()=> setTool("poly");
  $("#toolObb").onclick = ()=> setTool("obb");
  $("#toolAi").onclick = ()=> prelabelCurrent();
  $("#toolFit").onclick = ()=>{ fit(); draw(); };
  $("#toolZin").onclick = ()=> zoomBy(1.25);
  $("#toolZout").onclick = ()=> zoomBy(1/1.25);
  document.querySelectorAll("#filter button").forEach(b=> b.onclick = ()=>{
    document.querySelectorAll("#filter button").forEach(x=>x.classList.remove("on"));
    b.classList.add("on"); listFilter = b.dataset.f; selSet = null; renderList();
  });
  const si=$("#imgsearch"); if(si) si.oninput = e=>{ imgQuery=(e.target.value||"").trim().toLowerCase(); renderList(); };
}
function setTool(t){
  // Task gating: an OBB project authors oriented boxes ONLY (a plain box or free
  // polygon would write a wrong-shaped row), and only OBB projects author them
  // (a 4-corner quad in a taskless dataset makes the file OBB-ambiguous ->
  // read-only on the next load). Applies to shortcuts too, not just the toolbar.
  const isObb = DS && DS.task==="obb";
  if(isObb && t!=="obb") return;
  if(!isObb && t==="obb" && DS) return;
  if(DS && DS.task==="segment" && t==="box") return;   // a 5-field box row corrupts a segment dataset
  if(t==="seg" && !(assist && assist.sam)) return;
  if(tool==="poly" && t!=="poly" && polyDraft) cancelPolyDraft();
  tool = t;
  const tb=$("#toolBox"), ts=$("#toolSeg"), tp=$("#toolPoly"), tO=$("#toolObb");
  if(tb) tb.classList.toggle("on", t==="box");
  if(ts) ts.classList.toggle("on", t==="seg");
  if(tp) tp.classList.toggle("on", t==="poly");
  if(tO) tO.classList.toggle("on", t==="obb");
  if(t==="seg") banner("Smart segment: click an object - or drag a box around it - and SAM outlines it. B for box tool.");
  else if(t==="poly") banner("Polygon: click to add points; click the first point or press Enter to close. Esc cancels, Backspace removes the last point.");
  else if(t==="obb") banner("Oriented box: drag to draw, then drag the round handle above it to rotate (hold Shift to snap to 15°).");
  else $("#banner").style.display="none";
}
function toggleHelp(){
  const h=$("#help"); const opening = h.style.display!=="flex";
  h.style.display = opening ? "flex" : "none";
  rebindTarget = null;
  if(opening) renderHelp();
}

// ---- class picker ----
function renderPalette(){
  const pal = $("#pal"); pal.innerHTML = "";
  (DS.names||[]).forEach((nm,i)=>{
    const c = document.createElement("button");
    c.className = "pclass" + (i===active?" on":""); c.dataset.i = i;
    const k = i<9 ? (i+1) : (i===9 ? 0 : "");
    c.innerHTML = `<span class="sw" style="background:${color(i)}"></span>`+
      `<span class="pn">${esc(nm)}</span>`+ (k!==""?`<span class="pk">${k}</span>`:"");
    c.onclick = ()=>{ setActive(i); closePicker(); };
    pal.appendChild(c);
  });
  markPalette();
  renderLabelBar();
}
function markPalette(){
  const chip=$("#classchip");
  if(!(DS.names||[]).length){   // no classes yet: invite the user to add the first one
    chip.classList.add("empty");
    chip.innerHTML = `<span class="sw" style="background:#6a7280"></span><span>Add a class</span><span class="cc-h">start</span>`;
    return;
  }
  chip.classList.remove("empty");
  const nm = (DS.names&&DS.names[active]!=null)?DS.names[active]:active;
  chip.innerHTML = `<span class="sw" style="background:${color(active)}"></span>`+
    `<span>${esc(nm)}</span><span class="cc-h">class</span>`;
  document.querySelectorAll("#pal .pclass").forEach(c=> c.classList.toggle("on", +c.dataset.i===active));
  document.querySelectorAll("#labelbar .lchip").forEach(c=> c.classList.toggle("on", c.dataset.i!=null && +c.dataset.i===active));
}
function setActive(i){
  if(i<0||i>=(DS.names||[]).length) return;
  active = i; markPalette();
  if(!editable || (DS && !DS.writable)) return;   // view-only: don't reclassify
  if(selBoxes.size){   // reclassify the whole multi-selection in one undo step
    const targets=[...selBoxes].filter(bi=>boxes[bi] && boxes[bi].cls!==i);
    if(targets.length){ pushUndo(); targets.forEach(bi=>boxes[bi].cls=i); markDirty(); draw(); }
  }
  else if(sel>=0 && boxes[sel] && boxes[sel].cls!==i){ pushUndo(); boxes[sel].cls = i; markDirty(); draw(); }
  else if(selPoly>=0 && polys[selPoly] && polys[selPoly].cls!==i){ pushUndo(); polys[selPoly].cls = i; markDirty(); draw(); }
}
function selectAllBoxes(){
  if(!editable || (DS && !DS.writable) || !boxes.length) return;
  selBoxes = new Set(boxes.map((_,i)=>i));
  sel = boxes.length===1 ? 0 : -1; selPoly = -1;
  banner(`${boxes.length} box${boxes.length>1?"es":""} selected - press a number to reclassify, or Del to delete.`);
  draw();
}
function openPicker(){ $("#picker").classList.add("show"); const s=$("#psearch"); s.value=""; filterClasses(""); s.focus(); }
function closePicker(){ $("#picker").classList.remove("show"); }
function togglePicker(){
  if(!(DS.names||[]).length){ openClassEdit(); return; }   // nothing to pick yet -> add classes
  $("#picker").classList.contains("show")?closePicker():openPicker();
}
function filterClasses(q){ q=(q||"").toLowerCase();
  document.querySelectorAll("#pal .pclass").forEach(c=>{
    const nm=(DS.names[+c.dataset.i]||"").toLowerCase(); c.style.display = nm.includes(q)?"":"none"; }); }

// ---- bottom label bar (always-visible class picker with hotkey numbers) ----
function renderLabelBar(){
  const bar=$("#labelbar"); if(!bar) return;
  const names=(DS&&DS.names)||[];
  bar.innerHTML="";
  if(!names.length){
    const b=document.createElement("button"); b.className="lchip add";
    b.textContent="+ Add a class"; b.onclick=openClassEdit; bar.appendChild(b); return;
  }
  names.forEach((nm,i)=>{
    const k = i<9 ? (i+1) : (i===9 ? 0 : "");
    const c=document.createElement("button");
    c.className="lchip"+(i===active?" on":""); c.dataset.i=i;
    c.innerHTML=`<span class="sw" style="background:${color(i)}"></span>`+
      `<span class="ln">${esc(nm)}</span>`+(k!==""?`<span class="lk">${k}</span>`:"");
    c.onclick=()=>{ setActive(i); };   // sets the active class; reclasses a selected box too
    bar.appendChild(c);
  });
  const add=document.createElement("button");
  add.className="lchip add"; add.title="Add or rename classes"; add.textContent="+";
  add.onclick=openClassEdit; bar.appendChild(add);
}
// ---- Submit: save this image, then jump to the next unlabeled one ----
async function submitImage(){
  if(!imgOk) return;
  if((DS && !DS.writable) || !editable){ banner("This image/dataset is read-only - nothing to submit."); return; }
  const here=idx;
  if(await save()===false) return;   // save() already surfaced why (degenerate shape, conflict, ...)
  setRowStatus(here, (boxes.length+polys.length)?"labeled":"empty");
  const vis=visibleIds();
  const hasUnlabeled = vis.some(id=> id!==here && IMAGES[id] && IMAGES[id].status==="unlabeled");
  if(hasUnlabeled){ nextUnlabeled(1); }
  else { banner("All images labeled. Use Export (top bar) to save your dataset as YOLO, COCO or Pascal VOC."); }
}
// ---- Regions panel (Outliner): every annotation on this image, select/hover/delete ----
let regionsSig="";
function renderRegions(){
  const list=$("#rplist"), mn=document.querySelector("main"); if(!list||!mn) return;
  const n=boxes.length+polys.length;
  const sig=n+"|"+boxes.map(b=>b.cls).join(",")+"|P"+polys.map(p=>p.cls).join(",")
    +"|s"+sel+"|"+[...selBoxes].sort((a,b)=>a-b).join(",")+"|p"+selPoly
    +"|e"+(editable?1:0)+"|w"+((DS&&DS.writable)?1:0);
  if(sig===regionsSig) return; regionsSig=sig;
  const cnt=$("#rpcount"); if(cnt) cnt.textContent=n;
  if(!n){ list.innerHTML=`<div class="rp-empty">No labels yet - draw a box.</div>`; return; }
  const canEdit = editable && !(DS && !DS.writable);
  list.innerHTML="";
  boxes.forEach((b,i)=>{
    const nm=(DS.names&&DS.names[b.cls]!=null)?DS.names[b.cls]:b.cls;
    const row=document.createElement("div");
    row.className="rprow"+((i===sel||selBoxes.has(i))?" on":"");
    row.innerHTML=`<span class="sw" style="background:${color(b.cls)}"></span>`+
      `<span class="rpn">${esc(String(nm))}</span><span class="rpi">#${i+1}</span>`+
      (canEdit?`<span class="rpx" title="Delete">&times;</span>`:"");
    row.onclick=(e)=>{ if(e.target.classList.contains("rpx")) return;
      selBoxes=new Set([i]); sel=i; selPoly=-1; setActive(b.cls); draw(); };
    row.onmouseenter=()=>{ if(hover!==i){ hover=i; draw(); } };
    row.onmouseleave=()=>{ if(hover===i){ hover=-1; draw(); } };
    const x=row.querySelector(".rpx");
    if(x) x.onclick=(e)=>{ e.stopPropagation(); pushUndo(); boxes.splice(i,1); sel=-1; selBoxes.clear(); markDirty(); draw(); };
    list.appendChild(row);
  });
  polys.forEach((p,i)=>{
    const nm=(DS.names&&DS.names[p.cls]!=null)?DS.names[p.cls]:p.cls;
    const row=document.createElement("div");
    row.className="rprow"+((i===selPoly)?" on":"");
    row.innerHTML=`<span class="sw" style="background:${color(p.cls)}"></span>`+
      `<span class="rpn">${esc(String(nm))} · ${(DS&&DS.task==="obb")?"obb":"polygon"}</span><span class="rpi">#${i+1}</span>`+
      (canEdit?`<span class="rpx" title="Delete">&times;</span>`:"");
    row.onclick=(e)=>{ if(e.target.classList.contains("rpx")) return;
      selPoly=i; sel=-1; selBoxes.clear(); setActive(p.cls); draw(); };
    const x=row.querySelector(".rpx");
    if(x) x.onclick=(e)=>{ e.stopPropagation(); pushUndo(); polys.splice(i,1); selPoly=-1; markDirty(); draw(); };
    list.appendChild(row);
  });
}

// ---- class editor (rename existing + append new; never delete/reorder) ----
function openClassEdit(){
  ceRows = (DS.names||[]).map(n=>({name:n, isNew:false}));
  if(!ceRows.length) ceRows.push({name:"", isNew:true});
  $("#ceerr").textContent="";
  renderClassEdit();
  $("#classedit").classList.add("show");
  const first=$("#celist .ce-in"); if(first) first.focus();
}
function closeClassEdit(){ $("#classedit").classList.remove("show"); $("#ceerr").textContent=""; }
function renderClassEdit(){
  const wrap=$("#celist"); wrap.innerHTML="";
  ceRows.forEach((row,i)=>{
    const div=document.createElement("div"); div.className="ce-row";
    div.innerHTML = `<span class="sw" style="background:${color(i)}"></span>`+
      `<input class="ce-in" data-i="${i}" value="${esc(row.name)}" placeholder="class name" spellcheck="false">`+
      (row.isNew ? `<button class="ce-rm" data-i="${i}" title="Remove">&times;</button>`
                 : `<span class="ce-fixed" title="Existing class - rename only">#${i}</span>`);
    wrap.appendChild(div);
  });
  wrap.querySelectorAll(".ce-in").forEach(inp=> inp.oninput = e=>{ ceRows[+e.target.dataset.i].name = e.target.value; });
  wrap.querySelectorAll(".ce-rm").forEach(b=> b.onclick = ()=>{ ceRows.splice(+b.dataset.i,1); renderClassEdit(); });
}
function addClassRow(){ ceRows.push({name:"", isNew:true}); renderClassEdit();
  const all=document.querySelectorAll("#celist .ce-in"); if(all.length) all[all.length-1].focus(); }
async function saveClassEdit(){
  const names = ceRows.map(r=>r.name.trim());
  const err=$("#ceerr");
  if(names.some(n=>!n)){ err.textContent="Class names can't be empty."; return; }
  if(new Set(names.map(n=>n.toLowerCase())).size !== names.length){ err.textContent="Class names must be unique."; return; }
  if(names.length < (DS.nc||0)){ err.textContent="You can't remove existing classes here."; return; }
  const btn=$("#cesave"), t=btn.textContent; btn.disabled=true; btn.textContent="Saving…";
  try{
    const r=await fetch("/api/classes",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({names, epoch:(DS&&DS.epoch)||0})});
    const d=await r.json();
    if(!r.ok){ err.textContent=d.error||"Could not save classes."; return; }
    DS.names=d.names; DS.nc=d.nc;
    if(active>=DS.names.length) active=0;
    renderPalette();
    if(imgOk) draw();        // box labels may now read a renamed class
    closeClassEdit();
  }catch(e){ err.textContent="Could not save classes."; }
  finally{ btn.disabled=false; btn.textContent=t; }
}

// ---- sidebar list ----
function passFilter(im){
  if(im.status==="deleted") return false;
  if(imgQuery && !(im.name||"").toLowerCase().includes(imgQuery)) return false;
  if(selSet) return selSet.has(im.id);
  if(listFilter==="todo") return im.status==="unlabeled";
  if(listFilter==="review") return im.status==="suggested";
  return true;
}
let listLimit = 400;   // windowed sidebar: big datasets (10k+ images) must not build 10k DOM nodes
function renderList(){
  const el = $("#list"); el.innerHTML = "";
  const visible = IMAGES.filter(passFilter);
  const curPos = visible.findIndex(im=>im.id===idx);
  if(curPos>=listLimit) listLimit = curPos+200;   // the open image is always in the window
  visible.slice(0, listLimit).forEach(im=>{
    const r = document.createElement("button");
    r.className = "card" + (im.id===idx?" sel":""); r.dataset.id = im.id;
    r.innerHTML = `<img class="thumb" loading="lazy" src="/api/thumb/${im.id}?e=${(DS&&DS.epoch)||0}" alt="">`+
      `<span class="meta"><span class="fn">${esc(im.name)}</span>`+
      `<span class="st"><i class="dot ${im.status}"></i>${im.status}</span></span>`;
    r.onclick = ()=> load(im.id);
    el.appendChild(r);
  });
  if(visible.length > listLimit){
    const more = document.createElement("button");
    more.className = "card"; more.style.justifyContent = "center";
    more.textContent = `Show more (${visible.length-listLimit} hidden)`;
    more.onclick = ()=>{ listLimit += 400; renderList(); };
    el.appendChild(more);
    if(window.IntersectionObserver){   // auto-extend as the user scrolls near the end
      new IntersectionObserver((es,obs)=>{ if(es.some(x=>x.isIntersecting)){ obs.disconnect(); listLimit += 400; renderList(); } },
        {root:el, rootMargin:"600px"}).observe(more);
    }
  }
  if(!visible.length){ const lbl = listFilter==="todo"?"to-do ":listFilter==="review"?"review ":"";
    el.innerHTML = `<div class="empty">No ${lbl}images${imgQuery?" match the filter":""}</div>`; }
}
function markRow(){
  document.querySelectorAll("#list .card").forEach(r=> r.classList.toggle("sel", +r.dataset.id===idx));
  const cur = document.querySelector(`#list .card[data-id="${idx}"]`);
  if(cur) cur.scrollIntoView({block:"nearest"});
  else if(idx>=0 && IMAGES[idx] && passFilter(IMAGES[idx])) scheduleRelist();   // beyond the window -> extend it
}
let _relistTimer = null;
function scheduleRelist(){   // coalesce many status changes (e.g. a bulk run) into one re-render
  if(_relistTimer) return;
  _relistTimer = setTimeout(()=>{ _relistTimer=null; renderList(); }, 80);
}
function setRowStatus(id, status){
  IMAGES[id] && (IMAGES[id].status = status);
  const card = document.querySelector(`#list .card[data-id="${id}"]`);
  if(card){
    const st = card.querySelector(".st");
    if(st) st.innerHTML = `<i class="dot ${status}"></i>${status}`;
    if(IMAGES[id] && !passFilter(IMAGES[id])) card.remove();
  } else if(IMAGES[id] && passFilter(IMAGES[id])){
    scheduleRelist();   // a hidden row now matches the active filter -> bring it into the sidebar
  }
  updateProgress();
}

// ---- dataset health (live class distribution of accepted labels) ----
let statsTimer = null;
function scheduleStats(){ clearTimeout(statsTimer); statsTimer = setTimeout(renderStats, 500); }
async function renderStats(){
  const el = $("#stats"), tc = $("#traincta"); if(!el) return;
  const myEpoch = DS && DS.epoch;   // debounced from save()/fixDuplicate(): a late result must not render into a since-switched project
  let s; try{ s = await jget("/api/stats"); }catch(e){ return; }
  if(!DS || DS.epoch!==myEpoch) return;   // project switched mid-fetch -> drop stale histogram/counts/train CTA
  if(!s.boxes){
    el.innerHTML = `<div class="sh"><span>Dataset health</span></div>`+
      `<div class="none">No labels yet - Auto-label, then accept</div>`;
    if(tc) tc.style.display="none";
    return;
  }
  const max = s.classes.length ? s.classes[0][1] : 1;
  const rows = s.classes.slice(0,8).map(c=>{
    const i = (DS.names||[]).indexOf(c[0]); const col = i>=0?color(i):"#6a7280";
    const w = Math.max(5, Math.round(100*c[1]/max));
    return `<div class="statrow"><span class="sw" style="background:${col}"></span>`+
      `<span class="nm">${esc(c[0])}</span>`+
      `<span class="barwrap"><span class="bar" style="width:${w}%;background:${col}"></span></span>`+
      `<span class="ct">${c[1]}</span></div>`;
  }).join("");
  el.innerHTML = `<div class="sh"><span>Dataset health</span><b>${s.labeled}/${s.total} &middot; ${s.boxes} boxes</b></div>${rows}`;
  if(tc){
    if(s.labeled>0 && DS && DS.yaml){
      tc.style.display="flex";
      tc.innerHTML = `<span class="t-l">${ICO_CHECK}<span>${s.labeled} labeled</span></span>`+
        `<button class="t-cmd" id="exportcta"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12M8 11l4 4 4-4"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>Export</button>`;
      const b=$("#exportcta"); if(b) b.onclick=openExport;
    } else tc.style.display="none";
  }
}

// ---- dataset insights (dimensions + duplicates + leakage + geometry) ----
let lastInsights = null, lastQuality = null, lastStats = null;
async function openInsights(){
  $("#insights").classList.add("show");
  $("#insbody").innerHTML = `<div class="iload">Analyzing images…</div>`;
  const myEpoch = DS && DS.epoch;   // bind to the project: late results from a since-switched project would render stale ids
  let d, q, st;
  try{ [d, q, st] = await Promise.all([jget("/api/insights"),
        jget("/api/quality").catch(()=>null), jget("/api/stats").catch(()=>null)]); }
  catch(e){ $("#insbody").innerHTML = `<div class="iload">Could not compute insights.</div>`; return; }
  if(!DS || DS.epoch!==myEpoch) return;   // project switched mid-analysis -> drop stale results (Fix would target wrong files)
  lastInsights = d; lastQuality = q; lastStats = st;
  renderInsights(d, q);
}
function closeInsights(){ $("#insights").classList.remove("show"); }
function renderInsights(d, q){
  if(q===undefined) q = lastQuality;
  const W=d.width, H=d.height, MP=d.megapixels;
  const maxRes = d.top_resolutions.length ? d.top_resolutions[0][2] : 1;
  const resBars = d.top_resolutions.map(r=>
    `<div class="ibar"><span class="il">${r[0]}&times;${r[1]}</span><span class="it"><span style="width:${Math.round(100*r[2]/maxRes)}%"></span></span><span class="ic">${r[2]}</span></div>`).join("");
  const _leak=d.leakage_groups.length, _dups=d.duplicate_groups.length;
  let dup;
  if(_dups || _leak){
    const _p=[];
    if(_leak) _p.push(`<div class="iwarn leak">${_leak} group${_leak>1?"s":""} leak across splits (train↔val) - this inflates your validation score.</div>`);
    if(_dups){
      _p.push(`<div class="iwarn">${_dups} duplicate group${_dups>1?"s":""} · ${d.duplicate_image_count} images · Fix moves the extras to a reversible quarantine (keeps the train copy)</div>`);
      _p.push(d.duplicate_groups.slice(0,8).map(g=>
        `<div class="idup">${g.ids.slice(0,7).map(id=>`<img class="ithumb" loading="lazy" src="/api/thumb/${id}?e=${(DS&&DS.epoch)||0}">`).join("")}<span class="idsplit">${esc(g.splits.join(" + "))}</span><button class="ifix" data-ids='${JSON.stringify(g.ids)}'>Fix</button></div>`).join(""));
    } else {
      _p.push(`<div class="iwarn">An identical image path is listed in more than one split - remove the overlap in your data.yaml.</div>`);
    }
    dup = _p.join("");
  } else {
    dup = `<div class="iok">${ICO_CHECK}No duplicate or near-duplicate images detected</div>`;
  }
  $("#insbody").innerHTML =
    readinessSection(lastStats, d, q)
    + `<div class="igrid">`
    + `<div class="icard"><div class="ik">Images</div><div class="iv">${d.measured}</div></div>`
    + `<div class="icard"><div class="ik">Avg size</div><div class="iv">${W.mean}&times;${H.mean}</div></div>`
    + `<div class="icard"><div class="ik">Width</div><div class="iv">${W.min}&ndash;${W.max}</div></div>`
    + `<div class="icard"><div class="ik">Height</div><div class="iv">${H.min}&ndash;${H.max}</div></div>`
    + `<div class="icard"><div class="ik">Megapixels</div><div class="iv">${MP.min}&ndash;${MP.max}</div></div>`
    + `</div>`
    + `<div class="isec"><div class="ititle">Most common resolutions</div>${resBars||'<div class="iload">-</div>'}</div>`
    + `<div class="isec"><div class="ititle">Duplicates &amp; train/val leakage</div>${dup}</div>`
    + `<div class="isec"><div class="ititle">Label geometry</div>${qualitySection(q)}</div>`;
  document.querySelectorAll("#insbody .ifix[data-ids]").forEach(b=> b.onclick=()=>fixDuplicate(JSON.parse(b.dataset.ids), b));
  document.querySelectorAll("#insbody .qrow[data-id]").forEach(b=> b.onclick=()=>{ closeInsights(); load(+b.dataset.id); });
  const rc=$("#rdyexport"); if(rc) rc.onclick=()=>{ closeInsights(); openExport(); };
}
function readinessSection(st, d, q){
  if(!st) return "";
  const labeled=st.labeled||0, total=st.total||0;
  const leak=(d.leakage_groups||[]).length, geo=(q&&q.issues)||0, cls=st.classes||[];
  let imbalance=false;
  if(cls.length>=2){ const mx=cls[0][1], mn=cls[cls.length-1][1]; if(mn===0 || (mn>0 && mx/mn>=20)) imbalance=true; }
  const checks=[];
  checks.push(labeled>0 ? {s:"ok", t:`${labeled} of ${total} images labeled`} : {s:"bad", t:"No labeled images yet - draw some boxes first"});
  checks.push(leak ? {s:"bad", t:`${leak} duplicate group${leak>1?"s":""} leak across splits - Fix below`} : {s:"ok", t:"No leakage across splits"});
  checks.push(geo ? {s:"warn", t:`${geo} geometry issue${geo>1?"s":""} (tiny / sliver / full-frame) - see Label geometry`} : {s:"ok", t:"Box geometry is learnable"});
  if(cls.length>=2) checks.push(imbalance ? {s:"warn", t:"Class balance is skewed - add examples of the rare classes"} : {s:"ok", t:"Classes are reasonably balanced"});
  if(DS && DS.has_val===false) checks.push({s:"warn", t:"No val split yet - add one in Export for held-out metrics."});
  const go = labeled>0 && !leak;
  const ico = s=> s==="ok"?ICO_CHECK : s==="bad"?ICO_X : ICO_WARN;
  const rows = checks.map(c=>`<div class="rdy-row ${c.s}">${ico(c.s)}<span>${esc(c.t)}</span></div>`).join("");
  const cmd = (DS&&DS.yaml) ? `<button class="t-cmd" id="rdyexport"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12M8 11l4 4 4-4"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>Export dataset</button>` : "";
  return `<div class="rdy ${go?"go":""}"><div class="rdy-h">${go?ICO_CHECK:ICO_WARN}${go?"Ready to export":"Almost ready"}</div>${rows}${cmd}</div>`;
}
function qualitySection(q){
  if(!q || !q.issues) return `<div class="iok">${ICO_CHECK}No geometry problems - every box is a learnable size.</div>`;
  const c=q.counts||{};
  const parts=[c.tiny?`${c.tiny} tiny`:"", c.sliver?`${c.sliver} sliver`:"", c.fullframe?`${c.fullframe} full-frame`:""].filter(Boolean).join(" · ");
  const head=`<div class="iwarn">${q.issues} geometry issue${q.issues>1?"s":""} at imgsz ${q.imgsz}${parts?` · ${parts}`:""}</div>`;
  const rows=(q.flagged||[]).slice(0,12).map(f=>{
    const it=(f.issues&&f.issues[0])||{};
    return `<button class="qrow" data-id="${f.id}"><span class="qt ${esc(it.type||"")}">${esc(it.type||"")}</span><span class="qn">${esc(f.name)}</span><span class="qm">${esc(it.msg||"")}${f.count>1?`  (+${f.count-1} more)`:""}</span></button>`;
  }).join("");
  return head+rows;
}
async function fixDuplicate(ids, btn){
  // Fix operates on the on-disk labels and may quarantine/purge the current image,
  // so persist its edits first -- otherwise the only labelled copy could be moved
  // away before the user's edits are saved.
  if(dirty && idx>=0 && !(await save())){ banner("Save the current image before fixing duplicates."); return; }
  if(dirty){ banner("You have unsaved edits - save before fixing duplicates."); return; }
  btn.disabled=true; btn.textContent="…";
  const myEpoch = DS && DS.epoch;   // the post-fix re-fetches below must not land in a since-switched project
  try{
    const r=await fetch("/api/insights/fix",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({ids, epoch:(DS&&DS.epoch)||0})});
    const d=await r.json();
    if(!r.ok){ btn.textContent=(d.error||"failed").slice(0,42); return; }
    if(d.removed && d.removed.length){
      if(!DS || DS.epoch!==myEpoch) return;   // switched projects after the fix POST -> don't mutate the new project's IMAGES/insights
      const removedIds = new Set(d.removed.map(rm=>rm.id));
      d.removed.forEach(rm=>{ suggestedIds.delete(rm.id); if(IMAGES[rm.id]) IMAGES[rm.id].status="deleted"; });
      IMAGES=(await jget("/api/images")).images; mapPoints=[]; mapFit=null; renderList(); scheduleStats();
      if(removedIds.has(idx)){
        // The currently open image was quarantined/purged: its id is now tombstoned,
        // so any further save would be rejected. Drop the dirty flag and move to the
        // survivor (or any live image) instead of stranding the user on a dead canvas.
        dirty = false;
        let dest = (d.kept!=null && IMAGES[d.kept] && IMAGES[d.kept].status!=="deleted") ? d.kept : null;
        if(dest==null){ const live = IMAGES.find(im=> im && im.status!=="deleted"); dest = live? live.id : null; }
        if(dest!=null) await load(dest);
      }
      const ins = await jget("/api/insights");
      const ql = await jget("/api/quality").catch(()=>lastQuality);
      if(!DS || DS.epoch!==myEpoch) return;   // switched mid-refresh -> don't render stale duplicate ids whose Fix buttons would target the new project
      lastInsights = ins; lastQuality = ql;
      renderInsights(lastInsights, lastQuality);
    } else { btn.textContent="no change"; }
  }catch(e){ btn.textContent="failed"; btn.disabled=false; }
}

"""
