"""Closing markup + the first-party feedback control (see ADR 0009 addendum)."""

PART = r"""</script>
<!-- In-app feedback -> GitHub issue. First-party LibreLabel code (MIT, written for
     this repo); it only POSTs the message to LibreYOLO's feedback endpoint, which
     files it as a `feedback`-labelled issue on LibreYOLO/libreyolo. -->
<div class="fb" id="fb">
  <button class="fb-btn glass" id="fbbtn" title="Send feedback - files a GitHub issue"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 8.9 8.9 0 0 1-3.7-.8L3 20l1-4.1a8.3 8.3 0 0 1-1-4.4 8.4 8.4 0 0 1 9-8.4 8.4 8.4 0 0 1 9 8.4z"/></svg>Feedback</button>
  <div class="fb-panel" id="fbpanel">
    <div class="fb-head">Send feedback<button class="mx" id="fbclose">&times;</button></div>
    <textarea id="fbtext" placeholder="A bug, a missing feature, anything…"></textarea>
    <div class="fb-foot"><span class="fb-status" id="fbstatus"></span><button class="btn btn-primary btn-sm" id="fbsend">Send</button></div>
    <div class="fb-note">Posted publicly as a GitHub issue on LibreYOLO/libreyolo.</div>
  </div>
</div>
<script>
"use strict";
(function(){
  const q = s => document.querySelector(s);
  const btn=q("#fbbtn"), panel=q("#fbpanel"), text=q("#fbtext"), send=q("#fbsend"), status=q("#fbstatus");
  const open = ()=>{ panel.classList.add("show"); status.textContent=""; text.focus(); };
  const close = ()=>panel.classList.remove("show");
  btn.onclick = ()=> panel.classList.contains("show") ? close() : open();
  q("#fbclose").onclick = close;
  text.addEventListener("keydown", e=>{ if(e.key==="Escape"){ e.stopPropagation(); close(); } });
  send.onclick = async ()=>{
    const message = text.value.trim();
    if(!message){ status.textContent="Write something first."; status.className="fb-status err"; return; }
    send.disabled = true; status.textContent="Sending…"; status.className="fb-status";
    try{
      const r = await fetch("https://issue-creator.xuban-ceccon.workers.dev/feedback", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({repo:"LibreYOLO/libreyolo", app:"librelabel", message,
                              meta:{url:location.pathname, userAgent:navigator.userAgent}})});
      const d = await r.json().catch(()=>({}));
      if(r.ok && d.ok && d.url){
        status.innerHTML = 'Filed - <a href="'+d.url.replace(/"/g,"%22")+'" target="_blank" rel="noopener">view issue</a>';
        status.className="fb-status ok"; text.value="";
      } else { status.textContent = d.error || ("Something went wrong (HTTP "+r.status+")."); status.className="fb-status err"; }
    }catch(e){ status.textContent="Network error - could not reach the feedback service."; status.className="fb-status err"; }
    finally{ send.disabled=false; }
  };
})();
</script>
</body>
</html>
"""
