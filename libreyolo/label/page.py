"""The embedded LibreLabel single-page app (HTML + CSS + vanilla JS).

Served verbatim at ``/`` by :mod:`libreyolo.label.server`. No build step, no
framework, no third-party JS -- exactly like :mod:`libreyolo.ui.page`. The
canvas is hand-written Canvas 2D; boxes are edited in image-pixel space and
normalised to ``[0, 1]`` only at save time.
"""

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LibreLabel</title>
<style>
  :root{
    --bg:#0a0b0e; --bg2:#0c0e12;
    --s1:#13151b; --s2:#181b22; --s3:#20242d;
    --line:#23272f; --line2:#2d323c;
    --tx:#eceef3; --tx2:#a3abb9; --tx3:#6a7280;
    --ac:#6e7bff; --ai:#a78bfa;
    --ok:#2dd4a7; --warn:#f5b13d; --danger:#fb7185;
    --r:10px; --r2:8px; --sh:0 10px 34px rgba(0,0,0,.5); --shs:0 2px 8px rgba(0,0,0,.3);
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--tx);
    font:13px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
  button{font:inherit;color:inherit;cursor:pointer}
  .ic{width:16px;height:16px;display:block;flex:none}
  #app{display:grid;grid-template-rows:52px 1fr;height:100vh}
  /* topbar */
  .topbar{display:flex;align-items:center;gap:12px;padding:0 14px;
    background:linear-gradient(180deg,#101218,#0b0d11);border-bottom:1px solid var(--line)}
  .brand{display:flex;align-items:center;gap:8px;font-weight:650;letter-spacing:.2px}
  .brand .ic{width:21px;height:21px;color:var(--ac)}
  .brand b{color:var(--ac)}
  .topbar .sep{width:1px;height:20px;background:var(--line2)}
  .topbar .ds{color:var(--tx3);font-size:12px;max-width:190px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .topbar .counter{color:var(--tx2);font-variant-numeric:tabular-nums;font-size:12.5px}
  .topbar .counter b{color:var(--tx)}
  .grow{flex:1}
  .btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;height:32px;padding:0 13px;
    border-radius:var(--r2);border:1px solid transparent;font-weight:560;transition:.15s;white-space:nowrap}
  .btn .ic{width:15px;height:15px}
  .btn-primary{background:linear-gradient(180deg,#8089ff,#6a6cf6);color:#0a0b12;
    box-shadow:0 1px 0 rgba(255,255,255,.18) inset,0 5px 16px rgba(110,123,255,.34)}
  .btn-primary:hover{filter:brightness(1.07);transform:translateY(-1px)}
  .btn-primary:active{transform:translateY(0)}
  .btn-ghost{background:var(--s2);border-color:var(--line2);color:var(--tx)}
  .btn-ghost:hover{background:var(--s3)}
  .btn-sm{height:30px;padding:0 11px;font-size:12px}
  .btn-icon{display:grid;place-items:center;width:32px;height:32px;border-radius:var(--r2);
    background:transparent;border:1px solid transparent;color:var(--tx2);transition:.15s}
  .btn-icon:hover{background:var(--s2);color:var(--tx);border-color:var(--line)}
  .ai{display:flex;align-items:center;gap:7px;padding:4px 6px;border-radius:12px;
    background:rgba(167,139,250,.06);border:1px solid rgba(167,139,250,.16)}
  .ai .field{display:flex;align-items:center;gap:8px;height:32px;padding:0 11px;border-radius:var(--r2);
    background:var(--s1);border:1px solid var(--line);color:var(--tx3);font-size:12px}
  .ai .field b{color:var(--tx);font-variant-numeric:tabular-nums;min-width:28px;text-align:right}
  .ai input[type=range]{-webkit-appearance:none;appearance:none;width:92px;height:4px;border-radius:9px;background:var(--s3);outline:none}
  .ai input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;
    background:var(--ai);cursor:pointer;box-shadow:0 0 0 3px rgba(167,139,250,.2)}
  .select{height:32px;border-radius:var(--r2);background:var(--s1);color:var(--tx2);
    border:1px solid var(--line);padding:0 8px;font-size:12px;max-width:140px}
  .save{display:inline-flex;align-items:center;gap:7px;height:30px;padding:0 12px;border-radius:999px;
    border:1px solid var(--line);color:var(--tx3);font-size:12px;font-weight:540}
  .save::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
  .save.dirty{color:var(--warn);border-color:rgba(245,177,61,.35);background:rgba(245,177,61,.08)}
  .save.saved{color:var(--ok);border-color:rgba(45,212,167,.35);background:rgba(45,212,167,.08)}
  @keyframes pop{0%{transform:scale(1)}40%{transform:scale(1.14)}100%{transform:scale(1)}}
  .save.flash{animation:pop .42s ease-out}
  /* main */
  main{display:grid;grid-template-columns:300px 1fr;min-height:0}
  .sidebar{display:flex;flex-direction:column;min-height:0;background:var(--bg2);border-right:1px solid var(--line)}
  .side-head{padding:12px 12px 10px;border-bottom:1px solid var(--line)}
  .seg{display:flex;gap:2px;padding:3px;background:var(--s1);border:1px solid var(--line);border-radius:var(--r2)}
  .seg button{flex:1;height:28px;border:0;border-radius:6px;background:transparent;color:var(--tx3);font-size:12px;font-weight:560;transition:.12s}
  .seg button.on{background:var(--s3);color:var(--tx);box-shadow:var(--shs)}
  .seg button:hover:not(.on){color:var(--tx2)}
  .list{flex:1;overflow:auto;padding:8px;display:flex;flex-direction:column;gap:6px}
  .list::-webkit-scrollbar{width:11px}
  .list::-webkit-scrollbar-thumb{background:var(--s3);border-radius:9px;border:3px solid var(--bg2)}
  .card{display:flex;gap:10px;align-items:center;padding:7px;border-radius:var(--r2);
    background:var(--s1);border:1px solid transparent;text-align:left;transition:.12s;width:100%}
  .card:hover{background:var(--s2);border-color:var(--line)}
  .card.sel{background:var(--s2);border-color:var(--ac);box-shadow:0 0 0 1px var(--ac)}
  .card .thumb{width:54px;height:40px;border-radius:6px;object-fit:cover;background:var(--s3);flex:none}
  .card .meta{display:flex;flex-direction:column;gap:3px;min-width:0;flex:1}
  .card .fn{font-size:12.5px;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .card .st{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--tx3);text-transform:capitalize}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--tx3);flex:none;display:inline-block}
  .dot.labeled{background:var(--ok)} .dot.empty{background:#54607a}
  .dot.unlabeled{background:var(--tx3)} .dot.suggested{background:var(--ai)}
  .empty{padding:34px 12px;text-align:center;color:var(--tx3);font-size:12px}
  .side-stats{border-top:1px solid var(--line);padding:11px 12px 13px;max-height:232px;overflow:auto;background:var(--bg2)}
  .side-stats .sh{display:flex;justify-content:space-between;align-items:baseline;font-size:10.5px;color:var(--tx3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px}
  .side-stats .sh b{color:var(--tx2);letter-spacing:0;text-transform:none;font-size:11.5px}
  .statrow{display:flex;align-items:center;gap:8px;margin-bottom:6px}
  .statrow .sw{width:9px;height:9px;border-radius:3px;flex:none}
  .statrow .nm{font-size:11.5px;color:var(--tx2);width:72px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:none}
  .statrow .barwrap{flex:1;height:7px;background:var(--s1);border-radius:9px;overflow:hidden}
  .statrow .bar{height:100%;border-radius:9px;transition:width .3s ease}
  .statrow .ct{font-size:11px;color:var(--tx3);font-variant-numeric:tabular-nums;width:30px;text-align:right;flex:none}
  .side-stats .none{color:var(--tx3);font-size:11.5px;text-align:center;padding:6px 0}
  .traincta{display:none;align-items:center;justify-content:space-between;gap:8px;padding:10px 12px;
    border-top:1px solid var(--line);background:linear-gradient(180deg,rgba(45,212,167,.07),transparent)}
  .traincta .t-l{display:flex;align-items:center;gap:7px;font-size:11.5px;color:var(--ok);font-weight:560}
  .traincta .t-l .ic{width:14px;height:14px}
  .t-cmd{display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:7px;
    background:var(--s2);border:1px solid var(--line2);color:var(--tx2);transition:.13s}
  .t-cmd:hover{background:var(--s3);color:var(--tx)}
  .t-cmd code{font:11px ui-monospace,monospace}
  .t-cmd .ic{width:13px;height:13px}
  .t-cmd.copied{color:var(--ok);border-color:rgba(45,212,167,.4);background:rgba(45,212,167,.08)}
  /* stage */
  .stage{position:relative;min-width:0;overflow:hidden;
    background:radial-gradient(130% 130% at 50% 0%,#0f1116,#08090c)}
  canvas{display:block;width:100%;height:100%;touch-action:none;cursor:crosshair}
  .glass{background:rgba(17,19,25,.82);backdrop-filter:blur(12px);border:1px solid var(--line2)}
  .toolbar{position:absolute;top:14px;right:14px;display:flex;flex-direction:column;gap:5px;
    padding:6px;border-radius:13px;box-shadow:var(--sh)}
  .tool{display:grid;place-items:center;width:36px;height:36px;border-radius:9px;background:transparent;
    border:1px solid transparent;color:var(--tx2);transition:.12s}
  .tool:hover{background:var(--s2);color:var(--tx)}
  .tool.ai{color:var(--ai)} .tool.ai:hover{background:rgba(167,139,250,.16)}
  .tdiv{height:1px;background:var(--line);margin:2px 5px}
  .hud{position:absolute;top:14px;left:14px;padding:7px 12px;border-radius:10px;font-size:12px;
    color:var(--tx2);box-shadow:var(--shs);font-variant-numeric:tabular-nums}
  .classbar{position:absolute;left:50%;bottom:16px;transform:translateX(-50%)}
  .classchip{display:inline-flex;align-items:center;gap:9px;height:40px;padding:0 16px;border-radius:999px;
    color:var(--tx);box-shadow:var(--sh);font-weight:560;transition:.14s}
  .classchip:hover{border-color:var(--ac)}
  .classchip .sw{width:14px;height:14px;border-radius:4px}
  .classchip .cc-h{color:var(--tx3);font-size:10.5px;text-transform:uppercase;letter-spacing:.7px}
  .picker{position:absolute;left:50%;bottom:66px;transform:translateX(-50%) translateY(8px);
    width:min(540px,88vw);max-height:48vh;display:none;flex-direction:column;opacity:0;transition:.16s;
    background:var(--s1);border:1px solid var(--line2);border-radius:15px;box-shadow:var(--sh);overflow:hidden;z-index:4}
  .picker.show{display:flex;opacity:1;transform:translateX(-50%) translateY(0)}
  .psearch{display:flex;align-items:center;gap:9px;padding:12px 14px;border-bottom:1px solid var(--line)}
  .psearch .ic{width:15px;height:15px;color:var(--tx3)}
  .psearch input{flex:1;background:transparent;border:0;outline:none;color:var(--tx);font-size:13px}
  #pal{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:3px;padding:10px;overflow:auto}
  .pclass{display:flex;align-items:center;gap:9px;padding:7px 9px;border-radius:8px;background:transparent;border:1px solid transparent;text-align:left;transition:.1s}
  .pclass:hover{background:var(--s2)} .pclass.on{background:var(--s3);border-color:var(--ac)}
  .pclass .sw{width:12px;height:12px;border-radius:3px;flex:none}
  .pclass .pn{flex:1;font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .pclass .pk{color:var(--tx3);font-size:11px;font-variant-numeric:tabular-nums;background:var(--s3);border-radius:4px;padding:0 5px}
  .banner{position:absolute;left:50%;top:14px;transform:translateX(-50%);display:none;align-items:center;gap:8px;
    max-width:min(680px,84vw);padding:9px 14px;border-radius:10px;font-size:12.5px;
    background:rgba(26,21,12,.94);color:var(--warn);border:1px solid rgba(245,177,61,.32);box-shadow:var(--sh)}
  .progress{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
    background:rgba(8,9,12,.78);backdrop-filter:blur(3px);z-index:6}
  .pcard{width:384px;padding:26px;border-radius:16px;background:var(--s1);border:1px solid var(--line2);box-shadow:var(--sh);text-align:center}
  .pcard .ic{width:30px;height:30px;color:var(--ai);margin:0 auto 10px}
  .ptitle{font-weight:650;font-size:15px;margin-bottom:5px}
  .ptxt{color:var(--tx3);font-size:12.5px;font-variant-numeric:tabular-nums;margin-bottom:15px;min-height:18px}
  .ptrack{height:7px;border-radius:99px;background:var(--s3);overflow:hidden}
  .pbar{height:100%;width:0;border-radius:99px;background:linear-gradient(90deg,var(--ac),var(--ai));transition:width .25s ease}
  .help{position:absolute;inset:0;display:none;align-items:center;justify-content:center;background:rgba(8,9,12,.8);backdrop-filter:blur(3px);z-index:7}
  .help .card2{width:min(560px,90vw);background:var(--s1);border:1px solid var(--line2);border-radius:16px;padding:22px 24px;box-shadow:var(--sh)}
  .help h3{margin:0 0 14px;font-size:15px;display:flex;align-items:center;gap:8px}
  .help table{border-collapse:collapse;width:100%}
  .help td{padding:6px 8px;border-bottom:1px solid var(--line);font-size:12.5px;color:var(--tx2)}
  .help td:first-child{white-space:nowrap;width:46%}
  .help kbd{display:inline-block;background:var(--s3);border:1px solid var(--line2);border-bottom-width:2px;border-radius:6px;padding:1px 7px;font:11px ui-monospace,monospace;color:var(--tx)}
  :focus-visible{outline:2px solid var(--ac);outline-offset:2px}
</style>
</head>
<body>
<div id="app">
  <header class="topbar">
    <span class="brand"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16"/><rect x="9" y="9" width="6" height="6" rx="1.5" fill="currentColor" stroke="none"/></svg>Libre<b>Label</b></span>
    <span class="sep"></span>
    <span class="ds" id="dsname"></span>
    <span class="counter" id="counter"></span>
    <span class="grow"></span>
    <span class="ai" id="assistbar" style="display:none">
      <button class="btn btn-primary" id="aautolabel"><svg class="ic" viewBox="0 0 24 24" fill="currentColor"><path d="M11.5 2.5l1.6 4.4 4.4 1.6-4.4 1.6-1.6 4.4-1.6-4.4-4.4-1.6 4.4-1.6z"/><path d="M18.5 14l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/></svg>Auto-label all</button>
      <button class="btn btn-ghost btn-sm" id="aprelabel" title="Auto-label this image (R)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 4V2M19 8h2M5 20l9-9M14 6l4 4"/></svg>R</button>
      <span class="field"><span>conf</span><input type="range" id="aconf" min="0.05" max="0.9" step="0.05"><b id="aconfval">0.25</b></span>
      <select id="amodel" class="select"></select>
    </span>
    <span class="save" id="save"></span>
    <button class="btn-icon" id="helpbtn" title="Shortcuts (?)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.6 9a2.4 2.4 0 1 1 3.4 2.2c-.9.4-1.1.9-1.1 1.8"/><path d="M12 17h.01"/></svg></button>
  </header>
  <main>
    <aside class="sidebar">
      <div class="side-head">
        <div class="seg" id="filter">
          <button data-f="all" class="on">All</button>
          <button data-f="todo">To-do</button>
          <button data-f="review">Review</button>
        </div>
      </div>
      <div class="list" id="list"></div>
      <div class="side-stats" id="stats"></div>
      <div class="traincta" id="traincta"></div>
    </aside>
    <div class="stage" id="stage">
      <canvas id="cv"></canvas>
      <div class="hud glass" id="hud"></div>
      <div class="toolbar glass">
        <button class="tool on" id="toolBox" title="Box / select (B)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/></svg></button>
        <button class="tool" id="toolSeg" title="Smart segment — SAM click-to-mask (S)" style="display:none"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12a8 8 0 1 1 8 8"/><path d="M12 20a8 8 0 0 1-8-8" stroke-dasharray="2 3"/><circle cx="12" cy="12" r="2.5" fill="currentColor" stroke="none"/></svg></button>
        <div class="tdiv"></div>
        <button class="tool ai" id="toolAi" title="AI auto-label this image (R)"><svg class="ic" viewBox="0 0 24 24" fill="currentColor"><path d="M11.5 2.5l1.6 4.4 4.4 1.6-4.4 1.6-1.6 4.4-1.6-4.4-4.4-1.6 4.4-1.6z"/><path d="M18.5 14l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/></svg></button>
        <div class="tdiv"></div>
        <button class="tool" id="toolFit" title="Fit to view (F)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4"/></svg></button>
        <button class="tool" id="toolZin" title="Zoom in"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M11 8v6M8 11h6"/></svg></button>
        <button class="tool" id="toolZout" title="Zoom out"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M8 11h6"/></svg></button>
      </div>
      <div class="classbar">
        <button class="classchip glass" id="classchip"></button>
      </div>
      <div class="picker" id="picker">
        <div class="psearch"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg><input id="psearch" placeholder="Search classes…" autocomplete="off"></div>
        <div id="pal"></div>
      </div>
      <div class="banner" id="banner"></div>
      <div class="progress" id="progress"><div class="pcard">
        <svg class="ic" viewBox="0 0 24 24" fill="currentColor"><path d="M11.5 2.5l1.6 4.4 4.4 1.6-4.4 1.6-1.6 4.4-1.6-4.4-4.4-1.6 4.4-1.6z"/><path d="M18.5 14l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/></svg>
        <div class="ptitle">Auto-labeling with your model</div>
        <div class="ptxt" id="ptxt"></div>
        <div class="ptrack"><div class="pbar" id="pbar"></div></div>
      </div></div>
      <div class="help" id="help"><div class="card2">
        <h3><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M9.6 9a2.4 2.4 0 1 1 3.4 2.2c-.9.4-1.1.9-1.1 1.8M12 17h.01"/></svg>Keyboard shortcuts</h3>
        <table>
          <tr><td><kbd>drag</kbd></td><td>draw a box (uses active class)</td></tr>
          <tr><td><kbd>click</kbd> / drag handle</td><td>select &middot; move &middot; resize</td></tr>
          <tr><td><kbd>1</kbd>..<kbd>9</kbd> <kbd>0</kbd> / <kbd>/</kbd></td><td>set class &middot; open class search</td></tr>
          <tr><td><kbd>R</kbd></td><td>AI auto-label this image</td></tr>
          <tr><td><kbd>S</kbd> / <kbd>B</kbd></td><td>smart-segment (SAM click-to-mask) / box tool</td></tr>
          <tr><td><kbd>Enter</kbd> / <kbd>Shift</kbd>+<kbd>Enter</kbd></td><td>accept all AI suggestions / accept &amp; next</td></tr>
          <tr><td><kbd>A</kbd>/<kbd>D</kbd> &middot; <kbd>E</kbd></td><td>prev / next image &middot; next unlabeled</td></tr>
          <tr><td><kbd>Del</kbd> &middot; <kbd>Ctrl</kbd>+<kbd>Z</kbd></td><td>delete selected &middot; undo</td></tr>
          <tr><td><kbd>Space</kbd>+drag &middot; <kbd>wheel</kbd> &middot; <kbd>F</kbd></td><td>pan &middot; zoom &middot; fit</td></tr>
          <tr><td><kbd>Ctrl</kbd>+<kbd>S</kbd> &middot; <kbd>Esc</kbd></td><td>save &middot; cancel / clear / close</td></tr>
        </table>
      </div></div>
    </div>
  </main>
</div>
<script>
"use strict";
const $ = s => document.querySelector(s);
const cv = $("#cv"), ctx = cv.getContext("2d");
let DS = null, IMAGES = [], idx = -1;
let img = new Image(), imgOk = false;
let boxes = [], editable = true, dirty = false;
let active = 0, sel = -1;
let view = {scale:1, ox:0, oy:0};
let VW = 1, VH = 1;
let loadSeq = 0;
let undoStack = [];
let gestureSnap = null;
let cursor = null;
let hover = -1;
let stageMsg = "";
let progSig = "";
let assist = null, assistModel = null, conf = 0.35;
let ghosts = [];
let polys = [];              // polygon annotations (image-px pts), from SAM or file
let selPoly = -1;
let tool = "box";            // "box" (draw/select) or "seg" (SAM click-to-mask)
let segBusy = false;
let suggestedIds = new Set();
let listFilter = "all";
const HANDLES = ["nw","n","ne","e","se","s","sw","w"];
const HR = 6;
const color = i => { const h=(i*137.508)%360; const l=62 - 16*Math.cos((h-50)*Math.PI/180); return 'hsl('+h+' 70% '+l+'%)'; };
const clamp01 = v => v<0?0:v>1?1:v;
const ICO_CHECK = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 6"/></svg>';
const ICO_COPY = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';
const esc = s => String(s).replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// ---- transforms ----
const sx = x => view.ox + x*view.scale;
const sy = y => view.oy + y*view.scale;
const ix = px => (px - view.ox)/view.scale;
const iy = py => (py - view.oy)/view.scale;
function rr(x,y,w,h,r){ ctx.beginPath(); if(ctx.roundRect) ctx.roundRect(x,y,w,h,r); else ctx.rect(x,y,w,h); ctx.fill(); }

function fit(){
  if(!imgOk) return;
  const pad = 40, W = VW, H = VH;
  const s = Math.min((W-pad)/img.naturalWidth, (H-pad)/img.naturalHeight);
  view.scale = s>0 ? s : 1;
  view.ox = (W - img.naturalWidth*view.scale)/2;
  view.oy = (H - img.naturalHeight*view.scale)/2;
}
function resizeCanvas(){
  const r = $("#stage").getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio||1, 2);
  VW = Math.max(1, Math.floor(r.width));
  VH = Math.max(1, Math.floor(r.height));
  cv.width = Math.round(VW*dpr); cv.height = Math.round(VH*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  draw();
}
function zoomBy(f){
  const cx=VW/2, cy=VH/2, bx=ix(cx), by=iy(cy);
  view.scale = Math.max(0.02, Math.min(64, view.scale*f));
  view.ox = cx-bx*view.scale; view.oy = cy-by*view.scale; draw();
}

// ---- API ----
async function jget(u){ const r = await fetch(u); if(!r.ok) throw new Error((await r.json()).error||r.status); return r.json(); }

async function init(){
  DS = await jget("/api/dataset");
  $("#dsname").textContent = (DS.root||"").split(/[\\/]/).filter(Boolean).pop() || "dataset";
  renderPalette();
  IMAGES = (await jget("/api/images")).images;
  renderList();
  renderStats();
  resizeCanvas();
  wireChrome();
  initAssist();
  if(IMAGES.length) load(0);
  else { stageMsg = "No images found — check the data=… paths"; setSave("no images"); draw(); }
}
function wireChrome(){
  $("#classchip").onclick = togglePicker;
  $("#psearch").oninput = e=> filterClasses(e.target.value);
  $("#helpbtn").onclick = toggleHelp;
  $("#toolBox").onclick = ()=> setTool("box");
  $("#toolSeg").onclick = ()=> setTool("seg");
  $("#toolAi").onclick = ()=> prelabelCurrent();
  $("#toolFit").onclick = ()=>{ fit(); draw(); };
  $("#toolZin").onclick = ()=> zoomBy(1.25);
  $("#toolZout").onclick = ()=> zoomBy(1/1.25);
  document.querySelectorAll("#filter button").forEach(b=> b.onclick = ()=>{
    document.querySelectorAll("#filter button").forEach(x=>x.classList.remove("on"));
    b.classList.add("on"); listFilter = b.dataset.f; renderList();
  });
}
function setTool(t){
  if(t==="seg" && !(assist && assist.sam)) return;
  tool = t;
  const tb=$("#toolBox"), ts=$("#toolSeg");
  if(tb) tb.classList.toggle("on", t==="box");
  if(ts) ts.classList.toggle("on", t==="seg");
  if(t==="seg") banner("Smart segment: click an object → SAM outlines it. Esc / B for box tool.");
  else $("#banner").style.display="none";
}
function toggleHelp(){ const h=$("#help"); h.style.display = h.style.display==="flex"?"none":"flex"; }

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
}
function markPalette(){
  const nm = (DS.names&&DS.names[active]!=null)?DS.names[active]:active;
  $("#classchip").innerHTML = `<span class="sw" style="background:${color(active)}"></span>`+
    `<span>${esc(nm)}</span><span class="cc-h">class</span>`;
  document.querySelectorAll("#pal .pclass").forEach(c=> c.classList.toggle("on", +c.dataset.i===active));
}
function setActive(i){
  if(i<0||i>=(DS.names||[]).length) return;
  active = i; markPalette();
  if(sel>=0 && boxes[sel].cls!==i){ pushUndo(); boxes[sel].cls = i; markDirty(); draw(); }
}
function openPicker(){ $("#picker").classList.add("show"); const s=$("#psearch"); s.value=""; filterClasses(""); s.focus(); }
function closePicker(){ $("#picker").classList.remove("show"); }
function togglePicker(){ $("#picker").classList.contains("show")?closePicker():openPicker(); }
function filterClasses(q){ q=(q||"").toLowerCase();
  document.querySelectorAll("#pal .pclass").forEach(c=>{
    const nm=(DS.names[+c.dataset.i]||"").toLowerCase(); c.style.display = nm.includes(q)?"":"none"; }); }

// ---- sidebar list ----
function passFilter(im){
  if(listFilter==="todo") return im.status==="unlabeled";
  if(listFilter==="review") return im.status==="suggested";
  return true;
}
function renderList(){
  const el = $("#list"); el.innerHTML = ""; let shown=0;
  IMAGES.forEach(im=>{
    if(!passFilter(im)) return; shown++;
    const r = document.createElement("button");
    r.className = "card" + (im.id===idx?" sel":""); r.dataset.id = im.id;
    r.innerHTML = `<img class="thumb" loading="lazy" src="/api/thumb/${im.id}" alt="">`+
      `<span class="meta"><span class="fn">${esc(im.name)}</span>`+
      `<span class="st"><i class="dot ${im.status}"></i>${im.status}</span></span>`;
    r.onclick = ()=> load(im.id);
    el.appendChild(r);
  });
  if(!shown) el.innerHTML = `<div class="empty">No ${listFilter==="all"?"":esc(listFilter)+" "}images</div>`;
}
function markRow(){
  document.querySelectorAll("#list .card").forEach(r=> r.classList.toggle("sel", +r.dataset.id===idx));
  const cur = document.querySelector(`#list .card[data-id="${idx}"]`);
  if(cur) cur.scrollIntoView({block:"nearest"});
}
function setRowStatus(id, status){
  IMAGES[id] && (IMAGES[id].status = status);
  const card = document.querySelector(`#list .card[data-id="${id}"]`);
  if(card){
    const st = card.querySelector(".st");
    if(st) st.innerHTML = `<i class="dot ${status}"></i>${status}`;
    if(IMAGES[id] && !passFilter(IMAGES[id])) card.remove();
  }
  updateProgress();
}

// ---- dataset health (live class distribution of accepted labels) ----
let statsTimer = null;
function scheduleStats(){ clearTimeout(statsTimer); statsTimer = setTimeout(renderStats, 500); }
async function renderStats(){
  const el = $("#stats"), tc = $("#traincta"); if(!el) return;
  let s; try{ s = await jget("/api/stats"); }catch(e){ return; }
  if(!s.boxes){
    el.innerHTML = `<div class="sh"><span>Dataset health</span></div>`+
      `<div class="none">No labels yet — Auto-label, then accept</div>`;
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
      tc.innerHTML = `<span class="t-l">${ICO_CHECK}<span>${s.labeled} images ready</span></span>`+
        `<button class="t-cmd" id="traincmd">${ICO_COPY}<code>libreyolo train</code></button>`;
      const b=$("#traincmd");
      if(b) b.onclick=()=>{ try{ navigator.clipboard.writeText('libreyolo train data='+DS.yaml); }catch(e){}
        b.classList.add('copied'); setTimeout(()=>b.classList.remove('copied'),1200); };
    } else tc.style.display="none";
  }
}

// ---- undo ----
function snap(){ return JSON.stringify({b:boxes, p:polys}); }
function applyUndo(s){ const o=JSON.parse(s); boxes=o.b||[]; polys=o.p||[]; }
function pushUndo(){ undoStack.push(snap()); if(undoStack.length>50) undoStack.shift(); }
function snapStart(){ gestureSnap = snap(); }
function snapCommit(){
  if(gestureSnap!==null && gestureSnap!==snap()){
    undoStack.push(gestureSnap); if(undoStack.length>50) undoStack.shift();
  }
  gestureSnap = null;
}

// ---- load / save ----
async function load(i){
  if(i<0||i>=IMAGES.length) return;
  const myGen = ++loadSeq;
  if(dirty && idx>=0 && !(await save())){
    banner("Save failed — staying on this image so you don't lose work."); return;
  }
  idx = i; sel = -1; selPoly = -1; hover = -1; undoStack = []; gestureSnap = null; boxes = []; polys = []; ghosts = [];
  const lab = await jget(`/api/label/${i}`);
  if(myGen !== loadSeq) return;
  editable = lab.editable;
  imgOk = false; stageMsg = "Loading…"; draw();
  img = new Image();
  img.onload = ()=>{
    if(myGen !== loadSeq) return;
    imgOk = true;
    const anns = lab.annotations||[]; const iw=img.naturalWidth, ih=img.naturalHeight;
    boxes = anns.filter(a=>a.type==="box").map(b=>({
      cls:b.cls, x:(b.cx-b.w/2)*iw, y:(b.cy-b.h/2)*ih, w:b.w*iw, h:b.h*ih}));
    polys = anns.filter(a=>a.type==="poly").map(p=>({
      cls:p.cls, pts:p.points.map((v,k)=> k%2===0? v*iw : v*ih)}));
    dirty = false; setSave(editable?"saved":"read-only");
    stageMsg = ""; fit(); draw();
    if(assist && assist.available && editable && suggestedIds.has(i)){
      fetch(`/api/assist/pending/${i}`).then(r=>r.json()).then(d=>{
        if(myGen!==loadSeq) return;
        if((d.suggestions||[]).length) showGhosts(d.suggestions);
      }).catch(()=>{});
    }
  };
  img.onerror = ()=>{ if(myGen !== loadSeq) return; imgOk=false; stageMsg="Could not load image"; setSave("image error"); draw(); };
  img.src = `/api/image/${i}`;
  $("#banner").style.display = "none";
  if(!editable) banner("Read-only: this image has polygon/OBB labels (box-only mode won't overwrite them).");
  else if(DS && !DS.writable) banner(DS.reason);
  markRow(); updateProgress();
}
function pxToNorm(b){
  const iw=img.naturalWidth, ih=img.naturalHeight;
  let x=b.x, y=b.y, w=b.w, h=b.h;
  if(w<0){x+=w;w=-w;} if(h<0){y+=h;h=-h;}
  return {cls:b.cls, cx:clamp01((x+w/2)/iw), cy:clamp01((y+h/2)/ih), w:clamp01(w/iw), h:clamp01(h/ih)};
}
function polyToNorm(p){
  const iw=img.naturalWidth, ih=img.naturalHeight, out=[];
  for(let k=0;k<p.pts.length;k+=2){ out.push(clamp01(p.pts[k]/iw)); out.push(clamp01(p.pts[k+1]/ih)); }
  return out;
}
async function save(){
  if(!imgOk || !editable || (DS && !DS.writable)){ return true; }
  const anns = boxes.map(pxToNorm).filter(b=>b.w>0&&b.h>0)
    .map(b=>({type:"box", cls:b.cls, cx:b.cx, cy:b.cy, w:b.w, h:b.h}));
  polys.forEach(p=>{ const pts=polyToNorm(p); if(pts.length>=6) anns.push({type:"poly", cls:p.cls, points:pts}); });
  const cur = idx;
  try{
    const r = await fetch(`/api/label/${cur}`,{method:"POST",
      headers:{"Content-Type":"application/json"}, body:JSON.stringify({annotations:anns})});
    if(!r.ok){ setSave("save failed"); banner((await r.json()).error||"save failed"); return false; }
    dirty = false; setSave("saved");
    const el=$('#save'); el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
    suggestedIds.delete(cur);
    setRowStatus(cur, anns.length? "labeled":"empty");
    scheduleStats();
    return true;
  }catch(e){ setSave("save failed"); return false; }
}
function markDirty(){ dirty = true; setSave("unsaved"); }
function setSave(t){
  const e = $("#save"); e.textContent = t;
  e.className = "save" + (t==="unsaved"?" dirty": t==="saved"?" saved":"");
}
function banner(msg){ const b=$("#banner"); b.textContent=msg; b.style.display="flex"; }
function updateProgress(){
  if(idx<0 || !IMAGES[idx]) return;
  const done = IMAGES.filter(im=>im.status!=='unlabeled').length;
  const n = boxes.length + polys.length;
  const sig = done+"|"+IMAGES.length+"|"+suggestedIds.size+"|"+idx+"|"+n;
  if(sig===progSig) return; progSig = sig;
  const rev = suggestedIds.size ? ` &middot; <b style="color:var(--ai)">${suggestedIds.size}</b> to review` : "";
  $("#counter").innerHTML = `<b>${done}</b>/${IMAGES.length} labeled${rev}`;
  const hud = $("#hud");
  if(hud) hud.innerHTML = `${idx+1} / ${IMAGES.length} &nbsp;&middot;&nbsp; ${n} box${n===1?'':'es'}`
    + ` &nbsp;&middot;&nbsp; <span style="color:var(--tx3)">${esc(IMAGES[idx].name)}</span>`;
}
function nextUnlabeled(dir){
  const L=IMAGES.length;
  for(let n=1;n<=L;n++){ const j=((idx+dir*n)%L+L)%L;
    if(IMAGES[j].status==="unlabeled"){ load(j); return; } }
  banner("No more unlabeled images");
}

// ---- AI auto-label (suggest -> review -> accept; nothing written unverified) ----
async function initAssist(){
  try{ assist = await jget("/api/assist/status"); }catch(e){ assist = null; }
  const bar = $("#assistbar"), toolAi = $("#toolAi");
  if(!assist || !assist.available){ if(bar) bar.style.display="none"; if(toolAi) toolAi.style.display="none"; return; }
  assistModel = assist.default;
  const sel = $("#amodel"); sel.innerHTML = "";
  assist.models.forEach(m=>{ const o=document.createElement("option");
    o.value=m; o.textContent=m; if(m===assistModel) o.selected=true; sel.appendChild(o); });
  sel.onchange = ()=> assistModel = sel.value;
  const cs = $("#aconf"); cs.value = conf; $("#aconfval").textContent = conf.toFixed(2);
  cs.oninput = ()=>{ conf = parseFloat(cs.value); $("#aconfval").textContent = conf.toFixed(2); };
  $("#aprelabel").onclick = ()=> prelabelCurrent();
  $("#aautolabel").onclick = ()=> autolabelAll();
  bar.style.display = "flex";
  if(assist.sam){ const ts=$("#toolSeg"); if(ts) ts.style.display="grid"; }
}
function restoreSave(){ setSave(dirty?"unsaved":(editable?"saved":"read-only")); }
function ghostsFromNorm(list){
  if(!imgOk) return [];
  const iw=img.naturalWidth, ih=img.naturalHeight;
  return list.map(s=>({cls:s.cls, name:s.name, conf:s.conf, mapped:s.mapped,
    x:(s.cx-s.w/2)*iw, y:(s.cy-s.h/2)*ih, w:s.w*iw, h:s.h*ih}));
}
function showGhosts(list){
  ghosts = ghostsFromNorm(list); draw();
  if(ghosts.length){
    const unm = ghosts.filter(g=>!g.mapped).length;
    banner(`${ghosts.length} AI suggestion${ghosts.length===1?"":"s"} — `
      + "Enter: accept all · click one · Alt+click rejects · Esc clears"
      + (unm? ` · ${unm} unmatched (grey) — set a class to accept`:""));
  } else banner("No objects found above the confidence threshold");
}
async function prelabelCurrent(){
  if(!assist || !assist.available || idx<0 || !imgOk) return;
  if(!editable){ banner("This image is read-only (polygon/OBB labels)."); return; }
  const myGen = loadSeq; setSave("running model…");
  try{
    const r = await fetch(`/api/assist/prelabel/${idx}?model=${encodeURIComponent(assistModel)}&conf=${conf}`, {method:"POST"});
    if(myGen!==loadSeq) return;
    if(!r.ok){ const e=await r.json().catch(()=>({})); banner("Auto-label failed: "+(e.error||r.status)); restoreSave(); return; }
    const data = await r.json();
    showGhosts(data.suggestions||[]);
    if((data.suggestions||[]).length){ suggestedIds.add(idx); }
  }catch(e){ banner("Auto-label failed"); }
  restoreSave();
}
function pointInPoly(x,y,pts){
  let inside=false; const n=pts.length/2;
  for(let i=0, j=n-1; i<n; j=i++){
    const xi=pts[2*i], yi=pts[2*i+1], xj=pts[2*j], yj=pts[2*j+1];
    if(((yi>y)!==(yj>y)) && (x < (xj-xi)*(y-yi)/(yj-yi)+xi)) inside=!inside;
  }
  return inside;
}
function hitPoly(mx,my){
  const x=ix(mx), y=iy(my);
  for(let i=polys.length-1;i>=0;i--){ if(pointInPoly(x,y,polys[i].pts)) return i; }
  return -1;
}
function hitVertex(p, mx, my){
  for(let k=0;k<p.pts.length;k+=2){
    if(Math.abs(sx(p.pts[k])-mx)<=HR+2 && Math.abs(sy(p.pts[k+1])-my)<=HR+2) return k/2;
  }
  return -1;
}
async function segmentAt(mx,my){
  if(segBusy || !assist || !assist.sam || idx<0 || !imgOk || !editable) return;
  const iw=img.naturalWidth, ih=img.naturalHeight, X=ix(mx), Y=iy(my);
  if(X<0||Y<0||X>iw||Y>ih) return;
  segBusy=true; const myGen=loadSeq; banner("Segmenting… (SAM, on your machine)"); cv.style.cursor="wait";
  try{
    const r = await fetch(`/api/assist/segment/${idx}`, {method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({x:X/iw, y:Y/ih})});
    if(myGen!==loadSeq) return;
    if(!r.ok){ const e=await r.json().catch(()=>({})); banner("Segment failed: "+(e.error||r.status)); return; }
    const d = await r.json();
    if(!d.polygon || d.polygon.length<6){ banner("No object found there — try clicking on an object"); return; }
    pushUndo();
    polys.push({cls:active, pts:d.polygon.map((v,k)=> k%2===0? v*iw : v*ih)});
    selPoly=polys.length-1; sel=-1; markDirty(); $("#banner").style.display="none"; draw();
  }catch(e){ banner("Segment failed"); }
  finally{ segBusy=false; cv.style.cursor="crosshair"; }
}
function hitGhost(mx,my){
  const x=ix(mx), y=iy(my);
  for(let i=ghosts.length-1;i>=0;i--){ const g=ghosts[i];
    if(x>=g.x && x<=g.x+g.w && y>=g.y && y<=g.y+g.h) return i; }
  return -1;
}
function acceptGhost(i){
  const g=ghosts[i]; if(!g) return;
  pushUndo();
  boxes.push({cls:(g.cls!=null?g.cls:active), x:g.x, y:g.y, w:g.w, h:g.h});
  ghosts.splice(i,1); markDirty(); draw();
  if(!ghosts.length) $("#banner").style.display="none";
}
function rejectGhost(i){ if(ghosts[i]){ ghosts.splice(i,1); draw(); if(!ghosts.length) $("#banner").style.display="none"; } }
function acceptAllGhosts(){
  const take = ghosts.filter(g=>g.cls!=null);
  if(!take.length){ if(ghosts.length) banner("These suggestions have no matching dataset class — set one with a number key, or Esc to skip."); return; }
  pushUndo();
  take.forEach(g=> boxes.push({cls:g.cls, x:g.x, y:g.y, w:g.w, h:g.h}));
  ghosts = ghosts.filter(g=>g.cls==null);
  markDirty(); draw();
  if(!ghosts.length) $("#banner").style.display="none";
}
function clearGhosts(){ if(ghosts.length){ ghosts=[]; draw(); $("#banner").style.display="none"; } }
async function autolabelAll(){
  if(!assist || !assist.available) return;
  if(dirty && idx>=0 && !(await save())){ banner("Couldn't save the current image; fix that first."); return; }
  const ov=$("#progress"), bar=$("#pbar"), txt=$("#ptxt");
  ov.style.display="flex"; bar.style.width="0%"; txt.textContent="Starting… (first run loads your model)";
  suggestedIds = new Set(); let suggested=0, totalBoxes=0, classes=[]; const t0=Date.now();
  try{
    const r = await fetch(`/api/assist/autolabel?model=${encodeURIComponent(assistModel)}&conf=${conf}`, {method:"POST"});
    const reader=r.body.getReader(), dec=new TextDecoder(); let buf="";
    for(;;){
      const {value,done}=await reader.read(); if(done) break;
      buf += dec.decode(value,{stream:true}); let nl;
      while((nl=buf.indexOf("\n"))>=0){
        const line=buf.slice(0,nl).trim(); buf=buf.slice(nl+1);
        if(!line) continue; let o; try{o=JSON.parse(line);}catch(e){ continue; }
        if(o.type==="progress"){
          bar.style.width = Math.round(100*o.i/Math.max(1,o.total))+"%";
          txt.textContent = `${o.i} / ${o.total} — ${o.name}` + (o.count? `  (+${o.count})`:"");
          if(o.count>0){ suggestedIds.add(o.id); setRowStatus(o.id, "suggested"); }
        } else if(o.type==="done"){ suggested=o.suggested; totalBoxes=o.boxes; classes=o.classes||[]; }
        else if(o.type==="error"){ banner("Auto-label failed: "+o.error); }
      }
    }
    bar.style.width="100%";
    const secs = ((Date.now()-t0)/1000).toFixed(1);
    const top = classes.slice(0,5).map(c=>`${c[1]} ${esc(c[0])}`).join("  ·  ");
    $(".ptitle").textContent = "Done — fully offline, your own model";
    txt.innerHTML = `<b style="color:var(--tx);font-size:16px">${totalBoxes} boxes</b> across `
      + `<b style="color:var(--tx)">${suggested}</b> images in <b style="color:var(--tx)">${secs}s</b>`
      + (top? `<div style="margin-top:9px;color:var(--tx2)">${top}</div>`:"")
      + `<div style="margin-top:9px;color:var(--tx3)">Review &amp; accept — nothing is saved until you confirm →</div>`;
    setTimeout(()=>{ ov.style.display="none"; $(".ptitle").textContent="Auto-labeling with your model"; }, 2800);
    progSig = "";
    const first = [...suggestedIds].sort((a,b)=>a-b)[0];
    if(first!=null) await load(first);
    else banner("No objects found in the unlabeled images");
  }catch(e){ banner("Auto-label failed"); ov.style.display="none"; }
}
function nextSuggested(){
  if(!suggestedIds.size) return;
  const ids=[...suggestedIds].sort((a,b)=>a-b);
  const nxt = ids.find(j=>j>idx);
  load(nxt!=null? nxt : ids[0]);
}

// ---- drawing ----
function draw(){
  ctx.clearRect(0,0,VW,VH);
  if(!imgOk){
    if(stageMsg){ ctx.fillStyle="#6a7280"; ctx.font="15px system-ui,sans-serif";
      ctx.textAlign="center"; ctx.textBaseline="middle";
      ctx.fillText(stageMsg, VW/2, VH/2); ctx.textAlign="left"; ctx.textBaseline="bottom"; }
    return;
  }
  ctx.drawImage(img, view.ox, view.oy, img.naturalWidth*view.scale, img.naturalHeight*view.scale);
  ctx.lineWidth = 2; ctx.font = "600 11.5px ui-sans-serif,system-ui,sans-serif"; ctx.textBaseline="bottom";
  // AI ghost suggestions (dashed/translucent, under real boxes)
  ghosts.forEach(g=>{
    const c = g.mapped ? color(g.cls) : "#9aa3b2";
    const x=sx(g.x), y=sy(g.y), w=g.w*view.scale, h=g.h*view.scale;
    const a = 0.5 + 0.45*Math.min(1, g.conf||0);   // higher confidence -> more solid
    ctx.save();
    ctx.fillStyle = g.mapped ? "rgba(167,139,250,.10)" : "rgba(154,163,178,.10)"; ctx.fillRect(x,y,w,h);
    ctx.setLineDash([6,4]); ctx.globalAlpha=a; ctx.lineWidth=2; ctx.strokeStyle=c; ctx.strokeRect(x,y,w,h);
    ctx.globalAlpha=1; ctx.setLineDash([]);
    const lab = `${g.name}${g.mapped?"":" ?"} ${Math.round((g.conf||0)*100)}%`;
    const tw=ctx.measureText(lab).width+12; const tx=Math.min(x,x+w), ty=Math.min(y,y+h);
    ctx.shadowColor="rgba(0,0,0,.4)"; ctx.shadowBlur=5; ctx.shadowOffsetY=1;
    ctx.globalAlpha=.92; ctx.fillStyle=c; rr(tx,ty-17,tw,16,4); ctx.globalAlpha=1;
    ctx.shadowColor="transparent";
    ctx.fillStyle="#0a0b0e"; ctx.fillText(lab,tx+6,ty-3);
    ctx.restore();
  });
  boxes.forEach((b,i)=>{
    const c = color(b.cls);
    const x=sx(b.x), y=sy(b.y), w=b.w*view.scale, h=b.h*view.scale;
    if(i===sel){ ctx.fillStyle = "rgba(110,123,255,.10)"; ctx.fillRect(x,y,w,h); }
    else if(i===hover && mode===null){ ctx.fillStyle = "rgba(255,255,255,.06)"; ctx.fillRect(x,y,w,h); }
    ctx.save();
    if(i===sel){ ctx.shadowColor=c; ctx.shadowBlur=11; }
    else if(i===hover && mode===null){ ctx.shadowColor=c; ctx.shadowBlur=6; }
    ctx.strokeStyle = c; ctx.lineWidth = (i===sel||(i===hover&&mode===null))?2.5:2; ctx.strokeRect(x,y,w,h);
    ctx.restore();
    const nm = (DS.names&&DS.names[b.cls])!=null ? DS.names[b.cls] : b.cls;
    const lab = String(nm);
    const tw = ctx.measureText(lab).width+12;
    const tx = Math.min(x, x+w), ty = Math.min(y, y+h);
    ctx.save();
    ctx.shadowColor="rgba(0,0,0,.4)"; ctx.shadowBlur=5; ctx.shadowOffsetY=1;
    ctx.fillStyle = c; rr(tx, ty-17, tw, 16, 4);
    ctx.restore();
    ctx.fillStyle = "#0a0b0e"; ctx.fillText(lab, tx+6, ty-3);
  });
  polys.forEach((p,i)=>{
    const c = color(p.cls), pts = p.pts;
    ctx.save();
    ctx.beginPath();
    for(let k=0;k<pts.length;k+=2){ const X=sx(pts[k]), Y=sy(pts[k+1]); if(k===0) ctx.moveTo(X,Y); else ctx.lineTo(X,Y); }
    ctx.closePath();
    ctx.fillStyle = i===selPoly ? "rgba(110,123,255,.18)" : "rgba(124,131,255,.13)";
    ctx.fill();
    if(i===selPoly){ ctx.shadowColor=c; ctx.shadowBlur=10; }
    ctx.lineWidth = i===selPoly?2.5:2; ctx.strokeStyle=c; ctx.stroke();
    ctx.restore();
    if(i===selPoly){ ctx.fillStyle="#fff"; ctx.strokeStyle="#6e7bff"; ctx.lineWidth=1.2;
      for(let k=0;k<pts.length;k+=2){ const X=sx(pts[k]),Y=sy(pts[k+1]); ctx.beginPath(); ctx.arc(X,Y,3,0,6.2832); ctx.fill(); ctx.stroke(); } }
    const nm = (DS.names&&DS.names[p.cls])!=null ? DS.names[p.cls] : p.cls;
    let mnx=1e9,mny=1e9; for(let k=0;k<pts.length;k+=2){ if(pts[k]<mnx)mnx=pts[k]; if(pts[k+1]<mny)mny=pts[k+1]; }
    const lab=String(nm), tw=ctx.measureText(lab).width+12, tx=sx(mnx), ty=sy(mny);
    ctx.save(); ctx.shadowColor="rgba(0,0,0,.4)"; ctx.shadowBlur=5; ctx.shadowOffsetY=1;
    ctx.fillStyle=c; rr(tx,ty-17,tw,16,4); ctx.restore();
    ctx.fillStyle="#0a0b0e"; ctx.fillText(lab,tx+6,ty-3);
  });
  if(sel>=0) drawHandles(boxes[sel]);
  if(cursor && (mode===null||mode==='new')){
    ctx.save(); ctx.strokeStyle='rgba(110,123,255,.4)'; ctx.lineWidth=1;
    ctx.beginPath();
    ctx.moveTo(cursor.x+0.5,0); ctx.lineTo(cursor.x+0.5,VH);
    ctx.moveTo(0,cursor.y+0.5); ctx.lineTo(VW,cursor.y+0.5);
    ctx.stroke(); ctx.restore();
  }
  updateProgress();
}
function handlePts(b){
  const x=b.x,y=b.y,w=b.w,h=b.h;
  return {nw:[x,y], n:[x+w/2,y], ne:[x+w,y], e:[x+w,y+h/2],
          se:[x+w,y+h], s:[x+w/2,y+h], sw:[x,y+h], w:[x,y+h/2]};
}
function drawHandles(b){
  const pts = handlePts(b);
  ctx.fillStyle = "#fff"; ctx.strokeStyle = "#6e7bff"; ctx.lineWidth=1.5;
  HANDLES.forEach(k=>{
    const [hx,hy]=pts[k]; const px=sx(hx), py=sy(hy);
    ctx.beginPath(); ctx.rect(px-4,py-4,8,8); ctx.fill(); ctx.stroke();
  });
}
function hitHandle(b, mx, my){
  const pts = handlePts(b);
  for(const k of HANDLES){
    const [hx,hy]=pts[k];
    if(Math.abs(sx(hx)-mx)<=HR+2 && Math.abs(sy(hy)-my)<=HR+2) return k;
  }
  return null;
}
function hitBox(mx, my){
  const x=ix(mx), y=iy(my);
  for(let i=boxes.length-1;i>=0;i--){
    const b=boxes[i];
    const bx=Math.min(b.x,b.x+b.w), by=Math.min(b.y,b.y+b.h);
    if(x>=bx && x<=bx+Math.abs(b.w) && y>=by && y<=by+Math.abs(b.h)) return i;
  }
  return -1;
}

// ---- mouse ----
let mode=null, drag=null, spaceDown=false;
cv.addEventListener("pointerdown", e=>{
  cv.setPointerCapture(e.pointerId);
  const mx=e.offsetX, my=e.offsetY;
  if(spaceDown || e.button===1){ mode="pan"; drag={mx,my,ox:view.ox,oy:view.oy}; return; }
  if(!imgOk) return;
  if(sel>=0){
    const k = hitHandle(boxes[sel], mx, my);
    if(k){ snapStart(); mode="resize"; drag={k, b:boxes[sel]}; return; }
  }
  if(selPoly>=0){
    const vi = hitVertex(polys[selPoly], mx, my);
    if(vi>=0){
      if(e.altKey){ if(polys[selPoly].pts.length>6){ pushUndo(); polys[selPoly].pts.splice(vi*2,2); markDirty(); draw(); } return; }
      snapStart(); mode="vertex"; drag={vi}; return;
    }
  }
  const hit = hitBox(mx,my);
  if(hit>=0){
    snapStart(); sel=hit; selPoly=-1; mode="move";
    drag={mx, my, x:boxes[sel].x, y:boxes[sel].y};
    setActive(boxes[sel].cls); draw(); return;
  }
  const ph = hitPoly(mx,my);
  if(ph>=0){
    snapStart(); selPoly=ph; sel=-1; mode="movepoly";
    drag={mx, my, pts:polys[ph].pts.slice()};
    setActive(polys[ph].cls); draw(); return;
  }
  if(ghosts.length){
    const g = hitGhost(mx,my);
    if(g>=0){ if(e.altKey) rejectGhost(g); else acceptGhost(g); mode=null; drag=null; return; }
  }
  if(!editable || (DS && !DS.writable)) return;
  if(tool==="seg"){ segmentAt(mx,my); return; }
  const x=ix(mx), y=iy(my);
  snapStart();
  boxes.push({cls:active, x, y, w:0, h:0});
  sel=boxes.length-1; selPoly=-1; mode="new"; drag={};
  draw();
});
cv.addEventListener("pointermove", e=>{
  const mx=e.offsetX, my=e.offsetY;
  cursor={x:mx,y:my};
  if(mode==="pan"){ view.ox=drag.ox+(mx-drag.mx); view.oy=drag.oy+(my-drag.my); draw(); return; }
  if(mode==="new"){ const b=boxes[sel]; b.w=ix(mx)-b.x; b.h=iy(my)-b.y; draw(); return; }
  if(mode==="move"){ const dx=ix(mx)-ix(drag.mx), dy=iy(my)-iy(drag.my);
    boxes[sel].x=drag.x+dx; boxes[sel].y=drag.y+dy; markDirty(); draw(); return; }
  if(mode==="resize"){ resizeBox(drag.b, drag.k, ix(mx), iy(my)); markDirty(); draw(); return; }
  if(mode==="movepoly"){ const dx=ix(mx)-ix(drag.mx), dy=iy(my)-iy(drag.my); const p=polys[selPoly];
    for(let k=0;k<p.pts.length;k+=2){ p.pts[k]=drag.pts[k]+dx; p.pts[k+1]=drag.pts[k+1]+dy; } markDirty(); draw(); return; }
  if(mode==="vertex"){ const p=polys[selPoly]; p.pts[drag.vi*2]=ix(mx); p.pts[drag.vi*2+1]=iy(my); markDirty(); draw(); return; }
  let hb=-1;
  if(imgOk && sel>=0 && hitHandle(boxes[sel],mx,my)){ cv.style.cursor="pointer"; }
  else { hb = imgOk?hitBox(mx,my):-1; cv.style.cursor = spaceDown?"grab":(hb>=0?"move":"crosshair"); }
  hover = hb;
  draw();
});
cv.addEventListener("pointerleave", ()=>{ cursor=null; hover=-1; draw(); });
cv.addEventListener("pointerup", e=>{
  if(mode==="new"){
    const b=boxes[sel];
    if(Math.abs(b.w)*view.scale<3 || Math.abs(b.h)*view.scale<3){ boxes.pop(); sel=-1; }
    else { normalizeRect(b); clipToImage(b); markDirty(); }
  } else if(mode==="resize"){ normalizeRect(drag.b); clipToImage(drag.b); }
  else if(mode==="move"){ clipToImage(boxes[sel]); }
  else if(mode==="movepoly"){ clipPoly(polys[selPoly]); }
  else if(mode==="vertex"){ clipPoly(polys[selPoly]); }
  snapCommit();
  mode=null; drag=null; draw();
});
cv.addEventListener("dblclick", e=>{
  if(selPoly<0 || !imgOk) return;
  const mx=e.offsetX, my=e.offsetY, p=polys[selPoly], X=ix(mx), Y=iy(my), n=p.pts.length/2;
  let best=-1, bestD=1e18, bx=0, by=0;
  for(let i=0;i<n;i++){ const a=2*i, b=2*((i+1)%n);
    const ax=p.pts[a], ay=p.pts[a+1], cx=p.pts[b], cy=p.pts[b+1];
    const dx=cx-ax, dy=cy-ay, L=dx*dx+dy*dy||1;
    let t=((X-ax)*dx+(Y-ay)*dy)/L; t=Math.max(0,Math.min(1,t));
    const px=ax+t*dx, py=ay+t*dy, d=(X-px)*(X-px)+(Y-py)*(Y-py);
    if(d<bestD){ bestD=d; best=i; bx=px; by=py; } }
  if(best>=0 && Math.hypot(sx(bx)-mx, sy(by)-my)<14){ pushUndo(); p.pts.splice(2*(best+1),0, bx, by); markDirty(); draw(); }
});
function clipPoly(p){ if(!imgOk||!p) return; const iw=img.naturalWidth, ih=img.naturalHeight;
  for(let k=0;k<p.pts.length;k+=2){ p.pts[k]=Math.max(0,Math.min(p.pts[k],iw)); p.pts[k+1]=Math.max(0,Math.min(p.pts[k+1],ih)); } }
function normalizeRect(b){ if(b.w<0){b.x+=b.w;b.w=-b.w;} if(b.h<0){b.y+=b.h;b.h=-b.h;} }
function clipToImage(b){
  if(!imgOk) return;
  const iw=img.naturalWidth, ih=img.naturalHeight;
  const x1=Math.max(0,Math.min(b.x,iw)),     y1=Math.max(0,Math.min(b.y,ih));
  const x2=Math.max(0,Math.min(b.x+b.w,iw)), y2=Math.max(0,Math.min(b.y+b.h,ih));
  b.x=x1; b.y=y1; b.w=x2-x1; b.h=y2-y1;
}
function resizeBox(b, k, mx, my){
  if(k.includes("n")){ b.h += b.y-my; b.y=my; }
  if(k.includes("s")){ b.h = my-b.y; }
  if(k.includes("w")){ b.w += b.x-mx; b.x=mx; }
  if(k.includes("e")){ b.w = mx-b.x; }
}
cv.addEventListener("wheel", e=>{
  e.preventDefault();
  const f = e.deltaY<0 ? 1.1 : 1/1.1;
  const mx=e.offsetX, my=e.offsetY, bx=ix(mx), by=iy(my);
  view.scale = Math.max(0.02, Math.min(64, view.scale * f));
  view.ox = mx - bx*view.scale; view.oy = my - by*view.scale;
  draw();
}, {passive:false});

// ---- keyboard ----
window.addEventListener("keydown", e=>{
  const t=(e.target&&e.target.tagName)||"";
  if(t==="INPUT"||t==="SELECT"||t==="TEXTAREA"){ if(e.key==="Escape"){ e.target.blur(); closePicker(); } return; }
  if(e.key===" "){ spaceDown=true; cv.style.cursor="grab"; e.preventDefault(); return; }
  if((e.ctrlKey||e.metaKey) && (e.key==="s"||e.key==="S")){ e.preventDefault(); save(); return; }
  if((e.ctrlKey||e.metaKey) && (e.key==="z"||e.key==="Z")){ e.preventDefault();
    if(undoStack.length){ applyUndo(undoStack.pop()); sel=-1; selPoly=-1; markDirty(); draw(); } return; }
  if(e.key==="Enter"){
    if(ghosts.length){ e.preventDefault(); const adv=e.shiftKey; acceptAllGhosts();
      if(adv){ save().then(()=> nextSuggested()); } }
    return;
  }
  if(e.key>="0" && e.key<="9"){ const i = e.key==="0"?9:(+e.key-1); setActive(i); return; }
  if(e.key==="/"){ e.preventDefault(); togglePicker(); return; }
  if(e.key==="d"||e.key==="D"||e.key==="ArrowRight"){ e.preventDefault(); load(idx+1); return; }
  if(e.key==="a"||e.key==="A"||e.key==="ArrowLeft"){ e.preventDefault(); load(idx-1); return; }
  if(e.key==="e"||e.key==="E"){ e.preventDefault(); nextUnlabeled(e.shiftKey?-1:1); return; }
  if(e.key==="r"||e.key==="R"){ e.preventDefault(); prelabelCurrent(); return; }
  if(e.key==="b"||e.key==="B"){ setTool("box"); return; }
  if((e.key==="s"||e.key==="S") && assist && assist.sam){ setTool("seg"); return; }
  if(e.key==="Delete"||e.key==="Backspace"){
    if(selPoly>=0){ pushUndo(); polys.splice(selPoly,1); selPoly=-1; markDirty(); draw(); }
    else if(sel>=0){ pushUndo(); boxes.splice(sel,1); sel=-1; markDirty(); draw(); }
    return; }
  if(e.key==="f"||e.key==="F"){ fit(); draw(); return; }
  if(e.key==="?"){ toggleHelp(); return; }
  if(e.key==="Escape"){
    if($("#picker").classList.contains("show")){ closePicker(); }
    else if(ghosts.length){ clearGhosts(); }
    else if($("#help").style.display==="flex"){ $("#help").style.display="none"; }
    else if(mode==="new"){ boxes.pop(); sel=-1; mode=null; gestureSnap=null; draw(); }
    else { sel=-1; selPoly=-1; draw(); }
  }
});
window.addEventListener("keyup", e=>{ if(e.key===" "){ spaceDown=false; cv.style.cursor="crosshair"; } });
window.addEventListener("resize", resizeCanvas);
window.addEventListener("beforeunload", e=>{ if(dirty){ e.preventDefault(); e.returnValue=""; } });

init().catch(err=>{ document.body.insertAdjacentHTML("afterbegin",
  `<div style="padding:14px;color:#f5b13d">LibreLabel failed to start: ${err.message}</div>`); });
</script>
</body>
</html>
"""
