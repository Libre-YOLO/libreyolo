"""Static markup: topbar, sidebar, stage, panels, modals, home, wizard."""

PART = r"""</style>
</head>
<body>
<div id="app">
  <header class="topbar">
    <span class="brand" id="brandhome" title="Projects - back to the home screen" style="cursor:pointer"><svg class="ic" viewBox="0 0 400 400" fill="currentColor" fill-rule="evenodd" aria-label="LibreYOLO"><path d="M86 86 L123.09 86 A78 78 0 0 1 276.91 86 L314 86 L314 123.09 A78 78 0 0 1 314 276.91 L314 314 L276.91 314 A78 78 0 0 1 123.09 314 L86 314 L86 276.91 A78 78 0 0 1 86 123.09 Z M116 116 L157.26 116 A46 46 0 1 1 242.74 116 L284 116 L284 157.26 A46 46 0 1 1 284 242.74 L284 284 L242.74 284 A46 46 0 1 1 157.26 284 L116 284 L116 242.74 A46 46 0 1 1 116 157.26 Z"/></svg>Libre<b>Label</b></span>
    <span class="sep"></span>
    <span class="ds" id="dsname"></span>
    <span class="counter" id="counter"></span>
    <button class="insbtn" id="classesbtn" title="Edit classes - rename or add" style="display:none"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.6 13.4l-7.2 7.2a2 2 0 0 1-2.8 0l-7-7A2 2 0 0 1 3 12.2V5a2 2 0 0 1 2-2h7.2a2 2 0 0 1 1.4.6l7 7a2 2 0 0 1 0 2.8z"/><circle cx="7.5" cy="7.5" r="1.5" fill="currentColor" stroke="none"/></svg></button>
    <span class="grow"></span>
    <span class="ai" id="assistbar" style="display:none">
      <button class="btn btn-primary" id="aautolabel"><svg class="ic" viewBox="0 0 24 24" fill="currentColor"><path d="M11.5 2.5l1.6 4.4 4.4 1.6-4.4 1.6-1.6 4.4-1.6-4.4-4.4-1.6 4.4-1.6z"/><path d="M18.5 14l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/></svg>Auto-label all</button>
      <button class="btn btn-ghost btn-sm" id="aprelabel" title="Auto-label this image (R)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 4V2M19 8h2M5 20l9-9M14 6l4 4"/></svg>R</button>
      <button class="btn btn-ghost btn-sm" id="asam" title="Smart-segment with SAM (S)" style="display:none"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12a8 8 0 1 1 8 8"/><circle cx="12" cy="12" r="2.5" fill="currentColor" stroke="none"/></svg>SAM</button>
      <button class="btn btn-ghost btn-sm" id="aradar" title="Label-Error Radar - audit your accepted labels (Y)" style="display:none"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19.07 4.93A10 10 0 1 0 21 12"/><path d="M12 12l6.5-4.5"/><circle cx="12" cy="12" r="3"/></svg>Radar</button>
      <input id="laprompt" class="laprompt" placeholder="objects to find - e.g. person, helmet" style="display:none">
      <span class="field" id="aconffield"><span>conf</span><input type="range" id="aconf" min="0.05" max="0.9" step="0.05"><b id="aconfval">0.25</b></span>
      <select id="amodel" class="select"></select>
    </span>
    <span class="save" id="save"></span>
    <span class="chip" id="boostchip"></span>
    <span class="sep"></span>
    <span class="tgroup">
      <button class="insbtn" id="insbtn" title="Dataset insights &amp; readiness"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20V11M10 20V4M16 20v-6M21 20H3"/></svg></button>
      <button class="btn btn-ghost btn-sm" id="exportbtn" title="Export dataset (YOLO / COCO / Pascal VOC)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12M8 11l4 4 4-4"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>Export</button>
      <button class="insbtn" id="mapbtn" title="Embedding map - see your whole dataset (M)" style="display:none"><svg class="ic" viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="6" cy="7.5" r="1.7"/><circle cx="9.5" cy="15" r="1.7"/><circle cx="15" cy="9" r="1.7"/><circle cx="18" cy="16.5" r="1.7"/><circle cx="13" cy="18" r="1.5"/><circle cx="17.5" cy="6" r="1.4"/></svg></button>
      <button class="insbtn" id="boostbtn" title="Boost - fine-tune the model on your labels" style="display:none"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 7-7"/><path d="M14 8h6v6"/></svg></button>
    </span>
    <span class="sep"></span>
    <span class="tgroup">
      <button class="insbtn" id="homebtn" title="Projects - back to the home screen"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg></button>
      <button class="insbtn" id="sharebtn" title="Share - label together with teammates"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="2.4"/><circle cx="6" cy="12" r="2.4"/><circle cx="18" cy="19" r="2.4"/><path d="M8.1 10.9l7.8-4.8M8.1 13.1l7.8 4.8"/></svg></button>
    </span>
    <span class="sep"></span>
    <span class="tgroup">
      <button class="insbtn" id="themebtn" title="Toggle light / dark"></button>
      <button class="btn-icon" id="helpbtn" title="Keyboard shortcuts (?)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10"/></svg></button>
    </span>
  </header>
  <main>
    <aside class="sidebar">
      <div class="side-head">
        <div class="seg" id="filter">
          <button data-f="all" class="on">All</button>
          <button data-f="todo">To-do</button>
          <button data-f="review">Review</button>
        </div>
        <div class="sidesearch"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg><input id="imgsearch" placeholder="Filter by filename…" autocomplete="off"></div>
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
        <button class="tool" id="toolPoly" title="Polygon (P)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 6-3 9H7L4 9z"/></svg></button>
        <button class="tool" id="toolObb" title="Oriented box (O)" style="display:none"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="5" width="14" height="14" rx="1.5" transform="rotate(20 12 12)"/></svg></button>
        <button class="tool" id="toolSeg" title="Smart segment - SAM click-to-mask (S)" style="display:none"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12a8 8 0 1 1 8 8"/><path d="M12 20a8 8 0 0 1-8-8" stroke-dasharray="2 3"/><circle cx="12" cy="12" r="2.5" fill="currentColor" stroke="none"/></svg></button>
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
      <div class="bottombar" id="bottombar">
        <div class="labelbar" id="labelbar"></div>
        <button class="submitbtn" id="submitbtn" title="Submit & go to next image (Enter)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12l5 5L20 6"/></svg><span>Submit</span><kbd>&#9166;</kbd></button>
      </div>
      <div class="banner" id="banner"></div>
      <div class="progress" id="progress"><div class="pcard">
        <svg class="ic" viewBox="0 0 24 24" fill="currentColor"><path d="M11.5 2.5l1.6 4.4 4.4 1.6-4.4 1.6-1.6 4.4-1.6-4.4-4.4-1.6 4.4-1.6z"/><path d="M18.5 14l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/></svg>
        <div class="ptitle">Auto-labeling with your model</div>
        <div class="ptxt" id="ptxt"></div>
        <div class="ptrack"><div class="pbar" id="pbar"></div></div>
      </div></div>
      <div class="help" id="help"><div class="card2">
        <div class="hh">
          <h3><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10"/></svg>Keyboard shortcuts</h3>
          <button class="mx" id="helpclose">&times;</button>
        </div>
        <p class="help-sub">Everything you can do without leaving the canvas.</p>
        <div class="kgrid">
          <div class="kgroup"><h4>Drawing &amp; tools</h4>
            <div class="krow"><span>Draw a box (active class)</span><span class="keys"><kbd>drag</kbd></span></div>
            <div class="krow"><span>Box / polygon / smart-segment</span><span class="keys"><kbd>B</kbd><kbd>P</kbd><kbd>S</kbd></span></div>
            <div class="krow"><span>Tighten box to edges</span><span class="keys"><kbd>T</kbd></span></div>
            <div class="krow"><span>Loupe magnifier</span><span class="keys"><kbd>L</kbd></span></div>
          </div>
          <div class="kgroup"><h4>Classes</h4>
            <div class="krow"><span>Set active class</span><span class="keys"><kbd>1</kbd>..<kbd>9</kbd><kbd>0</kbd></span></div>
            <div class="krow"><span>Open class search</span><span class="keys"><kbd>/</kbd></span></div>
          </div>
          <div class="kgroup"><h4>Select &amp; edit</h4>
            <div class="krow"><span>Select / move / resize</span><span class="keys"><kbd>click</kbd></span></div>
            <div class="krow"><span>Add / remove from selection</span><span class="keys"><kbd>Shift</kbd><kbd>click</kbd></span></div>
            <div class="krow"><span>Select all</span><span class="keys"><kbd>Ctrl</kbd><kbd>A</kbd></span></div>
            <div class="krow"><span>Delete selected</span><span class="keys"><kbd>Del</kbd></span></div>
            <div class="krow"><span>Undo / redo</span><span class="keys"><kbd>Ctrl</kbd><kbd>Z</kbd><kbd>Ctrl</kbd><kbd>Y</kbd></span></div>
            <div class="krow"><span>Duplicate box</span><span class="keys"><kbd>Ctrl</kbd><kbd>D</kbd></span></div>
          </div>
          <div class="kgroup"><h4>Navigate</h4>
            <div class="krow"><span>Previous / next image</span><span class="keys"><kbd>A</kbd><kbd>D</kbd></span></div>
            <div class="krow"><span>Next unlabeled</span><span class="keys"><kbd>E</kbd></span></div>
            <div class="krow"><span>Copy previous labels</span><span class="keys"><kbd>C</kbd></span></div>
            <div class="krow"><span>Submit &amp; next unlabeled</span><span class="keys"><kbd>Enter</kbd></span></div>
          </div>
          <div class="kgroup"><h4>View</h4>
            <div class="krow"><span>Pan</span><span class="keys"><kbd>Space</kbd><kbd>drag</kbd></span></div>
            <div class="krow"><span>Zoom</span><span class="keys"><kbd>wheel</kbd><kbd>+</kbd><kbd>−</kbd></span></div>
            <div class="krow"><span>Fit to view</span><span class="keys"><kbd>F</kbd></span></div>
          </div>
          <div class="kgroup"><h4>AI assist</h4>
            <div class="krow"><span>Auto-label this image</span><span class="keys"><kbd>R</kbd></span></div>
            <div class="krow"><span>Label-Error Radar</span><span class="keys"><kbd>Y</kbd></span></div>
            <div class="krow"><span>Next flagged</span><span class="keys"><kbd>N</kbd></span></div>
            <div class="krow"><span>Embedding map</span><span class="keys"><kbd>M</kbd></span></div>
          </div>
          <div class="kgroup"><h4>Save</h4>
            <div class="krow"><span>Save</span><span class="keys"><kbd>Ctrl</kbd><kbd>S</kbd></span></div>
            <div class="krow"><span>Cancel / clear / close</span><span class="keys"><kbd>Esc</kbd></span></div>
          </div>
        </div>
      </div></div>
    </div>
    <aside class="regions" id="regionspanel">
      <div class="rp-head">Regions <b id="rpcount">0</b></div>
      <div class="rp-list" id="rplist"></div>
    </aside>
  </main>
  <div class="modal" id="insights"><div class="mcard">
    <div class="mhead"><h3>Dataset insights</h3><button class="mx" id="insclose">&times;</button></div>
    <div class="mbody" id="insbody"></div>
  </div></div>
  <div class="modal" id="radar"><div class="mcard">
    <div class="mhead"><h3>Label-Error Radar</h3><button class="mx" id="radarclose">&times;</button></div>
    <div class="mbody" id="radarbody"></div>
  </div></div>
  <div class="modal" id="mapmodal"><div class="mcard mapcard">
    <div class="mhead"><h3>Embedding map</h3><button class="mx" id="mapclose">&times;</button></div>
    <div class="mapstage">
      <canvas id="mapcv"></canvas>
      <div class="maplegend"><span><i style="background:#10b981"></i>labeled</span><span><i style="background:#22d3ee"></i>to review</span><span><i style="background:#64748b"></i>unlabeled</span></div>
      <div class="maphint" id="maphint">Drag a lasso around a region → jump to those images</div>
    </div>
  </div></div>
  <div class="modal" id="classedit"><div class="mcard cecard">
    <div class="mhead"><h3>Classes</h3><button class="mx" id="ceclose">&times;</button></div>
    <div class="mbody">
      <p class="ce-note">Rename a class, or add new ones. Existing classes keep their slot - you can't delete or reorder here (that would relabel boxes you've already drawn).</p>
      <div id="celist"></div>
      <button class="btn btn-ghost ce-add" id="ceadd">+ Add class</button>
      <div class="ce-err" id="ceerr"></div>
      <div class="ce-actions"><button class="btn btn-ghost" id="cecancel">Cancel</button><button class="btn btn-primary" id="cesave">Save</button></div>
    </div>
  </div></div>
  <div class="modal" id="renamemodal"><div class="mcard cecard">
    <div class="mhead"><h3>Rename project</h3><button class="mx" id="renclose">&times;</button></div>
    <div class="mbody">
      <input id="reninput" class="ce-in" style="width:100%" placeholder="Project name" spellcheck="false" autocomplete="off">
      <div class="ce-err" id="renerr"></div>
      <div class="ce-actions"><button class="btn btn-ghost" id="rencancel">Cancel</button><button class="btn btn-primary" id="rensave">Save</button></div>
    </div>
  </div></div>
  <div class="modal" id="deletemodal"><div class="mcard cecard">
    <div class="mhead"><h3>Delete dataset</h3><button class="mx" id="delclose">&times;</button></div>
    <div class="mbody">
      <p class="ce-note">This moves the whole project folder to LibreLabel's trash (<code>~/.librelabel/trash</code>). It is recoverable, but your images move with it. Nothing is erased.</p>
      <p class="ce-note" id="delpath" style="color:var(--tx);font:12px ui-monospace,monospace;word-break:break-all"></p>
      <div class="ce-err" id="delerr"></div>
      <div class="ce-actions"><button class="btn btn-ghost" id="delcancel">Cancel</button><button class="btn" id="delconfirm" style="background:var(--danger);color:#fff;border:1px solid var(--danger)">Move to trash</button></div>
    </div>
  </div></div>
  <div class="modal" id="exportmodal"><div class="mcard cecard">
    <div class="mhead"><h3>Export dataset</h3><button class="mx" id="expclose">&times;</button></div>
    <div class="mbody">
      <label class="wz-lbl">Split</label>
      <div class="seg" id="expsplit">
        <button data-s="trainval" class="on">Train / Val</button>
        <button data-s="trainvaltest">Train / Val / Test</button>
        <button data-s="none">No split</button>
      </div>
      <div class="exp-ratios" id="expratios">
        <label>Val %<input type="number" id="expval" value="20" min="1" max="90"></label>
        <label id="exptestwrap" hidden>Test %<input type="number" id="exptest" value="10" min="0" max="90"></label>
        <label>Seed<input type="number" id="expseed" value="1234"></label>
      </div>
      <label class="wz-check" style="margin-top:14px"><input type="checkbox" id="expinplace"> Re-split in place (reorganize this folder, no copy)</label>
      <p class="wz-note" id="expinplacewarn" hidden>Moves your images into images/train|val(/test) and rewrites data.yaml - it changes your working folder (YOLO only).</p>
      <div id="expcopyopts">
        <label class="wz-lbl">Formats</label>
        <div class="exp-fmts">
          <label class="wz-check"><input type="checkbox" id="expyolo" checked> YOLO</label>
          <label class="wz-check"><input type="checkbox" id="expcoco"> COCO</label>
          <label class="wz-check"><input type="checkbox" id="expvoc"> Pascal VOC</label>
        </div>
        <label class="wz-lbl">Destination folder</label>
        <div class="wz-folder"><input id="expdst" class="wz-in" placeholder="Where to write the exported dataset…" spellcheck="false" autocomplete="off"><button class="btn btn-ghost" id="expbrowse">Browse</button></div>
        <label class="wz-check" style="margin-top:10px"><input type="checkbox" id="expzip"> Also create a .zip</label>
      </div>
      <div class="ce-err" id="experr"></div>
      <div class="exp-result" id="expresult"></div>
      <div class="ce-actions"><button class="btn btn-ghost" id="expcancel">Cancel</button><button class="btn btn-primary" id="expgo">Export</button></div>
    </div>
  </div></div>
  <div id="home" class="home">
    <button class="insbtn home-theme" id="hometheme" title="Toggle light / dark"></button>
    <div class="home-inner">
      <div class="home-hero">
        <span class="home-brand"><svg class="ic" viewBox="0 0 400 400" fill="currentColor" fill-rule="evenodd" aria-label="LibreYOLO"><path d="M86 86 L123.09 86 A78 78 0 0 1 276.91 86 L314 86 L314 123.09 A78 78 0 0 1 314 276.91 L314 314 L276.91 314 A78 78 0 0 1 123.09 314 L86 314 L86 276.91 A78 78 0 0 1 86 123.09 Z M116 116 L157.26 116 A46 46 0 1 1 242.74 116 L284 116 L284 157.26 A46 46 0 1 1 284 242.74 L284 284 L242.74 284 A46 46 0 1 1 157.26 284 L116 284 L116 242.74 A46 46 0 1 1 116 157.26 Z"/></svg>Libre<b>Label</b></span>
        <p class="home-tag">Label your images and export to YOLO, COCO or Pascal VOC - your data, your machine, nothing uploaded.</p>
      </div>
      <div class="home-actions">
        <button class="btn btn-primary" id="homenew"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"/></svg>New project</button>
        <span id="homeresume" style="display:none"><button class="btn btn-ghost" id="homeresumebtn"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg>Resume current project</button></span>
      </div>
      <div class="home-err" id="homeerr"></div>
      <div class="home-sec">Your projects</div>
      <div class="home-grid" id="homegrid"></div>
    </div>
  </div>
  <div id="wizard" class="wz">
    <div class="wz-card">
      <header class="wz-head">
        <span class="wz-title">New project</span>
        <span class="wz-steps">
          <span class="wz-step on" data-s="1"><i>1</i>Data</span>
          <span class="wz-step" data-s="2"><i>2</i>Labels</span>
        </span>
        <button class="mx" id="wzclose">&times;</button>
      </header>
      <div class="wz-body">
        <section class="wz-pane" data-pane="1">
          <label class="wz-lbl">Project name <span class="wz-hint">optional, defaults to the folder name</span></label>
          <input id="wzname" class="wz-in" placeholder="e.g. Helmet detection" spellcheck="false" autocomplete="off">
          <label class="wz-lbl">Images <span class="req">*</span></label>
          <div class="wz-tabs">
            <button class="wz-tab on" data-tab="upload">Upload</button>
            <button class="wz-tab" data-tab="existing">Open existing</button>
          </div>
          <div class="wz-tabpane" data-tabpane="upload">
            <div class="wz-folder"><input id="wzdst" class="wz-in" placeholder="Destination folder for the dataset…" spellcheck="false" autocomplete="off"><button class="btn btn-ghost" id="wzbrowse">Browse</button></div>
            <div class="wz-drop" id="wzdrop">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M5 20h14"/></svg>
              <div class="wz-drop-t">Drag &amp; drop images here</div>
              <div class="wz-drop-s">or <button class="wz-link" id="wzpick">click to browse</button> · jpg, png, bmp, webp</div>
              <input type="file" id="wzfiles" accept="image/*" multiple hidden>
            </div>
            <div class="wz-files" id="wzfilelist"></div>
          </div>
          <div class="wz-tabpane" data-tabpane="existing" hidden>
            <div class="wz-folder"><input id="wzexist" class="wz-in" placeholder="Folder of images, or a data.yaml…" spellcheck="false" autocomplete="off"><button class="btn btn-ghost" id="wzexistbrowse">Browse</button></div>
            <p class="wz-note">LibreLabel labels the images where they already are and writes the dataset config for you. Nothing is copied or moved.</p>
          </div>
        </section>
        <section class="wz-pane" data-pane="2" hidden>
          <label class="wz-lbl">Task</label>
          <div class="wz-tasks" id="wztasks">
            <button class="wz-task on" data-task="detect">Bounding boxes</button>
            <button class="wz-task" data-task="obb">Oriented boxes</button>
            <button class="wz-task" data-task="segment" disabled>Segmentation <span class="soon">soon</span></button>
            <button class="wz-task" data-task="classify" disabled>Classification <span class="soon">soon</span></button>
          </div>
          <label class="wz-lbl">Classes <span class="wz-hint">optional, you can add or rename them anytime while labeling</span></label>
          <div class="wz-classes" id="wzclasses"></div>
          <button class="btn btn-ghost wz-addcls" id="wzaddcls">+ Add class</button>
        </section>
      </div>
      <footer class="wz-foot">
        <div class="wz-err" id="wzerr"></div>
        <div class="wz-nav">
          <button class="btn btn-ghost" id="wzback" hidden>Back</button>
          <button class="btn btn-primary" id="wznext">Next</button>
          <button class="btn btn-primary" id="wzcreate" hidden>Create project</button>
        </div>
      </footer>
    </div>
  </div>
  <div class="pop" id="sharepop"></div>
</div>
<script>
"""
