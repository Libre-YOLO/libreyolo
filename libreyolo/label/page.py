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
    /* LibreYOLO theme — dark (slate + cyan), matching the website */
    --bg:#020617; --bg2:#0b1120;
    --s1:#0f172a; --s2:#1e293b; --s3:#334155;
    --line:#1e293b; --line2:#334155;
    --tx:#e2e8f0; --tx2:#94a3b8; --tx3:#64748b;
    --ac:#06b6d4; --ac-ink:#012a33; --ai:#22d3ee;
    --ok:#10b981; --warn:#fbbf24; --danger:#ef4444;
    --r:10px; --r2:8px; --sh:0 12px 34px rgba(2,6,23,.55); --shs:0 2px 8px rgba(2,6,23,.4);
    --stage1:#0b1120; --stage2:#020617; --topbar1:#0f172a; --topbar2:#0b1120;
    --acg1:#22d3ee; --acg2:#0891b2; --glass:rgba(15,23,42,.85);
  }
  :root.light{
    --bg:#fafbfd; --bg2:#f1f5f9;
    --s1:#ffffff; --s2:#f8fafc; --s3:#eef2f7;
    --line:#e2e8f0; --line2:#cbd5e1;
    --tx:#1e293b; --tx2:#475569; --tx3:#64748b;
    --ac:#0891b2; --ac-ink:#ffffff; --ai:#0e7490;
    --ok:#059669; --warn:#d97706; --danger:#dc2626;
    --sh:0 10px 30px rgba(15,23,42,.12); --shs:0 2px 8px rgba(15,23,42,.08);
    --stage1:#eef2f7; --stage2:#dbe3ee; --topbar1:#ffffff; --topbar2:#f8fafc;
    --acg1:#22c3e0; --acg2:#0891b2; --glass:rgba(255,255,255,.85);
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--tx);
    font:13px/1.5 "Outfit",ui-sans-serif,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
  button{font:inherit;color:inherit;cursor:pointer}
  .ic{width:16px;height:16px;display:block;flex:none}
  #app{display:grid;grid-template-rows:52px 1fr;height:100vh}
  /* topbar */
  .topbar{display:flex;align-items:center;gap:12px;padding:0 14px;
    background:linear-gradient(180deg,var(--topbar1),var(--topbar2));border-bottom:1px solid var(--line)}
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
  .btn-primary{background:linear-gradient(180deg,var(--acg1),var(--acg2));color:var(--ac-ink);
    box-shadow:0 1px 0 rgba(255,255,255,.18) inset,0 5px 16px rgba(6,182,212,.34)}
  .btn-primary:hover{filter:brightness(1.07);transform:translateY(-1px)}
  .btn-primary:active{transform:translateY(0)}
  .btn-ghost{background:var(--s2);border-color:var(--line2);color:var(--tx)}
  .btn-ghost:hover{background:var(--s3)}
  .btn-sm{height:30px;padding:0 11px;font-size:12px}
  .btn-icon{display:grid;place-items:center;width:32px;height:32px;border-radius:var(--r2);
    background:transparent;border:1px solid transparent;color:var(--tx2);transition:.15s}
  .btn-icon:hover{background:var(--s2);color:var(--tx);border-color:var(--line)}
  .ai{display:flex;align-items:center;gap:7px;padding:4px 6px;border-radius:12px;
    background:rgba(6,182,212,.10);border:1px solid rgba(6,182,212,.26)}
  :root.light .ai{background:rgba(8,145,178,.09);border-color:rgba(8,145,178,.32)}
  .tgroup{display:inline-flex;align-items:center;gap:3px}
  .ai .field{display:flex;align-items:center;gap:8px;height:32px;padding:0 11px;border-radius:var(--r2);
    background:var(--s1);border:1px solid var(--line);color:var(--tx3);font-size:12px}
  .ai .field b{color:var(--tx);font-variant-numeric:tabular-nums;min-width:28px;text-align:right}
  .ai input[type=range]{-webkit-appearance:none;appearance:none;width:92px;height:4px;border-radius:9px;background:var(--s3);outline:none}
  .ai input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;
    background:var(--ai);cursor:pointer;box-shadow:0 0 0 3px rgba(34,211,238,.2)}
  .select{height:32px;border-radius:var(--r2);background:var(--s1);color:var(--tx2);
    border:1px solid var(--line);padding:0 8px;font-size:12px;max-width:170px}
  .laprompt{height:32px;width:210px;border-radius:var(--r2);background:var(--s1);color:var(--tx);
    border:1px solid var(--line);padding:0 11px;font-size:12px;outline:none}
  .laprompt:focus{border-color:var(--ac)}
  .save{display:inline-flex;align-items:center;gap:7px;height:30px;padding:0 12px;border-radius:999px;
    border:1px solid var(--line);color:var(--tx3);font-size:12px;font-weight:540}
  .save::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
  .save.dirty{color:var(--warn);border-color:rgba(245,177,61,.35);background:rgba(245,177,61,.08)}
  .save.saved{color:var(--ok);border-color:rgba(45,212,167,.35);background:rgba(45,212,167,.08)}
  .save.fail{color:var(--danger);border-color:color-mix(in srgb,var(--danger) 40%,transparent);background:rgba(239,68,68,.08)}
  @keyframes pop{0%{transform:scale(1)}40%{transform:scale(1.14)}100%{transform:scale(1)}}
  .save.flash{animation:pop .42s ease-out}
  /* main */
  main{display:grid;grid-template-columns:300px 1fr;min-height:0}
  main.has-regions{grid-template-columns:300px 1fr 234px}
  .regions{display:none;flex-direction:column;min-height:0;background:var(--bg2);border-left:1px solid var(--line)}
  main.has-regions .regions{display:flex}
  .rp-head{padding:13px 13px;border-bottom:1px solid var(--line);font-size:12px;font-weight:600;color:var(--tx2);display:flex;gap:7px;align-items:center}
  .rp-head b{color:var(--tx);font-variant-numeric:tabular-nums}
  .rp-list{flex:1;overflow-y:auto;padding:8px}
  .rprow{display:flex;align-items:center;gap:9px;padding:7px 9px;border-radius:8px;border:1px solid transparent;cursor:pointer;transition:.1s}
  .rprow:hover{background:var(--s2);border-color:var(--line)}
  .rprow.on{background:var(--s3);border-color:var(--ac)}
  .rprow .sw{width:12px;height:12px;border-radius:3px;flex:none}
  .rprow .rpn{flex:1;font-size:12.5px;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .rprow .rpi{font-size:11px;color:var(--tx3);font-variant-numeric:tabular-nums}
  .rprow .rpx{opacity:0;color:var(--tx3);font-size:16px;line-height:1;padding:0 3px;border-radius:5px}
  .rprow:hover .rpx{opacity:.55} .rprow .rpx:hover{opacity:1;color:var(--danger);background:var(--s1)}
  .rp-empty{padding:18px 12px;color:var(--tx3);font-size:12px;text-align:center}
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
  .modal{position:fixed;inset:0;display:none;align-items:center;justify-content:center;background:rgba(8,9,12,.82);backdrop-filter:blur(4px);z-index:20}
  .modal.show{display:flex}
  .mcard{width:min(680px,92vw);max-height:84vh;display:flex;flex-direction:column;background:var(--s1);border:1px solid var(--line2);border-radius:16px;box-shadow:var(--sh);overflow:hidden}
  .mhead{display:flex;align-items:center;justify-content:space-between;padding:15px 20px;border-bottom:1px solid var(--line)}
  .mhead h3{margin:0;font-size:15px}
  .mx{display:grid;place-items:center;width:30px;height:30px;border-radius:8px;background:var(--s2);border:1px solid var(--line);color:var(--tx2);font-size:17px;line-height:1}
  .mx:hover{background:var(--s3);color:var(--tx)}
  .mbody{padding:18px 20px;overflow:auto}
  .iload{color:var(--tx3);text-align:center;padding:24px}
  .igrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(108px,1fr));gap:10px;margin-bottom:18px}
  .icard{background:var(--s2);border:1px solid var(--line);border-radius:10px;padding:11px 13px}
  .icard .ik{font-size:10.5px;color:var(--tx3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px}
  .icard .iv{font-size:21px;font-weight:680;font-variant-numeric:tabular-nums;letter-spacing:-.2px}
  .isec{margin-bottom:18px}
  .isec .ititle{font-size:10.5px;color:var(--tx3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px}
  .ibar{display:flex;align-items:center;gap:10px;margin-bottom:6px;font-size:12px}
  .ibar .il{width:96px;color:var(--tx2);font-variant-numeric:tabular-nums}
  .ibar .it{flex:1;height:8px;background:var(--s2);border-radius:9px;overflow:hidden}
  .ibar .it span{display:block;height:100%;background:linear-gradient(90deg,var(--ac),var(--ai));border-radius:9px}
  .ibar .ic{width:34px;text-align:right;color:var(--tx3);font-variant-numeric:tabular-nums}
  .iok{display:flex;align-items:center;gap:8px;color:var(--ok);font-size:13px}
  .iok .ic{width:16px;height:16px}
  .iwarn{padding:10px 12px;border-radius:9px;background:var(--s2);border:1px solid var(--line);color:var(--tx2);font-size:12.5px;margin-bottom:12px}
  .iwarn.leak{background:rgba(251,113,113,.09);border-color:rgba(251,113,113,.35);color:var(--danger);font-weight:560}
  .idup{display:flex;align-items:center;gap:5px;margin-bottom:8px;flex-wrap:wrap}
  .ithumb{width:46px;height:36px;object-fit:cover;border-radius:5px;background:var(--s3)}
  .idsplit{margin-left:4px;font-size:11px;color:var(--tx3)}
  .insbtn{display:grid;place-items:center;width:32px;height:32px;border-radius:8px;background:transparent;border:1px solid transparent;color:var(--tx2)}
  .insbtn:hover{background:var(--s2);color:var(--tx);border-color:var(--line)}
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
    background:radial-gradient(130% 130% at 50% 0%,var(--stage1),var(--stage2))}
  canvas{display:block;width:100%;height:100%;touch-action:none;cursor:crosshair}
  .glass{background:var(--glass);backdrop-filter:blur(12px);border:1px solid var(--line2)}
  .toolbar{position:absolute;top:14px;right:14px;display:flex;flex-direction:column;gap:5px;
    padding:6px;border-radius:13px;box-shadow:var(--sh)}
  .tool{display:grid;place-items:center;width:36px;height:36px;border-radius:9px;background:transparent;
    border:1px solid transparent;color:var(--tx2);transition:.12s}
  .tool:hover{background:var(--s2);color:var(--tx)}
  .tool.ai{color:var(--ai)} .tool.ai:hover{background:rgba(34,211,238,.16)}
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
    background:var(--s1);color:var(--warn);border:1px solid color-mix(in srgb,var(--warn) 40%,transparent);box-shadow:var(--sh)}
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
  /* --- data-quality + AI superpowers: Radar, Boost, Map, dup-fixer --- */
  .chip{display:none;align-items:center;gap:7px;height:30px;padding:0 12px;border-radius:999px;
    font-size:11.5px;font-weight:560;border:1px solid var(--line2);background:var(--s2);color:var(--tx2);max-width:300px}
  .chip.show{display:inline-flex}
  .chip.good{color:var(--ok);border-color:rgba(16,185,129,.4);background:rgba(16,185,129,.09)}
  .chip.run{color:var(--ai);border-color:rgba(34,211,238,.35);background:rgba(34,211,238,.09)}
  .chip.bad{color:var(--danger);border-color:rgba(239,68,68,.4);background:rgba(239,68,68,.09)}
  .chip .ic{width:14px;height:14px}
  .chip .x{margin-left:1px;opacity:.55;font-size:14px;line-height:1}
  .chip .x:hover{opacity:1}
  .spin{animation:spin 1s linear infinite;transform-origin:50% 50%}
  @keyframes spin{to{transform:rotate(360deg)}}
  .rsummary{margin-bottom:13px}
  .rrow{display:flex;align-items:center;gap:11px;padding:8px 10px;border-radius:10px;background:var(--s2);
    border:1px solid var(--line);margin-bottom:7px;width:100%;text-align:left;transition:.12s}
  .rrow:hover{border-color:var(--ac);background:var(--s3)}
  .rrow .rthumb{width:62px;height:46px;object-fit:cover;border-radius:6px;background:var(--s3);flex:none}
  .rrow .rmeta{flex:1;min-width:0}
  .rrow .rfn{font-size:12.5px;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .rrow .rb{display:inline-flex;align-items:center;gap:5px;margin-top:5px;flex-wrap:wrap}
  .rbadge{font-size:10.5px;font-weight:600;padding:2px 7px;border-radius:999px}
  .rbadge.class{background:rgba(251,191,36,.16);color:var(--warn)}
  .rbadge.miss{background:rgba(34,211,238,.16);color:var(--ai)}
  .rbadge.phantom{background:rgba(239,68,68,.16);color:var(--danger)}
  .rscore{font-size:11px;color:var(--tx3);font-variant-numeric:tabular-nums;flex:none;width:30px;text-align:right}
  .rsev{width:44px;height:5px;background:var(--s3);border-radius:9px;overflow:hidden;flex:none}
  .rsev span{display:block;height:100%;border-radius:9px}
  .ifix{display:inline-flex;align-items:center;gap:6px;height:26px;padding:0 10px;border-radius:7px;margin-left:auto;
    background:var(--ac);color:var(--ac-ink);border:0;font-size:11.5px;font-weight:600;white-space:nowrap}
  .ifix:hover{filter:brightness(1.08)} .ifix:disabled{opacity:.6}
  .ifix.done{background:transparent;color:var(--ok);border:1px solid rgba(16,185,129,.4)}
  .qrow{display:flex;align-items:center;gap:9px;padding:7px 9px;border-radius:8px;background:var(--s2);margin-bottom:6px;width:100%;text-align:left}
  .qrow:hover{background:var(--s3)}
  .qrow .qt{font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;padding:2px 6px;border-radius:5px;background:var(--s3);color:var(--tx2);flex:none}
  .qrow .qt.tiny{color:var(--danger)} .qrow .qt.sliver{color:var(--warn)} .qrow .qt.fullframe{color:var(--ai)}
  .qrow .qn{font-size:12px;color:var(--tx);flex:none;width:118px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .qrow .qm{font-size:11.5px;color:var(--tx2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .mapcard{width:min(880px,95vw);height:min(700px,92vh)}
  .mapstage{position:relative;flex:1;min-height:0;background:radial-gradient(120% 120% at 50% 0%,var(--stage1),var(--stage2));overflow:hidden}
  #mapcv{display:block;width:100%;height:100%;cursor:crosshair}
  .maphint{position:absolute;left:14px;bottom:12px;font-size:11.5px;color:var(--tx2);background:var(--glass);padding:6px 10px;border-radius:8px;border:1px solid var(--line2)}
  .maplegend{position:absolute;right:14px;top:12px;display:flex;gap:14px;font-size:11px;color:var(--tx2);background:var(--glass);padding:6px 11px;border-radius:8px;border:1px solid var(--line2)}
  .maplegend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
  /* --- project home --- */
  .home{position:fixed;inset:0;z-index:40;display:none;overflow:auto;
    background:radial-gradient(120% 90% at 50% -10%,var(--bg2),var(--bg))}
  .home.show{display:block}
  .home-theme{position:absolute;top:16px;right:18px}
  .home-inner{max-width:880px;margin:0 auto;padding:62px 24px 48px}
  .home-hero{text-align:center;margin-bottom:28px}
  .home-brand{display:inline-flex;align-items:center;gap:11px;font-size:30px;font-weight:680;letter-spacing:.2px}
  .home-brand .ic{width:34px;height:34px;color:var(--ac)}
  .home-brand b{color:var(--ac)}
  .home-tag{color:var(--tx2);font-size:14px;margin:12px auto 0;max-width:540px}
  .home-open{display:flex;align-items:center;gap:11px;background:var(--s1);border:1px solid var(--line2);
    border-radius:14px;padding:9px 9px 9px 16px;box-shadow:var(--sh);max-width:660px;margin:0 auto}
  .home-open:focus-within{border-color:var(--ac);box-shadow:0 0 0 3px rgba(8,145,178,.12),var(--sh)}
  .home-open .ic{width:19px;height:19px;color:var(--tx3);flex:none}
  .home-open input{flex:1;background:transparent;border:0;outline:none;color:var(--tx);font-size:14px}
  .home-open .btn{height:38px;padding:0 18px}
  .home-err{max-width:660px;margin:9px auto 0;color:var(--danger);font-size:12.5px;text-align:center;min-height:16px}
  .home-sec{max-width:840px;margin:32px auto 12px;color:var(--tx3);font-size:11px;text-transform:uppercase;letter-spacing:.7px}
  .home-grid{max-width:840px;margin:0 auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(244px,1fr));gap:12px}
  .prj{position:relative;text-align:left;background:var(--s1);border:1px solid var(--line2);border-radius:13px;padding:15px 15px 14px;transition:.14s;width:100%;box-shadow:var(--shs)}
  .prj:hover{border-color:var(--ac);transform:translateY(-2px);box-shadow:var(--sh)}
  .prj-name{font-size:15px;font-weight:640;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-right:18px;letter-spacing:-.1px}
  .prj-path{font-size:11px;color:var(--tx3);margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .prj-barwrap{height:7px;background:var(--s3);border-radius:9px;overflow:hidden;margin:14px 0 8px}
  .prj-bar{height:100%;border-radius:9px;background:linear-gradient(90deg,var(--ac),var(--ai))}
  .prj-meta{display:flex;justify-content:space-between;gap:6px;font-size:11.5px;color:var(--tx2);font-variant-numeric:tabular-nums}
  .prj-forget{position:absolute;top:9px;right:9px;width:22px;height:22px;border-radius:6px;display:grid;place-items:center;
    background:var(--s2);border:1px solid var(--line);color:var(--tx3);font-size:14px;line-height:1;opacity:0;transition:.12s}
  .prj:hover .prj-forget{opacity:1}
  .prj-forget:hover{color:var(--danger);border-color:rgba(239,68,68,.4)}
  .home-empty{grid-column:1/-1;text-align:center;color:var(--tx3);font-size:13px;padding:28px}
  .rdy{margin-bottom:18px;padding:14px 15px;border-radius:11px;border:1px solid var(--line2);background:var(--s2)}
  .rdy.go{border-color:rgba(16,185,129,.4);background:rgba(16,185,129,.07)}
  .rdy-h{display:flex;align-items:center;gap:9px;font-size:16px;font-weight:670;margin-bottom:12px;letter-spacing:-.2px}
  .rdy-h .ic{width:20px;height:20px} .rdy.go .rdy-h{color:var(--ok)}
  .rdy-row{display:flex;align-items:center;gap:9px;font-size:12.5px;color:var(--tx2);margin-bottom:7px}
  .rdy-row .ic{width:15px;height:15px;flex:none}
  .rdy-row.ok .ic{color:var(--ok)} .rdy-row.bad .ic{color:var(--danger)} .rdy-row.warn .ic{color:var(--warn)}
  .rdy .t-cmd{margin-top:12px}
  .sidesearch{display:flex;align-items:center;gap:8px;margin-top:8px;background:var(--s1);border:1px solid var(--line);border-radius:8px;padding:0 10px;height:30px}
  .sidesearch:focus-within{border-color:var(--ac)}
  .sidesearch .ic{width:14px;height:14px;color:var(--tx3);flex:none}
  .sidesearch input{flex:1;background:transparent;border:0;outline:none;color:var(--tx);font-size:12px}
  /* share popover */
  .pop{position:fixed;top:54px;right:14px;z-index:25;width:330px;display:none;
    background:var(--s1);border:1px solid var(--line2);border-radius:13px;box-shadow:var(--sh);padding:15px 16px}
  .pop.show{display:block}
  .pop h4{margin:0 0 5px;font-size:13.5px}
  .pop p{margin:0 0 12px;font-size:12px;color:var(--tx2);line-height:1.5}
  .pop p code{background:var(--s3);border-radius:4px;padding:0 5px;font:11px ui-monospace,monospace}
  .urlrow{display:flex;align-items:center;gap:8px;background:var(--s2);border:1px solid var(--line);border-radius:9px;padding:8px 10px}
  .urlrow code{flex:1;font:12px ui-monospace,monospace;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .urlrow button{flex:none;display:grid;place-items:center;width:28px;height:28px;border-radius:7px;background:var(--s3);border:1px solid var(--line2);color:var(--tx2)}
  .urlrow button:hover{color:var(--tx)}
  .urlrow button.copied{color:var(--ok);border-color:rgba(16,185,129,.4)}
  .home-hint{max-width:660px;margin:10px auto 0;color:var(--tx3);font-size:12.5px;text-align:center}
  .home-create{max-width:660px;margin:14px auto 0;background:var(--s1);border:1px solid var(--line2);border-radius:14px;padding:16px 18px;box-shadow:var(--sh);text-align:left}
  .hc-head{color:var(--tx2);font-size:13px;margin-bottom:12px}
  .hc-head b{color:var(--tx)}
  .hc-head code{background:var(--s3);border-radius:6px;padding:1px 6px;color:var(--tx);font:12px ui-monospace,monospace;word-break:break-all}
  .hc-lbl{display:block;color:var(--tx);font-size:13px;font-weight:600;margin-bottom:6px}
  .hc-lbl .hc-hint{color:var(--tx3);font-weight:400}
  #hcclasses{width:100%;box-sizing:border-box;resize:vertical;min-height:84px;background:var(--s2);border:1px solid var(--line2);border-radius:10px;padding:10px 12px;color:var(--tx);font:13px ui-monospace,monospace;outline:none}
  #hcclasses:focus{border-color:var(--ac);box-shadow:0 0 0 3px rgba(8,145,178,.12)}
  .hc-actions{display:flex;justify-content:flex-end;gap:10px;margin-top:12px}
  .cecard{width:min(460px,92vw)}
  .ce-note{color:var(--tx2);font-size:12.5px;line-height:1.5;margin:0 0 14px}
  .ce-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
  .ce-row .sw{width:14px;height:14px;border-radius:4px;flex:none}
  .ce-in{flex:1;background:var(--s2);border:1px solid var(--line2);border-radius:9px;padding:8px 11px;color:var(--tx);font-size:13px;outline:none}
  .ce-in:focus{border-color:var(--ac);box-shadow:0 0 0 3px rgba(8,145,178,.12)}
  .ce-rm{flex:none;width:28px;height:28px;border-radius:8px;background:var(--s2);border:1px solid var(--line2);color:var(--tx2);font-size:16px;line-height:1;cursor:pointer}
  .ce-rm:hover{color:var(--danger);border-color:var(--danger)}
  .ce-fixed{flex:none;width:34px;text-align:center;color:var(--tx3);font:11px ui-monospace,monospace}
  .ce-add{margin-top:4px}
  .ce-err{color:var(--danger);font-size:12.5px;min-height:16px;margin-top:10px}
  .ce-actions{display:flex;justify-content:flex-end;gap:10px;margin-top:12px}
  .classchip.empty{border-style:dashed;color:var(--tx2)}
  .classbar{display:none}   /* class selection now lives in the bottom label bar */
  .bottombar{position:absolute;left:0;right:0;bottom:0;display:flex;align-items:flex-end;gap:10px;padding:12px 14px;z-index:5;pointer-events:none}
  .bottombar>*{pointer-events:auto}
  .labelbar{display:flex;align-items:center;gap:7px;flex:1;min-width:0;overflow-x:auto;padding:3px;scrollbar-width:thin}
  .lchip{display:inline-flex;align-items:center;gap:7px;height:33px;padding:0 11px;border-radius:9px;flex:none;background:var(--s1);border:1px solid var(--line2);color:var(--tx2);font-size:12.5px;font-weight:540;white-space:nowrap;box-shadow:var(--sh);transition:.1s}
  .lchip:hover{border-color:var(--ac);color:var(--tx)}
  .lchip.on{border-color:var(--ac);background:var(--s3);color:var(--tx);box-shadow:0 0 0 1px var(--ac) inset,var(--sh)}
  .lchip .sw{width:12px;height:12px;border-radius:3px;flex:none}
  .lchip .ln{overflow:hidden;text-overflow:ellipsis;max-width:160px}
  .lchip .lk{font:11px ui-monospace,monospace;color:var(--tx3);background:var(--s3);border-radius:4px;padding:1px 5px;font-variant-numeric:tabular-nums}
  .lchip.add{color:var(--tx3);border-style:dashed;box-shadow:none}
  .submitbtn{display:inline-flex;align-items:center;gap:8px;height:39px;padding:0 18px;border-radius:10px;flex:none;background:var(--ac);color:#fff;font-size:13px;font-weight:650;border:1px solid var(--ac);box-shadow:var(--sh);transition:.12s}
  .submitbtn:hover{filter:brightness(1.08)}
  .submitbtn:disabled{opacity:.45;cursor:default;filter:none}
  .submitbtn .ic{width:16px;height:16px}
  .submitbtn kbd{background:rgba(255,255,255,.22);border-radius:5px;padding:1px 6px;font:11px ui-monospace,monospace}
  .tasksel{display:flex;gap:8px;flex-wrap:wrap;margin:2px 0 14px}
  .taskopt{display:inline-flex;align-items:center;gap:6px;padding:8px 13px;border-radius:9px;background:var(--s1);border:1px solid var(--line2);color:var(--tx2);font-size:12.5px;font-weight:560;transition:.1s}
  .taskopt:hover:not([disabled]){border-color:var(--ac);color:var(--tx)}
  .taskopt.on{border-color:var(--ac);background:var(--s3);color:var(--tx);box-shadow:0 0 0 1px var(--ac) inset}
  .taskopt[disabled]{opacity:.5;cursor:default}
  .taskopt .soon{font-size:9.5px;text-transform:uppercase;letter-spacing:.5px;color:var(--tx3);background:var(--s3);border-radius:4px;padding:1px 5px}
</style>
</head>
<body>
<div id="app">
  <header class="topbar">
    <span class="brand" id="brandhome" title="Projects — back to the home screen" style="cursor:pointer"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16"/><rect x="9" y="9" width="6" height="6" rx="1.5" fill="currentColor" stroke="none"/></svg>Libre<b>Label</b></span>
    <span class="sep"></span>
    <span class="ds" id="dsname"></span>
    <span class="counter" id="counter"></span>
    <button class="insbtn" id="classesbtn" title="Edit classes — rename or add" style="display:none"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.6 13.4l-7.2 7.2a2 2 0 0 1-2.8 0l-7-7A2 2 0 0 1 3 12.2V5a2 2 0 0 1 2-2h7.2a2 2 0 0 1 1.4.6l7 7a2 2 0 0 1 0 2.8z"/><circle cx="7.5" cy="7.5" r="1.5" fill="currentColor" stroke="none"/></svg></button>
    <span class="grow"></span>
    <span class="ai" id="assistbar" style="display:none">
      <button class="btn btn-primary" id="aautolabel"><svg class="ic" viewBox="0 0 24 24" fill="currentColor"><path d="M11.5 2.5l1.6 4.4 4.4 1.6-4.4 1.6-1.6 4.4-1.6-4.4-4.4-1.6 4.4-1.6z"/><path d="M18.5 14l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/></svg>Auto-label all</button>
      <button class="btn btn-ghost btn-sm" id="aprelabel" title="Auto-label this image (R)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 4V2M19 8h2M5 20l9-9M14 6l4 4"/></svg>R</button>
      <button class="btn btn-ghost btn-sm" id="asam" title="Smart-segment with SAM (S)" style="display:none"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12a8 8 0 1 1 8 8"/><circle cx="12" cy="12" r="2.5" fill="currentColor" stroke="none"/></svg>SAM</button>
      <button class="btn btn-ghost btn-sm" id="aradar" title="Label-Error Radar — audit your accepted labels (Y)" style="display:none"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19.07 4.93A10 10 0 1 0 21 12"/><path d="M12 12l6.5-4.5"/><circle cx="12" cy="12" r="3"/></svg>Radar</button>
      <input id="laprompt" class="laprompt" placeholder="objects to find — e.g. person, helmet" style="display:none">
      <span class="field" id="aconffield"><span>conf</span><input type="range" id="aconf" min="0.05" max="0.9" step="0.05"><b id="aconfval">0.25</b></span>
      <select id="amodel" class="select"></select>
    </span>
    <span class="save" id="save"></span>
    <span class="chip" id="boostchip"></span>
    <span class="sep"></span>
    <span class="tgroup">
      <button class="insbtn" id="insbtn" title="Dataset insights &amp; readiness"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20V11M10 20V4M16 20v-6M21 20H3"/></svg></button>
      <button class="insbtn" id="mapbtn" title="Embedding map — see your whole dataset (M)" style="display:none"><svg class="ic" viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="6" cy="7.5" r="1.7"/><circle cx="9.5" cy="15" r="1.7"/><circle cx="15" cy="9" r="1.7"/><circle cx="18" cy="16.5" r="1.7"/><circle cx="13" cy="18" r="1.5"/><circle cx="17.5" cy="6" r="1.4"/></svg></button>
      <button class="insbtn" id="boostbtn" title="Boost — fine-tune the model on your labels" style="display:none"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 7-7"/><path d="M14 8h6v6"/></svg></button>
    </span>
    <span class="sep"></span>
    <span class="tgroup">
      <button class="insbtn" id="homebtn" title="Projects — back to the home screen"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg></button>
      <button class="insbtn" id="sharebtn" title="Share — label together with teammates"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="2.4"/><circle cx="6" cy="12" r="2.4"/><circle cx="18" cy="19" r="2.4"/><path d="M8.1 10.9l7.8-4.8M8.1 13.1l7.8 4.8"/></svg></button>
    </span>
    <span class="sep"></span>
    <span class="tgroup">
      <button class="insbtn" id="themebtn" title="Toggle light / dark"></button>
      <button class="btn-icon" id="helpbtn" title="Shortcuts (?)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.6 9a2.4 2.4 0 1 1 3.4 2.2c-.9.4-1.1.9-1.1 1.8"/><path d="M12 17h.01"/></svg></button>
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
        <h3><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M9.6 9a2.4 2.4 0 1 1 3.4 2.2c-.9.4-1.1.9-1.1 1.8M12 17h.01"/></svg>Keyboard shortcuts</h3>
        <table>
          <tr><td><kbd>drag</kbd></td><td>draw a box (uses active class)</td></tr>
          <tr><td><kbd>click</kbd> / drag handle</td><td>select &middot; move &middot; resize</td></tr>
          <tr><td><kbd>1</kbd>..<kbd>9</kbd> <kbd>0</kbd> / <kbd>/</kbd></td><td>set class &middot; open class search</td></tr>
          <tr><td><kbd>R</kbd></td><td>AI auto-label this image</td></tr>
          <tr><td><kbd>B</kbd> / <kbd>P</kbd> / <kbd>S</kbd></td><td>box &middot; polygon &middot; smart-segment (SAM, if available)</td></tr>
          <tr><td><kbd>Enter</kbd></td><td>Submit &amp; go to the next unlabeled image (accepts AI suggestions if any are shown)</td></tr>
          <tr><td><kbd>A</kbd>/<kbd>D</kbd> &middot; <kbd>E</kbd> &middot; <kbd>C</kbd></td><td>prev / next image &middot; next unlabeled &middot; copy previous labels</td></tr>
          <tr><td><kbd>Del</kbd> / the red <b>×</b> &middot; <kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Y</kbd> &middot; <kbd>Ctrl</kbd>+<kbd>D</kbd></td><td>delete selected (or click the × on the box) &middot; undo / redo &middot; duplicate box</td></tr>
          <tr><td><kbd>Shift</kbd>+click &middot; <kbd>Ctrl</kbd>+<kbd>A</kbd></td><td>add / remove box from selection &middot; select all — then a number reclassifies, or <kbd>Del</kbd> deletes them together</td></tr>
          <tr><td><kbd>Space</kbd>+drag &middot; <kbd>wheel</kbd> / <kbd>+</kbd> <kbd>−</kbd> &middot; <kbd>F</kbd></td><td>pan &middot; zoom &middot; fit</td></tr>
          <tr><td><kbd>T</kbd> &middot; <kbd>L</kbd></td><td>tighten box to edges &middot; loupe magnifier</td></tr>
          <tr><td><kbd>Y</kbd> &middot; <kbd>N</kbd> &middot; <kbd>M</kbd></td><td>Label-Error Radar &middot; next flagged &middot; embedding map</td></tr>
          <tr><td><kbd>Ctrl</kbd>+<kbd>S</kbd> &middot; <kbd>Esc</kbd></td><td>save &middot; cancel / clear / close</td></tr>
        </table>
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
      <p class="ce-note">Rename a class, or add new ones. Existing classes keep their slot — you can't delete or reorder here (that would relabel boxes you've already drawn).</p>
      <div id="celist"></div>
      <button class="btn btn-ghost ce-add" id="ceadd">+ Add class</button>
      <div class="ce-err" id="ceerr"></div>
      <div class="ce-actions"><button class="btn btn-ghost" id="cecancel">Cancel</button><button class="btn btn-primary" id="cesave">Save</button></div>
    </div>
  </div></div>
  <div id="home" class="home">
    <button class="insbtn home-theme" id="hometheme" title="Toggle light / dark"></button>
    <div class="home-inner">
      <div class="home-hero">
        <span class="home-brand"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16"/><rect x="9" y="9" width="6" height="6" rx="1.5" fill="currentColor" stroke="none"/></svg>Libre<b>Label</b></span>
        <p class="home-tag">Label your images and train LibreYOLO on them — your data, your machine, nothing uploaded.</p>
      </div>
      <div class="home-resume" id="homeresume" style="display:none;margin:14px auto 0;text-align:center"><button class="btn btn-primary" id="homeresumebtn">← Resume current project</button></div>
      <div class="home-open">
        <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7.5A1.5 1.5 0 0 1 4.5 6h4l2 2h7A1.5 1.5 0 0 1 19 9.5v7A1.5 1.5 0 0 1 17.5 18h-13A1.5 1.5 0 0 1 3 16.5z"/></svg>
        <input id="homepath" placeholder="Paste a folder of images — or an existing data.yaml…" spellcheck="false" autocomplete="off">
        <button class="btn btn-ghost" id="homebrowse" title="Choose a folder…"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>Browse</button>
        <button class="btn btn-primary" id="homeopen">Open</button>
      </div>
      <div class="home-hint" id="homehint">A folder of images is all you need — LibreLabel writes the dataset config for you.</div>
      <div class="home-err" id="homeerr"></div>
      <div class="home-create" id="homecreate" style="display:none">
        <div class="hc-head">New project · <b id="hccount">0</b> images in <code id="hcfolder"></code></div>
        <label class="hc-lbl">Task type</label>
        <div class="tasksel" id="tasksel">
          <button class="taskopt on" data-task="detect">Detection</button>
          <button class="taskopt" data-task="segment">Segmentation</button>
          <button class="taskopt" data-task="obb" disabled>Oriented boxes <span class="soon">soon</span></button>
          <button class="taskopt" data-task="classify" disabled>Classification <span class="soon">soon</span></button>
        </div>
        <label class="hc-lbl">Class names <span class="hc-hint">— one per line. Leave it empty and add them while you label.</span></label>
        <textarea id="hcclasses" rows="4" placeholder="person&#10;helmet&#10;vest" spellcheck="false"></textarea>
        <div class="hc-actions"><button class="btn btn-ghost" id="hccancel">Cancel</button><button class="btn btn-primary" id="hccreate">Create project</button></div>
      </div>
      <div class="home-sec">Projects</div>
      <div class="home-grid" id="homegrid"></div>
    </div>
  </div>
  <div class="pop" id="sharepop"></div>
</div>
<script>
"use strict";
const $ = s => document.querySelector(s);
const cv = $("#cv"), ctx = cv.getContext("2d");
let DS = null, IMAGES = [], idx = -1;
let pendingFolder = null, ceRows = [];   // home create-flow folder + class-editor working rows
let autosaveTimer = null, saveInFlight = false;   // crash-proof periodic flush of the manual loop
let img = new Image(), imgOk = false;
let boxes = [], editable = true, dirty = false;
let active = 0, sel = -1;
let view = {scale:1, ox:0, oy:0};
let VW = 1, VH = 1;
let loadSeq = 0;
let undoStack = [], redoStack = [];
let savedSnap = "";   // snapshot of the last saved/loaded state (undo-back-to-clean baseline)
let gestureSnap = null;
let cursor = null;
let hover = -1;
let stageMsg = "";
let progSig = "";
let assist = null, assistModel = null, conf = 0.35;
let ghosts = [];
let polys = [];              // polygon annotations (image-px pts), from SAM or file
let selPoly = -1;
let selBoxes = new Set();   // multi-selection of box indices (shift-click / Ctrl+A); always cleared on image/project change so indices never go stale
let tool = "box";            // "box" (draw/select) or "seg" (SAM click/box-to-mask)
let segBusy = false;
let segRect = null;          // box being dragged in segment mode (image px)
let polyDraft = null;        // in-progress manual polygon: flat [x1,y1,x2,y2,...] image px
let curRev = null;           // the loaded label file's revision (mtime), for stale-save conflict detection
let suggestedIds = new Set();
let listFilter = "all";
let imgQuery = "";             // sidebar filename filter
let selSet = null;             // ids selected from the embedding map (sidebar filter)
let radarFindings = [];        // Radar disagreements overlaid on the current image
let radarDeck = [];            // worst-first deck from the last Radar scan
let loupeOn = false;           // magnifier toggled with L (opt-in; no auto-show while drawing)
let gradData = null, gradW = 0, gradH = 0, gradScale = 1;  // edge map for Tighten/magnetic
let mapPoints = [], mapFit = null, mapLasso = null;        // embedding scatter + lasso
let boostTimer = null;
let _cssCache = {};
const HANDLES = ["nw","n","ne","e","se","s","sw","w"];
const HR = 6;
const color = i => { const h=(i*137.508)%360; const l=62 - 16*Math.cos((h-50)*Math.PI/180); return 'hsl('+h+' 70% '+l+'%)'; };
const colorA = (i,a) => { const h=(i*137.508)%360; const l=62 - 16*Math.cos((h-50)*Math.PI/180); return 'hsl('+h+' 70% '+l+'% / '+a+')'; };
const clamp01 = v => v<0?0:v>1?1:v;
const ICO_CHECK = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 6"/></svg>';
const ICO_COPY = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';
const ICO_X = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>';
const ICO_WARN = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l9 16H3z"/><path d="M12 10v4M12 17h.01"/></svg>';
const ICO_SUN = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const ICO_MOON = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
function applyTheme(t){
  document.documentElement.classList.toggle("light", t==="light");
  _cssCache = {};   // theme changed -> re-read CSS vars used by canvas overlays
  try{ localStorage.setItem("ll-theme", t); }catch(e){}
  const b=document.querySelector("#themebtn"); if(b) b.innerHTML = (t==="light") ? ICO_MOON : ICO_SUN;
  const h=document.querySelector("#hometheme"); if(h) h.innerHTML = (t==="light") ? ICO_MOON : ICO_SUN;
  if(typeof imgOk!=="undefined" && imgOk) draw();
}
function toggleTheme(){ applyTheme(document.documentElement.classList.contains("light") ? "dark" : "light"); }
(function(){ let t; try{ t=localStorage.getItem("ll-theme"); }catch(e){}
  if(location.hash==="#light") t="light"; else if(location.hash==="#dark") t="dark";
  if(!t) t=(window.matchMedia && matchMedia("(prefers-color-scheme: light)").matches)?"light":"dark";
  document.documentElement.classList.toggle("light", t==="light"); })();
const esc = s => String(s).replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// ---- transforms ----
const sx = x => view.ox + x*view.scale;
const sy = y => view.oy + y*view.scale;
const ix = px => (px - view.ox)/view.scale;
const iy = py => (py - view.oy)/view.scale;
function rr(x,y,w,h,r){ ctx.beginPath(); if(ctx.roundRect) ctx.roundRect(x,y,w,h,r); else ctx.rect(x,y,w,h); ctx.fill(); }
// A readable label tag: dark pill + class-color dot + light text (never neon text on the photo).
function drawChip(x, y, text, ci){
  ctx.font = "600 11.5px ui-sans-serif,system-ui,sans-serif";
  const padL=18, padR=9, h=18, tw=ctx.measureText(text).width, w=padL+tw+padR;
  let cy = y - h - 3; if(cy < 1) cy = y + 2;          // flip below if it'd clip the top edge
  ctx.save();
  ctx.shadowColor="rgba(2,6,23,.5)"; ctx.shadowBlur=5; ctx.shadowOffsetY=1;
  ctx.fillStyle="rgba(13,17,28,.92)"; rr(x, cy, w, h, 5);
  ctx.shadowColor="transparent"; ctx.shadowBlur=0; ctx.shadowOffsetY=0;
  ctx.fillStyle=color(ci); ctx.beginPath(); ctx.arc(x+9, cy+h/2, 3.4, 0, 6.2832); ctx.fill();
  ctx.fillStyle="#e8edf6"; ctx.textBaseline="middle"; ctx.fillText(text, x+padL, cy+h/2+0.5);
  ctx.restore();
}

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
  try{ sessionStorage.removeItem("ll-home"); }catch(e){}
  if(active>=(DS.names||[]).length) active=0;        // stale class index from a prior project
  $("#dsname").textContent = (DS.root||"").split(/[\\/]/).filter(Boolean).pop() || "dataset";
  const cb=$("#classesbtn"); if(cb) cb.style.display = (DS.writable!==false) ? "" : "none";
  renderPalette();
  setTool(DS.task==="segment" ? "poly" : "box");   // segment projects open with the polygon tool
  const imgs = (await jget("/api/images")).images;
  if(gen!==loadSeq) return;            // another project opened while we were fetching
  IMAGES = imgs;
  hideHome();                          // only leave home once images are actually in hand
  renderList();
  renderStats();
  resizeCanvas();
  initAssist();
  if(IMAGES.length) load(0);
  else { idx=-1; imgOk=false; stageMsg = "No images found — check the dataset paths"; setSave("no images"); draw(); }
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
  if(t==="seg" && !(assist && assist.sam)) return;
  if(tool==="poly" && t!=="poly" && polyDraft) cancelPolyDraft();
  tool = t;
  const tb=$("#toolBox"), ts=$("#toolSeg"), tp=$("#toolPoly");
  if(tb) tb.classList.toggle("on", t==="box");
  if(ts) ts.classList.toggle("on", t==="seg");
  if(tp) tp.classList.toggle("on", t==="poly");
  if(t==="seg") banner("Smart segment: click an object — or drag a box around it — and SAM outlines it. B for box tool.");
  else if(t==="poly") banner("Polygon: click to add points; click the first point or press Enter to close. Esc cancels, Backspace removes the last point.");
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
  banner(`${boxes.length} box${boxes.length>1?"es":""} selected — press a number to reclassify, or Del to delete.`);
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
  if((DS && !DS.writable) || !editable){ banner("This image/dataset is read-only — nothing to submit."); return; }
  const here=idx;
  if(await save()===false) return;   // save() already surfaced why (degenerate shape, conflict, ...)
  setRowStatus(here, (boxes.length+polys.length)?"labeled":"empty");
  const vis=visibleIds();
  const hasUnlabeled = vis.some(id=> id!==here && IMAGES[id] && IMAGES[id].status==="unlabeled");
  if(hasUnlabeled) nextUnlabeled(1); else step(1);
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
  mn.classList.toggle("has-regions", n>0);
  const cnt=$("#rpcount"); if(cnt) cnt.textContent=n;
  if(!n){ list.innerHTML=`<div class="rp-empty">No labels yet — draw a box.</div>`; return; }
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
      `<span class="rpn">${esc(String(nm))} · polygon</span><span class="rpi">#${i+1}</span>`+
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
                 : `<span class="ce-fixed" title="Existing class — rename only">#${i}</span>`);
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
    const r=await fetch("/api/classes",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({names})});
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
function renderList(){
  const el = $("#list"); el.innerHTML = ""; let shown=0;
  IMAGES.forEach(im=>{
    if(!passFilter(im)) return; shown++;
    const r = document.createElement("button");
    r.className = "card" + (im.id===idx?" sel":""); r.dataset.id = im.id;
    r.innerHTML = `<img class="thumb" loading="lazy" src="/api/thumb/${im.id}?e=${(DS&&DS.epoch)||0}" alt="">`+
      `<span class="meta"><span class="fn">${esc(im.name)}</span>`+
      `<span class="st"><i class="dot ${im.status}"></i>${im.status}</span></span>`;
    r.onclick = ()=> load(im.id);
    el.appendChild(r);
  });
  if(!shown){ const lbl = listFilter==="todo"?"to-do ":listFilter==="review"?"review ":"";
    el.innerHTML = `<div class="empty">No ${lbl}images${imgQuery?" match the filter":""}</div>`; }
}
function markRow(){
  document.querySelectorAll("#list .card").forEach(r=> r.classList.toggle("sel", +r.dataset.id===idx));
  const cur = document.querySelector(`#list .card[data-id="${idx}"]`);
  if(cur) cur.scrollIntoView({block:"nearest"});
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
    if(_leak) _p.push(`<div class="iwarn leak">${_leak} group${_leak>1?"s":""} leak across splits (train↔val) — this inflates your validation score.</div>`);
    if(_dups){
      _p.push(`<div class="iwarn">${_dups} duplicate group${_dups>1?"s":""} · ${d.duplicate_image_count} images · Fix moves the extras to a reversible quarantine (keeps the train copy)</div>`);
      _p.push(d.duplicate_groups.slice(0,8).map(g=>
        `<div class="idup">${g.ids.slice(0,7).map(id=>`<img class="ithumb" loading="lazy" src="/api/thumb/${id}?e=${(DS&&DS.epoch)||0}">`).join("")}<span class="idsplit">${esc(g.splits.join(" + "))}</span><button class="ifix" data-ids='${JSON.stringify(g.ids)}'>Fix</button></div>`).join(""));
    } else {
      _p.push(`<div class="iwarn">An identical image path is listed in more than one split — remove the overlap in your data.yaml.</div>`);
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
    + `<div class="isec"><div class="ititle">Most common resolutions</div>${resBars||'<div class="iload">—</div>'}</div>`
    + `<div class="isec"><div class="ititle">Duplicates &amp; train/val leakage</div>${dup}</div>`
    + `<div class="isec"><div class="ititle">Label geometry</div>${qualitySection(q)}</div>`;
  document.querySelectorAll("#insbody .ifix[data-ids]").forEach(b=> b.onclick=()=>fixDuplicate(JSON.parse(b.dataset.ids), b));
  document.querySelectorAll("#insbody .qrow[data-id]").forEach(b=> b.onclick=()=>{ closeInsights(); load(+b.dataset.id); });
  const rc=$("#rdycmd"); if(rc) rc.onclick=()=>{ try{ navigator.clipboard.writeText('libreyolo train data='+(DS&&DS.yaml||'')); }catch(e){}
    rc.classList.add('copied'); setTimeout(()=>rc.classList.remove('copied'),1200); };
}
function readinessSection(st, d, q){
  if(!st) return "";
  const labeled=st.labeled||0, total=st.total||0;
  const leak=(d.leakage_groups||[]).length, geo=(q&&q.issues)||0, cls=st.classes||[];
  let imbalance=false;
  if(cls.length>=2){ const mx=cls[0][1], mn=cls[cls.length-1][1]; if(mn===0 || (mn>0 && mx/mn>=20)) imbalance=true; }
  const checks=[];
  checks.push(labeled>0 ? {s:"ok", t:`${labeled} of ${total} images labeled`} : {s:"bad", t:"No labeled images yet — draw some boxes first"});
  checks.push(leak ? {s:"bad", t:`${leak} duplicate group${leak>1?"s":""} leak across splits — Fix below`} : {s:"ok", t:"No train/val leakage"});
  checks.push(geo ? {s:"warn", t:`${geo} geometry issue${geo>1?"s":""} (tiny / sliver / full-frame) — see Label geometry`} : {s:"ok", t:"Box geometry is learnable"});
  if(cls.length>=2) checks.push(imbalance ? {s:"warn", t:"Class balance is skewed — add examples of the rare classes"} : {s:"ok", t:"Classes are reasonably balanced"});
  const go = labeled>0 && !leak;
  const ico = s=> s==="ok"?ICO_CHECK : s==="bad"?ICO_X : ICO_WARN;
  const rows = checks.map(c=>`<div class="rdy-row ${c.s}">${ico(c.s)}<span>${esc(c.t)}</span></div>`).join("");
  const cmd = (DS&&DS.yaml) ? `<button class="t-cmd" id="rdycmd">${ICO_COPY}<code>libreyolo train data=${esc(DS.yaml)}</code></button>` : "";
  return `<div class="rdy ${go?"go":""}"><div class="rdy-h">${go?ICO_CHECK:ICO_WARN}${go?"Ready to train":"Almost ready to train"}</div>${rows}${cmd}</div>`;
}
function qualitySection(q){
  if(!q || !q.issues) return `<div class="iok">${ICO_CHECK}No geometry problems — every box is a learnable size.</div>`;
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
  if(dirty){ banner("You have unsaved edits — save before fixing duplicates."); return; }
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

// ---- undo ----
function snap(){ return JSON.stringify({b:boxes, p:polys}); }
function applyUndo(s){ const o=JSON.parse(s); boxes=o.b||[]; polys=o.p||[]; }
function pushUndo(){ undoStack.push(snap()); if(undoStack.length>50) undoStack.shift(); redoStack = []; }
function snapStart(){ gestureSnap = snap(); }
function snapCommit(){
  if(gestureSnap!==null && gestureSnap!==snap()){
    undoStack.push(gestureSnap); if(undoStack.length>50) undoStack.shift(); redoStack = [];
  }
  gestureSnap = null;
}
// Undo/redo move one snapshot between the two stacks; both reschedule autosave so a
// restored state is just as crash-safe as a freshly drawn one.
function applyHistory(fromStack, toStack){
  if(!fromStack.length) return;
  toStack.push(snap()); applyUndo(fromStack.pop()); sel=-1; selPoly=-1;
  dirty = (snap()!==savedSnap); setSave(dirty?"unsaved":(editable?"saved":"read-only"));
  if(dirty) scheduleAutosave();
  draw();
}

// ---- load / save ----
async function load(i){
  if(i<0||i>=IMAGES.length) return;
  if(IMAGES[i] && IMAGES[i].status==="deleted") return;   // never open a quarantined image
  const myGen = ++loadSeq;
  if(dirty && idx>=0 && !(await save())){
    banner("Save failed — staying on this image so you don't lose work."); return;
  }
  if(myGen !== loadSeq) return;   // a newer load() started during the await -> bail before mutating idx
  if(dirty){   // edits landed *during* the save (snap changed mid-POST) -> still unsaved; don't discard them
    banner("You edited while saving — press → again to save those changes."); return;
  }
  idx = i; sel = -1; selPoly = -1; selBoxes.clear(); hover = -1; undoStack = []; redoStack = []; gestureSnap = null; boxes = []; polys = []; ghosts = [];
  radarFindings = []; gradData = null;
  imgOk = false; editable = false; curRev = null;   // canvas stays inert until metadata loads; a failed fetch must not leave it editable on the previous image
  let lab;
  try{ lab = await jget(`/api/label/${i}`); }
  catch(e){ if(myGen!==loadSeq) return; stageMsg = "Couldn't load this image's labels — pick another image."; draw(); return; }
  if(myGen !== loadSeq) return;
  editable = lab.editable;
  curRev = (lab.rev!=null) ? lab.rev : null;
  stageMsg = "Loading…"; draw();
  img = new Image();
  img.onload = ()=>{
    if(myGen !== loadSeq) return;
    imgOk = true;
    const anns = lab.annotations||[]; const iw=img.naturalWidth, ih=img.naturalHeight;
    boxes = anns.filter(a=>a.type==="box").map(b=>({
      cls:b.cls, x:(b.cx-b.w/2)*iw, y:(b.cy-b.h/2)*ih, w:b.w*iw, h:b.h*ih}));
    polys = anns.filter(a=>a.type==="poly").map(p=>({
      cls:p.cls, pts:p.points.map((v,k)=> k%2===0? v*iw : v*ih)}));
    dirty = false; savedSnap = snap(); setSave(editable?"saved":"read-only");
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
  clearTimeout(autosaveTimer); autosaveTimer = null;   // a real save supersedes any pending autosave
  if(!imgOk || !editable || (DS && !DS.writable)){ return true; }
  const totalShapes = boxes.length + polys.length;   // everything on the canvas
  const anns = boxes.map(pxToNorm).filter(b=>b.w>0&&b.h>0)
    .map(b=>({type:"box", cls:b.cls, cx:b.cx, cy:b.cy, w:b.w, h:b.h}));
  polys.forEach(p=>{ const pts=polyToNorm(p); if(pts.length>=6) anns.push({type:"poly", cls:p.cls, points:pts}); });
  const cur = idx, sent = snap();   // snapshot of exactly what we're sending
  try{
    const q = `epoch=${(DS&&DS.epoch)||0}` + (curRev!=null ? `&rev=${encodeURIComponent(curRev)}` : "");
    const r = await fetch(`/api/label/${cur}?${q}`,{method:"POST",
      headers:{"Content-Type":"application/json"}, body:JSON.stringify({annotations:anns})});
    if(!r.ok){ setSave("save failed"); banner((await r.json()).error||"save failed"); return false; }
    const d = await r.json().catch(()=>({}));
    if(d && d.rev!=null) curRev = d.rev;   // adopt the new revision so our own next save doesn't self-conflict
    const saved = (d && typeof d.count==="number") ? d.count : anns.length;
    // Compare against EVERY shape on the canvas, not just what we sent: a box clipped
    // to zero area (or a <3-pt polygon) is filtered out client-side and would
    // otherwise look like a clean save while silently vanishing on reload.
    const dropped = totalShapes - saved;
    if(dropped > 0){
      // The shape didn't reach disk (client-filtered zero-area, or server-sanitized
      // degenerate/collinear). Keep the edit DIRTY/failed so close & navigation warn
      // and it can't silently vanish on the next load -- the user must fix or delete it.
      dirty = true; setSave("unsaved");
      suggestedIds.delete(cur); setRowStatus(cur, saved? "labeled":"empty"); scheduleStats();
      banner(`${dropped} invalid shape${dropped>1?"s":""} dropped (degenerate) — fix or delete it to save.`);
      return false;
    }
    savedSnap = sent; dirty = (snap()!==sent);   // edits made during the POST keep it unsaved
    setSave(dirty?"unsaved":"saved");
    const el=$('#save'); el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
    suggestedIds.delete(cur);
    setRowStatus(cur, saved? "labeled":"empty");
    scheduleStats();
    return true;
  }catch(e){ setSave("save failed"); return false; }
}
function markDirty(){ dirty = true; setSave("unsaved"); scheduleAutosave(); }
// Crash-proof the manual loop: flush dirty edits to disk on a steady cadence so a
// browser/tab/server crash can't lose work drawn since the last navigation. Reuses
// the proven save() path; never fires mid-gesture or while a save is already on the wire.
function scheduleAutosave(){ if(!autosaveTimer) autosaveTimer = setTimeout(autosaveTick, 1500); }
async function autosaveTick(){
  autosaveTimer = null;
  if(mode){ scheduleAutosave(); return; }                       // a drag/resize is in progress
  if(!dirty || saveInFlight || !imgOk || !editable || (DS && !DS.writable)) return;
  saveInFlight = true;
  try{ await save(); } finally{ saveInFlight = false; }
  if(dirty) scheduleAutosave();                                 // edits made during the save -> flush again
}
function setSave(t){
  const e = $("#save"); e.textContent = t;
  const fail = (t==="save failed" || t==="image error" || t==="no images");
  e.className = "save" + (t==="unsaved"?" dirty": t==="saved"?" saved": fail?" fail":"");
}
function banner(msg){ const b=$("#banner"); b.textContent=msg; b.style.display="flex"; }
function updateProgress(){
  if(idx<0 || !IMAGES[idx]) return;
  const live = IMAGES.filter(im=>im.status!=='deleted');   // tombstones out of the count
  const total = live.length;
  const done = live.filter(im=>im.status==='labeled').length;  // only accepted labels, matches Dataset Health
  const n = boxes.length + polys.length;
  const sig = done+"|"+total+"|"+suggestedIds.size+"|"+idx+"|"+n;
  if(sig===progSig) return; progSig = sig;
  const rev = suggestedIds.size ? ` &middot; <b style="color:var(--ai)">${suggestedIds.size}</b> to review` : "";
  $("#counter").innerHTML = `<b>${done}</b>/${total} labeled${rev}`;
  const hud = $("#hud");
  if(hud) hud.innerHTML = `${idx+1} / ${total} &nbsp;&middot;&nbsp; ${n} box${n===1?'':'es'}`
    + ` &nbsp;&middot;&nbsp; <span style="color:var(--tx3)">${esc(IMAGES[idx].name)}</span>`;
}
function visibleIds(){ return IMAGES.filter(passFilter).map(im=>im.id); }  // ordered, filter-aware, excludes deleted
function step(dir){
  const vis = visibleIds(); const L=vis.length; if(!L) return;
  const p = vis.indexOf(idx);
  load(p<0 ? vis[dir>0?0:L-1] : vis[(p+dir+L)%L]);
}
function nextUnlabeled(dir){
  const vis = visibleIds(); const L=vis.length; if(!L) return;
  let p = vis.indexOf(idx); if(p<0) p = dir>0 ? -1 : L;
  for(let n=1;n<=L;n++){ const id=vis[((p+dir*n)%L+L)%L];
    if(IMAGES[id] && IMAGES[id].status==="unlabeled"){ load(id); return; } }
  banner("No more unlabeled images");
}
async function carryForward(){
  if(!imgOk || !editable || (DS && !DS.writable)) return;
  if(boxes.length + polys.length > 0){ banner("This image already has labels — carry-forward only fills an empty image."); return; }
  const vis = visibleIds(), p = vis.indexOf(idx);
  const prevId = p>0 ? vis[p-1] : -1;
  if(prevId<0){ banner("No previous image to copy from."); return; }
  const myGen = loadSeq, srcIdx = idx;
  let lab; try{ lab = await jget(`/api/label/${prevId}`); }catch(e){ banner("Couldn't read the previous image's labels."); return; }
  if(myGen!==loadSeq || idx!==srcIdx) return;   // navigated during the fetch -> don't paste onto the wrong image
  if(boxes.length + polys.length > 0){ banner("This image already has labels — carry-forward only fills an empty image."); return; }   // drawn during the fetch -> don't merge stale labels
  if(lab.editable===false){ banner("The previous image's labels are read-only (keypoints/unsupported) — can't carry them forward."); return; }   // partial view: don't drop the unparsed rows
  const anns = lab.annotations||[];
  if(!anns.length){ banner("The previous image has no labels to copy."); return; }
  const iw=img.naturalWidth, ih=img.naturalHeight; let n=0;
  pushUndo();
  anns.forEach(a=>{
    if(a.type==="box"){ const b={cls:a.cls, x:(a.cx-a.w/2)*iw, y:(a.cy-a.h/2)*ih, w:a.w*iw, h:a.h*ih};
      clipToImage(b); if(b.w>0 && b.h>0){ boxes.push(b); n++; } }
    else if(a.type==="poly"){ const q={cls:a.cls, pts:a.points.map((v,k)=> k%2===0? v*iw : v*ih)}; clipPoly(q); polys.push(q); n++; }
  });
  if(!n){ undoStack.pop(); return; }
  markDirty(); draw();
  banner(`Copied ${n} annotation${n===1?"":"s"} from the previous image — nudge them into place, then save.`);
}
function duplicateSelected(){
  if(sel<0 || !imgOk || !editable || (DS && !DS.writable)) return;
  const b=boxes[sel]; pushUndo();
  const off=Math.max(6, Math.abs(b.w)*0.06);
  boxes.push({cls:b.cls, x:b.x+off, y:b.y+off, w:b.w, h:b.h});
  selBoxes=new Set([boxes.length-1]); sel=boxes.length-1; selPoly=-1; clipToImage(boxes[sel]); markDirty(); draw();
}

// ---- AI auto-label (suggest -> review -> accept; nothing written unverified) ----
async function initAssist(){
  try{ assist = await jget("/api/assist/status"); }catch(e){ assist = null; }
  const bar = $("#assistbar"), toolAi = $("#toolAi");
  if(!assist || !assist.available){ if(bar) bar.style.display="none"; if(toolAi) toolAi.style.display="none"; return; }
  assistModel = assist.default;
  const sel = $("#amodel"); sel.innerHTML = "";
  if(assist.locate){ const o=document.createElement("option"); o.value="__locate__"; o.textContent="Locate Anything (text)"; sel.appendChild(o); }
  assist.models.forEach(m=>{ const o=document.createElement("option");
    o.value=m; o.textContent=m; if(m===assistModel) o.selected=true; sel.appendChild(o); });
  sel.onchange = ()=>{ assistModel = sel.value; updateEngineUI(); };
  if(assist.boosted) addBoostedOption(false);
  const cs = $("#aconf"); cs.value = conf; $("#aconfval").textContent = conf.toFixed(2);
  cs.oninput = ()=>{ conf = parseFloat(cs.value); $("#aconfval").textContent = conf.toFixed(2); };
  $("#aprelabel").onclick = ()=> prelabelCurrent();
  $("#aautolabel").onclick = ()=> autolabelAll();
  const asam = $("#asam");
  if(assist.sam && asam){ asam.style.display="inline-flex"; asam.onclick = ()=> setTool("seg"); }
  bar.style.display = "flex";
  if(assist.sam){ const ts=$("#toolSeg"); if(ts) ts.style.display="grid"; }
  const ar=$("#aradar"); if(ar){ ar.style.display="inline-flex"; ar.onclick=()=>runRadar(); }
  const mb=$("#mapbtn"); if(mb && assist.embed) mb.style.display="grid";
  const bb=$("#boostbtn"); if(bb && assist.boost) bb.style.display="grid";
  updateEngineUI();
}
function updateEngineUI(){
  const la = assistModel==="__locate__";
  const lp=$("#laprompt"), cf=$("#aconffield");
  if(lp) lp.style.display = la? "inline-block":"none";
  if(cf) cf.style.display = la? "none":"inline-flex";
  if(la){ banner("Locate Anything — NVIDIA non-commercial model (downloads a 3B model + runs remote code). Type the objects to find, then Auto-label."); if(lp) lp.focus(); }
  else { $("#banner").style.display="none"; }
}
function laQuery(){
  if(assistModel==="__locate__") return "engine=locate&classes="+encodeURIComponent(($("#laprompt").value||"").trim());
  return "model="+encodeURIComponent(assistModel)+"&conf="+conf;
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
  if(!editable || (DS && !DS.writable)){ banner("This image/dataset is read-only — auto-label is disabled."); return; }
  const myGen = loadSeq; setSave("running model…");
  try{
    const r = await fetch(`/api/assist/prelabel/${idx}?${laQuery()}`, {method:"POST"});
    if(myGen!==loadSeq) return;
    if(!r.ok){ const e=await r.json().catch(()=>({})); banner("Auto-label failed: "+(e.error||r.status)); restoreSave(); return; }
    const data = await r.json();
    showGhosts(data.suggestions||[]);
    if((data.suggestions||[]).length){
      suggestedIds.add(idx);
      setRowStatus(idx, "suggested");   // make manual prelabels findable via the Review filter, like bulk autolabel
    } else {
      clearReviewState();   // no ghosts -> don't leave a phantom 'suggested' row in the Review filter
    }
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
  if(segBusy || !assist || !assist.sam || idx<0 || !imgOk || !editable || (DS && !DS.writable)) return;   // view-only dataset: no canvas mutation
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
async function segmentBox(r){
  if(segBusy || !assist || !assist.sam || idx<0 || !imgOk || !editable) return;
  const iw=img.naturalWidth, ih=img.naturalHeight;
  const x1=Math.max(0,Math.min(r.x0,r.x1)), y1=Math.max(0,Math.min(r.y0,r.y1));
  const x2=Math.min(iw,Math.max(r.x0,r.x1)), y2=Math.min(ih,Math.max(r.y0,r.y1));
  if(x2-x1<4 || y2-y1<4) return;
  segBusy=true; const myGen=loadSeq; banner("Segmenting… (SAM box prompt)"); cv.style.cursor="wait";
  try{
    const rr = await fetch(`/api/assist/segment/${idx}`, {method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({box:[x1/iw, y1/ih, x2/iw, y2/ih]})});
    if(myGen!==loadSeq) return;
    if(!rr.ok){ const e=await rr.json().catch(()=>({})); banner("Segment failed: "+(e.error||rr.status)); return; }
    const d = await rr.json();
    if(!d.polygon || d.polygon.length<6){ banner("No object found in that box"); return; }
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
  if(!editable || (DS && !DS.writable)){ banner("This image/dataset is read-only — suggestions can't be accepted."); return; }
  // Unmatched suggestion (no dataset class): honour the UI's promise and apply the
  // active palette class the user selected, so open-vocab / custom-name detections
  // are acceptable instead of impossible to take without redrawing.
  const cls = (g.cls==null) ? active : g.cls;
  pushUndo();
  boxes.push({cls:cls, x:g.x, y:g.y, w:g.w, h:g.h});
  ghosts.splice(i,1); markDirty(); draw();
  if(!ghosts.length) $("#banner").style.display="none";
}
function clearReviewState(){   // a fully-dismissed image leaves the review queue (no phantom 'suggested' row)
  if(idx>=0 && suggestedIds.has(idx)){
    suggestedIds.delete(idx);
    // A present-but-empty label file (rev != "0") is a reviewed *background* -> keep it
    // 'empty' (done), don't bounce it back into To-do as 'unlabeled'.
    const fileExists = curRev!=null && String(curRev)!=="0";
    setRowStatus(idx, (boxes.length + polys.length) ? "labeled" : (fileExists ? "empty" : "unlabeled"));
  }
}
function rejectGhost(i){ if(ghosts[i]){ ghosts.splice(i,1); draw(); if(!ghosts.length){ $("#banner").style.display="none"; clearReviewState(); } } }
function acceptAllGhosts(){
  if(!editable || (DS && !DS.writable)){ banner("This image/dataset is read-only — suggestions can't be accepted."); return; }
  if(!ghosts.length) return;
  // Matched ghosts keep their mapped class; unmatched ones take the active palette
  // class (same fallback as acceptGhost), so keyboard accept-all works on
  // open-vocab / custom-name datasets where no suggestion maps by name.
  pushUndo();
  ghosts.forEach(g=> boxes.push({cls:(g.cls==null?active:g.cls), x:g.x, y:g.y, w:g.w, h:g.h}));
  ghosts = [];
  markDirty(); draw();
  $("#banner").style.display="none";
}
function clearGhosts(){ if(ghosts.length){ ghosts=[]; draw(); $("#banner").style.display="none"; clearReviewState(); } }
async function autolabelAll(){
  if(!assist || !assist.available) return;
  if(dirty && idx>=0 && !(await save())){ banner("Couldn't save the current image; fix that first."); return; }
  if(dirty){ banner("You edited while saving — press → to save before auto-labeling all."); return; }   // in-flight edits: don't queue suggestions against stale labels
  const ov=$("#progress"), bar=$("#pbar"), txt=$("#ptxt");
  ov.style.display="flex"; bar.style.width="0%"; txt.textContent="Starting… (first run loads your model)";
  IMAGES.forEach(im=>{ if(im.status==="suggested") setRowStatus(im.id, "unlabeled"); });  // clear last run's stale review rows
  suggestedIds = new Set(); let suggested=0, totalBoxes=0, classes=[], failed=null; const t0=Date.now();
  const myEpoch = DS && DS.epoch;   // project-scoped: A/D image navigation must NOT abort a dataset-wide run
  try{
    const r = await fetch(`/api/assist/autolabel?${laQuery()}`, {method:"POST"});
    if(!r.ok){ const e=await r.json().catch(()=>({})); ov.style.display="none"; banner("Auto-label unavailable: "+(e.error||r.status)); return; }   // e.g. admin-only 403 on a shared server -> don't stream a JSON error as NDJSON
    const reader=r.body.getReader(), dec=new TextDecoder(); let buf="";
    for(;;){
      const {value,done}=await reader.read(); if(done) break;
      if(!DS || DS.epoch!==myEpoch){ ov.style.display="none"; return; }   // project switched -> stop
      buf += dec.decode(value,{stream:true}); let nl;
      while((nl=buf.indexOf("\n"))>=0){
        const line=buf.slice(0,nl).trim(); buf=buf.slice(nl+1);
        if(!line) continue; let o; try{o=JSON.parse(line);}catch(e){ continue; }
        if(o.type==="progress"){
          bar.style.width = Math.round(100*o.i/Math.max(1,o.total))+"%";
          txt.textContent = `${o.i} / ${o.total} — ${o.name}` + (o.count? `  (+${o.count})`:"");
          if(o.count>0){ suggestedIds.add(o.id); setRowStatus(o.id, "suggested"); }
        } else if(o.type==="done"){ suggested=o.suggested; totalBoxes=o.boxes; classes=o.classes||[]; }
        else if(o.type==="error"){ failed = o.error || "auto-label failed"; banner("Auto-label failed: "+failed); }
      }
    }
    if(failed){   // systemic failure (e.g. missing local weights) -> show it, don't claim a clean "Done"
      $(".ptitle").textContent = "Auto-label failed";
      txt.textContent = failed;
      bar.style.width="0%";
      setTimeout(()=>{ ov.style.display="none"; $(".ptitle").textContent="Auto-labeling with your model"; }, 3500);
      return;
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
  ctx.fillStyle="rgba(2,6,23,.08)";  // subtle scrim so annotations sit above the photo
  ctx.fillRect(view.ox, view.oy, img.naturalWidth*view.scale, img.naturalHeight*view.scale);
  ctx.lineWidth = 2; ctx.font = "600 11.5px ui-sans-serif,system-ui,sans-serif"; ctx.textBaseline="bottom";
  // AI ghost suggestions (dashed/translucent, under real boxes)
  ghosts.forEach(g=>{
    const c = g.mapped ? color(g.cls) : "#9aa3b2";
    const x=sx(g.x), y=sy(g.y), w=g.w*view.scale, h=g.h*view.scale;
    const a = 0.5 + 0.45*Math.min(1, g.conf||0);   // higher confidence -> more solid
    ctx.save();
    ctx.fillStyle = g.mapped ? "rgba(34,211,238,.10)" : "rgba(154,163,178,.10)"; ctx.fillRect(x,y,w,h);
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
    const isSel = (i===sel) || selBoxes.has(i);
    const dim = (selPoly>=0) || ((sel>=0 || selBoxes.size>0) && !isSel);   // dim others when anything is selected
    const isHover = i===hover && mode===null;
    ctx.save();
    ctx.globalAlpha = dim ? 0.42 : 1;
    ctx.fillStyle = colorA(b.cls, isSel?0.22:0.13); ctx.fillRect(x,y,w,h);
    if(isSel||isHover){ ctx.shadowColor=c; ctx.shadowBlur=isSel?10:5; }
    ctx.strokeStyle = c; ctx.lineWidth = isSel?2.6:(isHover?2.2:1.8); ctx.strokeRect(x,y,w,h);
    ctx.shadowColor="transparent"; ctx.shadowBlur=0;
    const nm = (DS.names&&DS.names[b.cls])!=null ? DS.names[b.cls] : b.cls;
    drawChip(Math.min(x,x+w), Math.min(y,y+h), String(nm), b.cls);
    ctx.restore();
  });
  polys.forEach((p,i)=>{
    const c = color(p.cls), pts = p.pts;
    const selP = i===selPoly, dim = (sel>=0) || (selPoly>=0 && !selP);
    ctx.save();
    ctx.globalAlpha = dim ? 0.42 : 1;
    ctx.beginPath();
    for(let k=0;k<pts.length;k+=2){ const X=sx(pts[k]), Y=sy(pts[k+1]); if(k===0) ctx.moveTo(X,Y); else ctx.lineTo(X,Y); }
    ctx.closePath();
    ctx.fillStyle = colorA(p.cls, selP?0.22:0.14); ctx.fill();
    if(selP){ ctx.shadowColor=c; ctx.shadowBlur=10; }
    ctx.lineWidth = selP?2.6:1.8; ctx.strokeStyle=c; ctx.stroke();
    ctx.shadowColor="transparent"; ctx.shadowBlur=0;
    if(selP){ ctx.fillStyle="#fff"; ctx.strokeStyle="#06b6d4"; ctx.lineWidth=1.2;
      for(let k=0;k<pts.length;k+=2){ const X=sx(pts[k]),Y=sy(pts[k+1]); ctx.beginPath(); ctx.arc(X,Y,3,0,6.2832); ctx.fill(); ctx.stroke(); } }
    const nm = (DS.names&&DS.names[p.cls])!=null ? DS.names[p.cls] : p.cls;
    let mnx=1e9,mny=1e9; for(let k=0;k<pts.length;k+=2){ if(pts[k]<mnx)mnx=pts[k]; if(pts[k+1]<mny)mny=pts[k+1]; }
    drawChip(sx(mnx), sy(mny), String(nm), p.cls);
    ctx.restore();
  });
  if(polyDraft && polyDraft.length){
    ctx.save();
    ctx.strokeStyle=color(active); ctx.lineWidth=2; ctx.setLineDash([5,4]);
    ctx.beginPath(); ctx.moveTo(sx(polyDraft[0]), sy(polyDraft[1]));
    for(let k=2;k<polyDraft.length;k+=2) ctx.lineTo(sx(polyDraft[k]), sy(polyDraft[k+1]));
    if(cursor) ctx.lineTo(cursor.x, cursor.y);
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle="#fff"; ctx.strokeStyle=color(active); ctx.lineWidth=1.4;
    for(let k=0;k<polyDraft.length;k+=2){ ctx.beginPath(); ctx.arc(sx(polyDraft[k]),sy(polyDraft[k+1]),4,0,6.2832); ctx.fill(); ctx.stroke(); }
    if(polyDraft.length>=6 && cursor && Math.hypot(sx(polyDraft[0])-cursor.x, sy(polyDraft[1])-cursor.y)<12){
      ctx.beginPath(); ctx.arc(sx(polyDraft[0]),sy(polyDraft[1]),7,0,6.2832); ctx.strokeStyle="#06b6d4"; ctx.lineWidth=2.5; ctx.stroke();
    }
    ctx.restore();
  }
  drawRadarFindings();
  if(sel>=0){ drawHandles(boxes[sel]); if(editable && !(DS && !DS.writable)) drawDelBadge(boxes[sel]); }
  if(cursor && (mode===null||mode==='new')){
    ctx.save(); ctx.strokeStyle='rgba(6,182,212,.4)'; ctx.lineWidth=1;
    ctx.beginPath();
    ctx.moveTo(cursor.x+0.5,0); ctx.lineTo(cursor.x+0.5,VH);
    ctx.moveTo(0,cursor.y+0.5); ctx.lineTo(VW,cursor.y+0.5);
    ctx.stroke(); ctx.restore();
  }
  if(mode==="segbox" && segRect){
    const x=sx(Math.min(segRect.x0,segRect.x1)), y=sy(Math.min(segRect.y0,segRect.y1));
    const w=Math.abs(segRect.x1-segRect.x0)*view.scale, h=Math.abs(segRect.y1-segRect.y0)*view.scale;
    ctx.save(); ctx.setLineDash([5,4]); ctx.strokeStyle="#22d3ee"; ctx.lineWidth=1.5; ctx.strokeRect(x,y,w,h);
    ctx.fillStyle="rgba(34,211,238,.10)"; ctx.fillRect(x,y,w,h); ctx.restore();
  }
  drawLoupe();
  updateProgress();
  renderRegions();
}
function handlePts(b){
  const x=b.x,y=b.y,w=b.w,h=b.h;
  return {nw:[x,y], n:[x+w/2,y], ne:[x+w,y], e:[x+w,y+h/2],
          se:[x+w,y+h], s:[x+w/2,y+h], sw:[x,y+h], w:[x,y+h/2]};
}
function drawHandles(b){
  const pts = handlePts(b);
  ctx.fillStyle = "#fff"; ctx.strokeStyle = "#06b6d4"; ctx.lineWidth=1.5;
  HANDLES.forEach(k=>{
    const [hx,hy]=pts[k]; const px=sx(hx), py=sy(hy);
    ctx.beginPath(); ctx.rect(px-4,py-4,8,8); ctx.fill(); ctx.stroke();
  });
}
// A red "x" badge just outside the top-right corner of the selected box -> click to delete.
function delBadgePos(b){ return {x: sx(Math.max(b.x,b.x+b.w))+11, y: sy(Math.min(b.y,b.y+b.h))-11}; }
function drawDelBadge(b){
  const p=delBadgePos(b);
  ctx.save();
  ctx.beginPath(); ctx.arc(p.x,p.y,9,0,6.2832);
  ctx.shadowColor="rgba(2,6,23,.5)"; ctx.shadowBlur=5; ctx.fillStyle="#ef4444"; ctx.fill();
  ctx.shadowColor="transparent"; ctx.strokeStyle="#fff"; ctx.lineWidth=1.8;
  ctx.beginPath();
  ctx.moveTo(p.x-3.4,p.y-3.4); ctx.lineTo(p.x+3.4,p.y+3.4);
  ctx.moveTo(p.x+3.4,p.y-3.4); ctx.lineTo(p.x-3.4,p.y+3.4); ctx.stroke();
  ctx.restore();
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
// ---- manual polygon creation (no SAM needed): click vertices, close to commit ----
function commitPolyDraft(){
  if(polyDraft && polyDraft.length>=6){   // >=3 points
    pushUndo(); polys.push({cls:active, pts:polyDraft.slice()});
    selPoly=polys.length-1; sel=-1; selBoxes.clear(); markDirty();
  }
  polyDraft=null; draw();
}
function cancelPolyDraft(){ polyDraft=null; draw(); }

// ---- mouse ----
let mode=null, drag=null, spaceDown=false;
cv.addEventListener("pointerdown", e=>{
  cv.setPointerCapture(e.pointerId);
  const mx=e.offsetX, my=e.offsetY;
  if(spaceDown || e.button===1){ mode="pan"; drag={mx,my,ox:view.ox,oy:view.oy}; return; }
  if(!imgOk) return;
  if(!editable || (DS && !DS.writable)){   // view-only: select to inspect, never mutate
    const hb=hitBox(mx,my);
    if(hb>=0){ sel=hb; selPoly=-1; active=boxes[hb].cls; markPalette(); draw(); return; }
    const hp=hitPoly(mx,my);
    if(hp>=0){ selPoly=hp; sel=-1; active=polys[hp].cls; markPalette(); draw(); return; }
    sel=-1; selPoly=-1; draw(); return;
  }
  if(tool==="poly"){   // manual polygon: each click drops a vertex; click the first to close
    if(!(DS.names||[]).length){ banner("Add a class before drawing — every annotation needs one."); openClassEdit(); return; }
    if(polyDraft && polyDraft.length>=6 && Math.hypot(sx(polyDraft[0])-mx, sy(polyDraft[1])-my)<12){ commitPolyDraft(); return; }
    if(!polyDraft) polyDraft=[];
    polyDraft.push(ix(mx), iy(my)); draw(); return;
  }
  if(sel>=0 && boxes[sel]){   // click the red x badge on the selected box to delete it
    const bp=delBadgePos(boxes[sel]);
    if(Math.hypot(mx-bp.x, my-bp.y)<=11){ pushUndo(); boxes.splice(sel,1); sel=-1; selBoxes.clear(); markDirty(); draw(); return; }
  }
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
    if(e.shiftKey){   // toggle this box in/out of the multi-selection (no move)
      selBoxes.has(hit) ? selBoxes.delete(hit) : selBoxes.add(hit);
      sel = selBoxes.size===1 ? [...selBoxes][0] : -1; selPoly=-1;
      if(selBoxes.size) banner(`${selBoxes.size} box${selBoxes.size>1?"es":""} selected`);
      draw(); return;
    }
    snapStart(); selBoxes = new Set([hit]); sel=hit; selPoly=-1; mode="move";
    drag={mx, my, x:boxes[hit].x, y:boxes[hit].y};
    setActive(boxes[hit].cls); draw(); return;
  }
  const ph = hitPoly(mx,my);
  if(ph>=0){
    snapStart(); selBoxes.clear(); selPoly=ph; sel=-1; mode="movepoly";
    drag={mx, my, pts:polys[ph].pts.slice()};
    setActive(polys[ph].cls); draw(); return;
  }
  if(ghosts.length){
    const g = hitGhost(mx,my);
    if(g>=0){ if(e.altKey) rejectGhost(g); else acceptGhost(g); mode=null; drag=null; return; }
  }
  if(!editable || (DS && !DS.writable)) return;
  if(!(DS.names||[]).length){ banner("Add a class before drawing — every box needs one."); openClassEdit(); return; }
  if(tool==="seg"){ mode="segbox"; drag={mx, my, x0:ix(mx), y0:iy(my)}; segRect=null; return; }
  const x=ix(mx), y=iy(my);
  snapStart();
  boxes.push({cls:active, x, y, w:0, h:0});
  selBoxes.clear(); sel=boxes.length-1; selPoly=-1; mode="new"; drag={};
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
  if(mode==="segbox"){ segRect={x0:drag.x0, y0:drag.y0, x1:ix(mx), y1:iy(my)}; draw(); return; }
  let hb=-1;
  if(imgOk && sel>=0 && hitHandle(boxes[sel],mx,my)){ cv.style.cursor="pointer"; }
  else { hb = imgOk?hitBox(mx,my):-1; cv.style.cursor = spaceDown?"grab":(hb>=0?"move":"crosshair"); }
  hover = hb;
  draw();
});
cv.addEventListener("pointerleave", ()=>{ cursor=null; hover=-1; draw(); });
cv.addEventListener("pointerup", e=>{
  if(mode==="segbox"){
    const dmx=drag.mx, dmy=drag.my, r=segRect; segRect=null; mode=null; drag=null; draw();
    if((Math.abs(e.offsetX-dmx)>5 || Math.abs(e.offsetY-dmy)>5) && r) segmentBox(r); else segmentAt(dmx, dmy);
    return;
  }
  if(mode==="new"){
    const b=boxes[sel];
    if(Math.abs(b.w)*view.scale<6 || Math.abs(b.h)*view.scale<6){ boxes.pop(); sel=-1; }   // ignore tiny accidental drags (a click shouldn't spawn a box)
    else { normalizeRect(b); clipToImage(b); markDirty(); }  // WYSIWYG: keep the box exactly where drawn; press T to magnet-tighten on demand
  } else if(mode==="resize"){ normalizeRect(drag.b); clipToImage(drag.b); }
  else if(mode==="move"){ clipToImage(boxes[sel]); }
  else if(mode==="movepoly"){ clipPoly(polys[selPoly]); }
  else if(mode==="vertex"){ clipPoly(polys[selPoly]); }
  snapCommit();
  mode=null; drag=null; draw();
});
cv.addEventListener("dblclick", e=>{
  if(selPoly<0 || !imgOk) return;
  if(!editable || (DS && !DS.writable)) return;   // view-only: select to inspect, never insert a vertex
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
  if(t==="INPUT"||t==="SELECT"||t==="TEXTAREA"){ if(e.key==="Escape"){ e.target.blur(); closePicker(); if($("#classedit").classList.contains("show")) closeClassEdit(); } return; }
  if(e.key===" "){ spaceDown=true; cv.style.cursor="grab"; e.preventDefault(); return; }
  if((e.ctrlKey||e.metaKey) && (e.key==="s"||e.key==="S")){ e.preventDefault(); save(); return; }
  if((e.ctrlKey||e.metaKey) && !e.shiftKey && (e.key==="z"||e.key==="Z")){ e.preventDefault(); applyHistory(undoStack, redoStack); return; }
  if((e.ctrlKey||e.metaKey) && ((e.key==="y"||e.key==="Y") || (e.shiftKey && (e.key==="z"||e.key==="Z")))){ e.preventDefault(); applyHistory(redoStack, undoStack); return; }
  if((e.ctrlKey||e.metaKey) && (e.key==="d"||e.key==="D")){ e.preventDefault(); duplicateSelected(); return; }
  if((e.ctrlKey||e.metaKey) && (e.key==="a"||e.key==="A")){ e.preventDefault(); selectAllBoxes(); return; }
  if(e.key==="Enter"){
    e.preventDefault();
    if(polyDraft){ commitPolyDraft(); return; }   // close the in-progress polygon
    if(ghosts.length){ const adv=e.shiftKey; acceptAllGhosts();
      if(adv){ save().then(()=> nextSuggested()); } return; }
    submitImage();   // Enter / Ctrl+Enter: save this image and advance to the next unlabeled one
    return;
  }
  if(e.key>="0" && e.key<="9"){ const i = e.key==="0"?9:(+e.key-1); setActive(i); return; }
  if(e.key==="/"){ e.preventDefault(); togglePicker(); return; }
  if(e.key==="d"||e.key==="D"||e.key==="ArrowRight"){ e.preventDefault(); step(1); return; }
  if(e.key==="a"||e.key==="A"||e.key==="ArrowLeft"){ e.preventDefault(); step(-1); return; }
  if(e.key==="e"||e.key==="E"){ e.preventDefault(); nextUnlabeled(e.shiftKey?-1:1); return; }
  if(e.key==="c"||e.key==="C"){ e.preventDefault(); carryForward(); return; }
  if(e.key==="r"||e.key==="R"){ e.preventDefault(); prelabelCurrent(); return; }
  if(e.key==="b"||e.key==="B"){ setTool("box"); return; }
  if(e.key==="p"||e.key==="P"){ setTool("poly"); return; }
  if((e.key==="s"||e.key==="S") && assist && assist.sam){ setTool("seg"); return; }
  if(e.key==="t"||e.key==="T"){ e.preventDefault(); tightenSelected(); return; }
  if(e.key==="l"||e.key==="L"){ loupeOn=!loupeOn; draw(); return; }
  if(e.key==="y"||e.key==="Y"){ e.preventDefault(); runRadar(); return; }
  if(e.key==="m"||e.key==="M"){ e.preventDefault(); openMap(); return; }
  if(e.key==="n"||e.key==="N"){ e.preventDefault(); nextFlagged(); return; }
  if(e.key==="Delete"||e.key==="Backspace"){
    if(!editable || (DS && !DS.writable)) return;
    if(polyDraft){ if(polyDraft.length>=2) polyDraft.splice(-2,2); draw(); return; }   // undo last vertex
    if(selBoxes.size){   // bulk delete: splice high indices first so the rest stay valid
      pushUndo(); [...selBoxes].sort((a,b)=>b-a).forEach(bi=>boxes.splice(bi,1));
      selBoxes.clear(); sel=-1; markDirty(); draw(); }
    else if(selPoly>=0){ pushUndo(); polys.splice(selPoly,1); selPoly=-1; markDirty(); draw(); }
    else if(sel>=0){ pushUndo(); boxes.splice(sel,1); sel=-1; markDirty(); draw(); }
    return; }
  if(e.key==="+"||e.key==="="){ e.preventDefault(); zoomBy(1.25); return; }
  if(e.key==="-"||e.key==="_"){ e.preventDefault(); zoomBy(1/1.25); return; }
  if(e.key==="f"||e.key==="F"){ fit(); draw(); return; }
  if(e.key==="?"){ toggleHelp(); return; }
  if(e.key==="Escape"){
    if(polyDraft){ cancelPolyDraft(); return; }
    if($("#sharepop").classList.contains("show")){ $("#sharepop").classList.remove("show"); }
    else if($("#radar").classList.contains("show")){ closeRadar(); }
    else if($("#mapmodal").classList.contains("show")){ closeMap(); }
    else if($("#insights").classList.contains("show")){ closeInsights(); }
    else if($("#classedit").classList.contains("show")){ closeClassEdit(); }
    else if($("#picker").classList.contains("show")){ closePicker(); }
    else if(radarFindings.length){ radarFindings=[]; $("#banner").style.display="none"; draw(); }
    else if(ghosts.length){ clearGhosts(); }
    else if($("#help").style.display==="flex"){ $("#help").style.display="none"; }
    else if(mode==="new"){ boxes.pop(); sel=-1; mode=null; gestureSnap=null; draw(); }
    else { selBoxes.clear(); sel=-1; selPoly=-1; draw(); }
  }
});
window.addEventListener("keyup", e=>{ if(e.key===" "){ spaceDown=false; cv.style.cursor="crosshair"; } });
window.addEventListener("resize", ()=>{ resizeCanvas(); if($("#mapmodal").classList.contains("show")) resizeMapCanvas(); });
window.addEventListener("beforeunload", e=>{ if(dirty){ e.preventDefault(); e.returnValue=""; } });

// ===== Data-quality + AI superpowers: Radar · Tighten · Loupe · Boost · Map =====
const ICO_SPIN = '<svg class="ic spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M12 3a9 9 0 1 0 9 9"/></svg>';
function getCss(v){ if(_cssCache[v]!=null) return _cssCache[v]; const c=(getComputedStyle(document.documentElement).getPropertyValue(v)||"").trim()||"#06b6d4"; _cssCache[v]=c; return c; }
function dsname(c){ return (DS && DS.names && DS.names[c]!=null) ? DS.names[c] : c; }

// ---- Label-Error Radar: audit accepted labels with the model ----
function radarQuery(){ const m=(assistModel==="__locate__")?(assist.default||""):assistModel; return "model="+encodeURIComponent(m||"")+"&conf="+conf; }
function openRadar(){ $("#radar").classList.add("show"); }
function closeRadar(){ $("#radar").classList.remove("show"); }
async function runRadar(){
  if(!assist || !assist.available) return;
  if(dirty && idx>=0 && !(await save())){ banner("Save the current image first, then run Radar."); return; }
  if(dirty){ banner("You edited while saving — press → to save before running Radar."); return; }   // in-flight edits: don't audit stale on-disk labels
  const myEpoch = DS && DS.epoch;   // project-scoped: image navigation must not abort the dataset-wide scan
  openRadar();
  $("#radarbody").innerHTML = `<div class="iload">Auditing your accepted labels with your model…<div class="ptrack" style="width:240px;margin:14px auto 0"><div class="pbar" id="rbar"></div></div></div>`;
  radarDeck = [];
  try{
    const r = await fetch(`/api/assist/radar?${radarQuery()}`, {method:"POST"});
    if(!r.ok){ const e=await r.json().catch(()=>({})); $("#radarbody").innerHTML=`<div class="iload">Radar unavailable: ${esc(e.error||r.status)}</div>`; return; }
    const reader=r.body.getReader(), dec=new TextDecoder(); let buf="";
    for(;;){ const {value,done}=await reader.read(); if(done) break;
      buf += dec.decode(value,{stream:true}); let nl;
      while((nl=buf.indexOf("\n"))>=0){ const line=buf.slice(0,nl).trim(); buf=buf.slice(nl+1);
        if(!line) continue; let o; try{o=JSON.parse(line);}catch(e){ continue; }
        if(o.type==="progress"){ const bar=$("#rbar"); if(bar) bar.style.width=Math.round(100*o.i/Math.max(1,o.total))+"%"; }
        else if(o.type==="done"){ if(!DS || DS.epoch!==myEpoch) return; radarDeck=o.deck||[]; renderRadarDeck(o); }
        else if(o.type==="error"){ $("#radarbody").innerHTML=`<div class="iload">Radar failed: ${esc(o.error)}</div>`; }
      }
    }
  }catch(e){ $("#radarbody").innerHTML=`<div class="iload">Radar failed.</div>`; }
}
function renderRadarDeck(o){
  const deck=o.deck||[];
  if(!o.scanned){ $("#radarbody").innerHTML=`<div class="iok">${ICO_CHECK}No labelled images to audit yet — accept some boxes, then run Radar to check them against your model.</div>`; return; }
  if(!deck.length){ $("#radarbody").innerHTML=`<div class="iok">${ICO_CHECK}No disagreements — your model agrees with your labels across ${o.scanned} labelled image${o.scanned===1?"":"s"}.</div>`; return; }
  const totalIssues=deck.reduce((a,d)=>a+d.issues,0);
  const order={class:0,miss:1,phantom:2};
  const sum=`<div class="rsummary"><div class="iwarn">${deck.length} image${deck.length>1?"s":""} · ${totalIssues} likely issue${totalIssues>1?"s":""} — worst first. Click one to review; nothing changes until you edit by hand.</div></div>`;
  const maxS = deck[0] ? (deck[0].score||1) : 1;
  const rows=deck.slice(0,80).map(d=>{
    const badges=Object.entries(d.counts||{}).sort((a,b)=>(order[a[0]]||9)-(order[b[0]]||9))
      .map(([t,c])=>`<span class="rbadge ${t}">${c} ${t==="class"?"class slip":t}</span>`).join("");
    const ratio=Math.max(0.1,(d.score||0)/maxS);
    const sevCol=ratio>0.66?"var(--danger)":ratio>0.33?"var(--warn)":"var(--ai)";
    return `<button class="rrow" data-id="${d.id}"><img class="rthumb" loading="lazy" src="/api/thumb/${d.id}?e=${(DS&&DS.epoch)||0}"><span class="rmeta"><span class="rfn">${esc(d.name)}</span><span class="rb">${badges}</span></span><span class="rsev"><span style="width:${Math.round(ratio*100)}%;background:${sevCol}"></span></span><span class="rscore">${d.score}</span></button>`;
  }).join("");
  $("#radarbody").innerHTML=sum+rows;
  document.querySelectorAll("#radarbody .rrow").forEach(b=> b.onclick=()=>{ closeRadar(); reviewFinding(+b.dataset.id); });
}
async function reviewFinding(id){ await load(id); if(idx===id) await loadRadarFindings(id); }   // only overlay if load() actually landed on this image
async function loadRadarFindings(id){
  const myGen=loadSeq;
  try{ const d=await jget(`/api/assist/radar/${id}`);
    if(myGen!==loadSeq) return;            // navigated away mid-fetch -> drop stale findings
    radarFindings=d.findings||[]; draw();
    if(radarFindings.length) banner(`Radar flagged ${radarFindings.length} here — fix by hand, then save. N: next flagged · Esc: clear.`);
  }catch(e){ if(myGen===loadSeq) radarFindings=[]; }
}
function nextFlagged(){ if(!radarDeck.length) return; const ids=radarDeck.map(d=>d.id); const nxt=ids.find(j=>j>idx); reviewFinding(nxt!=null?nxt:ids[0]); }
function drawRadarFindings(){
  if(!radarFindings.length || !imgOk) return;
  const iw=img.naturalWidth, ih=img.naturalHeight;
  ctx.save(); ctx.lineWidth=2.5; ctx.font="600 11.5px ui-sans-serif,system-ui,sans-serif"; ctx.textBaseline="bottom";
  radarFindings.forEach(f=>{
    const col = f.type==="class"? getCss("--warn") : f.type==="miss"? getCss("--ai") : getCss("--danger");
    const b=f.box, x=sx((b[0]-b[2]/2)*iw), y=sy((b[1]-b[3]/2)*ih), w=b[2]*iw*view.scale, h=b[3]*ih*view.scale;
    ctx.save(); ctx.setLineDash([7,4]); ctx.strokeStyle=col; ctx.shadowColor=col; ctx.shadowBlur=8; ctx.strokeRect(x,y,w,h); ctx.restore();
    const lab = f.type==="class"? `you: ${esc(String(dsname(f.label_cls)))} · model: ${esc(String(f.pred_name||dsname(f.pred_cls)))}`
              : f.type==="miss"? `missed? ${esc(String(f.pred_name||dsname(f.pred_cls)))} ${Math.round((f.conf||0)*100)}%`
              : `no object here? (${esc(String(dsname(f.label_cls)))})`;
    const tw=ctx.measureText(lab).width+12;
    ctx.save(); ctx.shadowColor="rgba(0,0,0,.4)"; ctx.shadowBlur=5; ctx.shadowOffsetY=1; ctx.globalAlpha=.95; ctx.fillStyle=col; rr(x,y-17,tw,16,4); ctx.restore();
    ctx.fillStyle="#0a0b0e"; ctx.fillText(lab,x+6,y-3);
  });
  ctx.restore();
}

// ---- Magnetic edges + one-key Tighten (T): snap box edges to image gradients ----
function ensureGrad(){
  if(gradData || !imgOk) return gradData;
  try{
    const iw=img.naturalWidth, ih=img.naturalHeight, cap=1280;
    const s=Math.min(1, cap/Math.max(iw,ih));
    const w=Math.max(1,Math.round(iw*s)), h=Math.max(1,Math.round(ih*s));
    const oc=document.createElement("canvas"); oc.width=w; oc.height=h;
    const octx=oc.getContext("2d",{willReadFrequently:true});
    octx.drawImage(img,0,0,w,h);
    const d=octx.getImageData(0,0,w,h).data, lum=new Float32Array(w*h);
    for(let i=0,p=0;i<w*h;i++,p+=4) lum[i]=0.299*d[p]+0.587*d[p+1]+0.114*d[p+2];
    const g=new Float32Array(w*h);
    for(let y=1;y<h-1;y++) for(let x=1;x<w-1;x++){ const i=y*w+x; g[i]=Math.abs(lum[i+1]-lum[i-1])+Math.abs(lum[i+w]-lum[i-w]); }
    gradData=g; gradW=w; gradH=h; gradScale=s;
  }catch(e){ gradData=null; }   // cross-origin / oversize -> graceful no-op
  return gradData;
}
function snapBox(b, m){
  const g=ensureGrad(); if(!g) return false;
  const s=gradScale, W=gradW, H=gradH;
  normalizeRect(b); clipToImage(b);
  let x1=Math.round(b.x*s), y1=Math.round(b.y*s), x2=Math.round((b.x+b.w)*s), y2=Math.round((b.y+b.h)*s);
  x1=Math.max(1,Math.min(W-2,x1)); x2=Math.max(1,Math.min(W-2,x2));
  y1=Math.max(1,Math.min(H-2,y1)); y2=Math.max(1,Math.min(H-2,y2));
  if(x2-x1<3 || y2-y1<3) return false;
  m=Math.max(2, Math.min(m, Math.floor(Math.min(x2-x1,y2-y1)/2)));
  const colSum=(x,a,bb)=>{ let v=0; for(let y=a;y<=bb;y++) v+=g[y*W+x]; return v; };
  const rowSum=(y,a,bb)=>{ let v=0; for(let x=a;x<=bb;x++) v+=g[y*W+x]; return v; };
  const bestCol=(cx,a,bb)=>{ let bx=cx,bv=-1; for(let x=Math.max(1,cx-m);x<=Math.min(W-2,cx+m);x++){ const v=colSum(x,a,bb); if(v>bv){bv=v;bx=x;} } return bx; };
  const bestRow=(cy,a,bb)=>{ let by=cy,bv=-1; for(let y=Math.max(1,cy-m);y<=Math.min(H-2,cy+m);y++){ const v=rowSum(y,a,bb); if(v>bv){bv=v;by=y;} } return by; };
  const nx1=bestCol(x1,y1,y2), nx2=bestCol(x2,y1,y2), ny1=bestRow(y1,x1,x2), ny2=bestRow(y2,x1,x2);
  const X1=Math.min(nx1,nx2)/s, X2=Math.max(nx1,nx2)/s, Y1=Math.min(ny1,ny2)/s, Y2=Math.max(ny1,ny2)/s;
  if(X2-X1<2 || Y2-Y1<2) return false;
  b.x=X1; b.y=Y1; b.w=X2-X1; b.h=Y2-Y1; return true;
}
function tightenSelected(){
  if(sel<0 || !imgOk) return;
  if(!editable || (DS && !DS.writable)) return;   // view-only: never reshape a protected box
  if(!ensureGrad()){ banner("Tighten needs the image pixels (unavailable for this image)."); return; }
  const b=boxes[sel], m=Math.max(10, Math.round(Math.min(b.w,b.h)*gradScale*0.22));
  gestureSnap=snap();
  if(snapBox(b,m)){ snapCommit(); markDirty(); draw(); banner("Tightened to the nearest edges (T)"); }
  else gestureSnap=null;
}

// ---- Loupe magnifier (L pins it; auto-shows while drawing/resizing) ----
function drawLoupe(){
  if(!imgOk || !cursor) return;
  if(!loupeOn) return;   // opt-in only (press L); no auto-zoom-on-background while drawing
  const R=64, zoom=3.4, pad=22;
  let lx=cursor.x+pad+R, ly=cursor.y+pad+R;
  if(lx+R>VW) lx=cursor.x-pad-R;
  if(ly+R>VH) ly=cursor.y-pad-R;
  if(lx-R<0) lx=cursor.x+pad+R;
  if(ly-R<0) ly=cursor.y+pad+R;
  lx=Math.max(R,Math.min(VW-R,lx)); ly=Math.max(R,Math.min(VH-R,ly));   // never clip off the stage
  const bx=ix(cursor.x), by=iy(cursor.y), eff=view.scale*zoom;
  ctx.save();
  ctx.beginPath(); ctx.arc(lx,ly,R,0,6.2832); ctx.closePath();
  ctx.fillStyle=getCss("--bg2"); ctx.fill();
  ctx.save(); ctx.clip();
  ctx.imageSmoothingEnabled=false;
  ctx.drawImage(img, lx-bx*eff, ly-by*eff, img.naturalWidth*eff, img.naturalHeight*eff);
  ctx.imageSmoothingEnabled=true;
  ctx.strokeStyle="rgba(6,182,212,.85)"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(lx-R,ly); ctx.lineTo(lx+R,ly); ctx.moveTo(lx,ly-R); ctx.lineTo(lx,ly+R); ctx.stroke();
  ctx.restore();
  ctx.lineWidth=2.5; ctx.strokeStyle=getCss("--ac"); ctx.shadowColor="rgba(2,6,23,.5)"; ctx.shadowBlur=10;
  ctx.beginPath(); ctx.arc(lx,ly,R,0,6.2832); ctx.stroke();
  ctx.restore();
}

// ---- Boost: train-in-the-loop (background fine-tune + agreement delta) ----
async function runBoost(){
  if(!assist || !assist.boost) return;
  if(dirty && idx>=0 && !(await save())){
    banner("Save the current image first — Boost trains on your accepted labels."); return;
  }
  if(dirty){ banner("You edited while saving — press → to save before boosting."); return; }   // in-flight edits: don't train on stale labels
  setBoostChip("run", "Boosting — warming up…");
  try{
    const r=await fetch("/api/boost",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});
    const d=await r.json();
    if(!d.started){ setBoostChip("bad", d.reason||"Couldn't start Boost"); return; }
    pollBoost();
  }catch(e){ setBoostChip("bad","Boost failed to start"); }
}
function pollBoost(){
  clearTimeout(boostTimer);
  // Guard on the PROJECT generation (epoch), not loadSeq -- loadSeq bumps on every
  // image navigation, which would kill the poll the moment the user moves images
  // mid-boost. The epoch only changes on a project switch.
  const myEpoch = DS && DS.epoch;
  const live = ()=> DS && DS.epoch===myEpoch;
  boostTimer=setTimeout(async ()=>{
    if(!live()) return;        // project switched -> stop polling the old boost
    let s; try{ s=await jget("/api/boost/status"); }catch(e){ if(live()) pollBoost(); return; }
    if(!live()) return;
    if(s.state==="running"){ setBoostChip("run", s.phase||"Boosting…"); pollBoost(); }
    else if(s.state==="done"){
      const a=Math.round((s.boosted_agreement||0)*100), b=Math.round((s.base_agreement||0)*100), dl=Math.round((s.delta||0)*100);
      setBoostChip(dl<0?"bad":"good", `model agrees ${a}% (was ${b}% · ${dl>=0?"+":""}${dl}pts)`);
      if(s.usable){ addBoostedOption(true); banner("Boosted model ready & selected — press R to auto-label with it, or use Auto-label all."); }
    } else if(s.state==="error"){ setBoostChip("bad", s.error||"Boost failed"); }
  }, 1600);
}
function setBoostChip(cls, txt){
  const c=$("#boostchip"); if(!c) return;
  const ic = cls==="run"? ICO_SPIN : cls==="good"? ICO_CHECK : "";
  c.className="chip show "+(cls||"");
  c.innerHTML=`${ic}<span>${esc(txt)}</span><span class="x" id="boostx" title="dismiss">&times;</span>`;
  const x=$("#boostx"); if(x) x.onclick=()=>{ c.className="chip"; };
}
function addBoostedOption(select){
  const sel=$("#amodel"); if(!sel) return;
  if(!sel.querySelector('option[value="boosted"]')){
    const o=document.createElement("option"); o.value="boosted"; o.textContent="✦ boosted (your labels)";
    sel.insertBefore(o, sel.firstChild);
  }
  if(select){ sel.value="boosted"; assistModel="boosted"; updateEngineUI(); }
}

// ---- Embedding map: the whole dataset as a 2-D scatter you can lasso ----
function openMap(){ if(!assist || !assist.embed) return; $("#mapmodal").classList.add("show"); resizeMapCanvas(); if(mapPoints.length){ drawMap(); } else { runEmbeddings(); } }
function closeMap(){ $("#mapmodal").classList.remove("show"); mapLasso=null; }
function resizeMapCanvas(){
  const cv2=$("#mapcv"); if(!cv2) return;
  const r=cv2.getBoundingClientRect(), dpr=Math.min(window.devicePixelRatio||1,2);
  cv2.width=Math.max(1,Math.round(r.width*dpr)); cv2.height=Math.max(1,Math.round(r.height*dpr));
  cv2.getContext("2d").setTransform(dpr,0,0,dpr,0,0);
  if(mapPoints.length) drawMap();
}
async function runEmbeddings(){
  const myEpoch = DS && DS.epoch;   // project-scoped: image navigation must not abort the dataset-wide embed
  const hint=$("#maphint"); hint.textContent="Embedding your images… (first run loads a tiny model)";
  try{
    const r=await fetch("/api/embeddings",{method:"POST"});
    if(!r.ok){ const e=await r.json().catch(()=>({})); hint.textContent="Embedding unavailable: "+(e.error||r.status); return; }
    const reader=r.body.getReader(), dec=new TextDecoder(); let buf="";
    for(;;){ const {value,done}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true}); let nl;
      while((nl=buf.indexOf("\n"))>=0){ const line=buf.slice(0,nl).trim(); buf=buf.slice(nl+1);
        if(!line) continue; let o; try{o=JSON.parse(line);}catch(e){ continue; }
        if(o.type==="progress"){ hint.textContent=`Embedding ${o.i} / ${o.total}…`; }
        else if(o.type==="done"){ if(!DS || DS.epoch!==myEpoch) return; mapPoints=o.points||[]; fitMap(); drawMap(); hint.textContent=mapPoints.length? "Drag a lasso around a region → jump to those images" : "No images to embed."; }
        else if(o.type==="error"){ hint.textContent="Embedding failed: "+o.error; }
      }
    }
  }catch(e){ hint.textContent="Embedding failed."; }
}
function fitMap(){
  if(!mapPoints.length){ mapFit=null; return; }
  let a=1e18,b=-1e18,c=1e18,d=-1e18;
  mapPoints.forEach(p=>{ if(p.x<a)a=p.x; if(p.x>b)b=p.x; if(p.y<c)c=p.y; if(p.y>d)d=p.y; });
  mapFit={minx:a,maxx:b,miny:c,maxy:d};
}
function mapProject(p, W, H){
  const pad=34, f=mapFit||{minx:0,maxx:1,miny:0,maxy:1};
  const dx=(f.maxx-f.minx)||1, dy=(f.maxy-f.miny)||1;
  return { x: pad + (p.x-f.minx)/dx*(W-2*pad), y: pad + (p.y-f.miny)/dy*(H-2*pad) };
}
function drawMap(){
  const cv2=$("#mapcv"); if(!cv2) return; const c2=cv2.getContext("2d");
  const r=cv2.getBoundingClientRect(); c2.clearRect(0,0,r.width,r.height);
  mapPoints.forEach(p=>{
    const im=IMAGES[p.id], st=im?im.status:"unlabeled";
    if(st==="deleted") return;   // skip points for quarantined images
    const col = st==="labeled"? "#10b981" : st==="suggested"? getCss("--ai") : "#64748b";
    const q=mapProject(p, r.width, r.height);
    c2.beginPath(); c2.arc(q.x,q.y,p.id===idx?5:3.2,0,6.2832);
    c2.fillStyle=col; c2.globalAlpha=p.id===idx?1:.85; c2.fill();
    if(p.id===idx){ c2.globalAlpha=1; c2.lineWidth=2; c2.strokeStyle=getCss("--ac"); c2.stroke(); }
  });
  c2.globalAlpha=1;
  if(mapLasso && mapLasso.length>3){
    c2.beginPath(); c2.moveTo(mapLasso[0],mapLasso[1]);
    for(let i=2;i<mapLasso.length;i+=2) c2.lineTo(mapLasso[i],mapLasso[i+1]);
    c2.closePath(); c2.fillStyle="rgba(6,182,212,.12)"; c2.fill();
    c2.lineWidth=1.5; c2.setLineDash([5,4]); c2.strokeStyle=getCss("--ac"); c2.stroke(); c2.setLineDash([]);
  }
}
function nearestMapPoint(x,y){
  const r=$("#mapcv").getBoundingClientRect(); let best=null,bd=225;
  mapPoints.forEach(p=>{ if(IMAGES[p.id] && IMAGES[p.id].status==="deleted") return;
    const q=mapProject(p,r.width,r.height); const dd=(q.x-x)*(q.x-x)+(q.y-y)*(q.y-y); if(dd<bd){bd=dd;best=p.id;} });
  return best;
}
function wireMap(){
  const cv2=$("#mapcv"); if(!cv2) return; let drawing=false;
  cv2.addEventListener("pointerdown", e=>{ if(!mapPoints.length) return; cv2.setPointerCapture(e.pointerId); drawing=true; mapLasso=[e.offsetX,e.offsetY]; });
  cv2.addEventListener("pointermove", e=>{ if(!drawing || !mapLasso) return; const L=mapLasso; if(Math.hypot(e.offsetX-L[L.length-2], e.offsetY-L[L.length-1])>3){ L.push(e.offsetX,e.offsetY); drawMap(); } });
  cv2.addEventListener("pointerup", e=>{ if(!drawing) return; drawing=false; const L=mapLasso; mapLasso=null;
    if(!L || L.length<8){ const hit=nearestMapPoint(e.offsetX,e.offsetY); drawMap(); if(hit!=null){ closeMap(); load(hit); } return; }
    const r=cv2.getBoundingClientRect();
    const ids=mapPoints.filter(p=>{ if(IMAGES[p.id] && IMAGES[p.id].status==="deleted") return false;
      const q=mapProject(p,r.width,r.height); return pointInPoly(q.x,q.y,L); }).map(p=>p.id);
    drawMap(); mapSelect(ids);
  });
}
function mapSelect(ids){
  if(!ids.length){ $("#maphint").textContent="Nothing in the lasso — try again."; return; }
  if(ids.length===1){ closeMap(); load(ids[0]); return; }
  selSet=new Set(ids); listFilter="all";
  document.querySelectorAll("#filter button").forEach(x=>x.classList.toggle("on", x.dataset.f==="all"));
  renderList(); closeMap();
  banner(`${ids.length} images selected from the map — sidebar filtered to them. Click a filter tab to clear.`);
}

// ===== Project home: list projects, open/switch, reset per-project state =====
function wireHome(){
  $("#homeopen").onclick = ()=> smartOpen($("#homepath").value.trim());
  { const hb=$("#homebrowse"); if(hb) hb.onclick = browseFolder; }
  document.querySelectorAll("#tasksel .taskopt").forEach(b=> b.onclick=()=>{
    if(b.disabled) return;
    document.querySelectorAll("#tasksel .taskopt").forEach(x=>x.classList.remove("on"));
    b.classList.add("on");
  });
  $("#homepath").addEventListener("keydown", e=>{ if(e.key==="Enter"){ e.preventDefault(); smartOpen($("#homepath").value.trim()); } });
  $("#homepath").addEventListener("input", ()=>{ homeError(""); hideCreate(); });
  $("#hometheme").onclick = toggleTheme;
  $("#hccancel").onclick = hideCreate;
  $("#hccreate").onclick = doCreate;
}
function looksLikeYaml(s){ return /\.ya?ml$/i.test(s); }
// One smart input: a data.yaml opens directly; any other path is inspected so we
// can open an existing dataset, or offer to *create* one from a bare image folder.
// Native "choose folder" dialog (pops on the host via the server's tkinter) so you
// don't have to paste a path. Falls back to the text input if no GUI is available.
async function browseFolder(){
  const btn=$("#homebrowse"); if(btn) btn.disabled=true;
  try{
    const r=await fetch("/api/pick-folder",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});
    const d=await r.json().catch(()=>({}));
    if(!r.ok){ homeError(d.error||"Couldn't open the folder dialog — paste the path instead."); return; }
    if(d.unavailable){ homeError("No folder dialog available on this machine — paste the path instead."); return; }
    if(d.folder){ const hp=$("#homepath"); if(hp) hp.value=d.folder; smartOpen(d.folder); }
  }catch(e){ homeError("Couldn't open the folder dialog — paste the path instead."); }
  finally{ if(btn) btn.disabled=false; }
}
async function smartOpen(p){
  if(!p){ homeError("Paste a folder of images, or a data.yaml."); return; }
  if(dirty && idx>=0 && !(await save())){ homeError("Save the current image before switching projects."); return; }
  if(dirty){ homeError("You edited while saving — save before switching projects."); return; }
  homeError(""); hideCreate();
  if(looksLikeYaml(p)) return openProject(p);
  const btn=$("#homeopen"), t=btn.textContent; btn.disabled=true; btn.textContent="Checking…";
  let info;
  try{
    const r=await fetch("/api/projects/inspect",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({folder:p})});
    info=await r.json(); if(!r.ok) throw new Error(info.error||"inspect failed");
  }catch(e){ homeError("Couldn't read that path."); return; }
  finally{ btn.disabled=false; btn.textContent=t; }
  if(info.has_yaml) return openProject(info.yaml||p);
  if(!info.is_dir){ homeError("That path isn't a folder or a data.yaml."); return; }
  if(info.images>0){ showCreate(p, info.images); return; }
  homeError("No images found in that folder.");
}
function showCreate(folder, n){
  pendingFolder = folder;
  $("#hcfolder").textContent = folder;
  $("#hccount").textContent = n;
  $("#hcclasses").value = "";
  document.querySelectorAll("#tasksel .taskopt").forEach(x=>x.classList.toggle("on", x.dataset.task==="detect"));
  $("#homecreate").style.display = "";
  $("#hcclasses").focus();
}
function hideCreate(){ const c=$("#homecreate"); if(c) c.style.display="none"; pendingFolder=null; }
async function doCreate(){
  const folder = pendingFolder; if(!folder) return;
  const seen=new Set(), classes=[];
  for(const raw of $("#hcclasses").value.split(/\r?\n/)){
    const c=raw.trim(); if(!c) continue;
    const k=c.toLowerCase(); if(seen.has(k)){ homeError("Class names must be unique."); return; }
    seen.add(k); classes.push(c);
  }
  homeError("");
  const taskEl=document.querySelector("#tasksel .taskopt.on");
  const task=(taskEl&&taskEl.dataset.task)||"detect";
  const btn=$("#hccreate"), t=btn.textContent; btn.disabled=true; btn.textContent="Creating…";
  try{
    const r=await fetch("/api/projects/create",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({folder, classes, task})});
    const d=await r.json();
    if(!r.ok || !d.open){ homeError(d.error||"Could not create that project."); return; }
    hideCreate(); resetClientState(); await enterLabeler(d);
    const hp=$("#homepath"); if(hp) hp.value="";
  }catch(e){ homeError("Could not create that project."); }
  finally{ btn.disabled=false; btn.textContent=t; }
}
function showHome(openMeta){
  $("#home").classList.add("show");
  hideCreate(); homeError("");
  // Offer a way back into the already-open session. Crucial on a --share server,
  // where a LAN teammate can't open a project (admin-only) and would otherwise be
  // stranded on Home once ll-home is set; resume needs no admin -- just the live session.
  const rz=$("#homeresume"), btn=$("#homeresumebtn");
  if(rz && btn){
    if(openMeta && openMeta.open){
      rz.style.display="";
      btn.onclick = async ()=>{ try{ sessionStorage.removeItem("ll-home"); }catch(e){} hideHome(); await enterLabeler(openMeta); };
    } else { rz.style.display="none"; }
  }
  renderProjects();
}
function hideHome(){ $("#home").classList.remove("show"); }
async function backToHome(){
  if(dirty && idx>=0 && !(await save())){ banner("Save failed — fix it before leaving this image."); return; }
  if(dirty){ banner("You edited while saving — press → to save before going home."); return; }   // in-flight edits: don't drop them on Home
  try{ sessionStorage.setItem("ll-home","1"); }catch(e){}
  showHome(DS);   // pass the open session so "Resume current project" is offered
}
function homeError(msg){ const e=$("#homeerr"); if(e) e.textContent = msg||""; }
async function renderProjects(){
  const grid=$("#homegrid"); if(!grid) return;
  grid.innerHTML = `<div class="home-empty">Loading…</div>`;
  let d; try{ d = await jget("/api/projects"); }
  catch(e){ grid.innerHTML = `<div class="home-empty">Could not load projects.</div>`; return; }
  const list = d.projects||[];
  if(!list.length){ grid.innerHTML = `<div class="home-empty">No projects yet — paste a folder of images above and LibreLabel sets up the rest.</div>`; return; }
  grid.innerHTML = list.map(p=>{
    const n=p.count||0, l=p.labeled||0, pct = n? Math.round(100*l/n) : 0;
    return `<button class="prj" data-data="${esc(p.data)}">`
      + `<span class="prj-forget" data-forget="${esc(p.data)}" title="Forget (does not delete files)">&times;</span>`
      + `<div class="prj-name">${esc(p.name||"dataset")}</div>`
      + `<div class="prj-path" title="${esc(p.data)}">${esc(p.data)}</div>`
      + `<div class="prj-barwrap"><div class="prj-bar" style="width:${pct}%"></div></div>`
      + `<div class="prj-meta"><span>${n} images · ${l} labeled</span><span>${esc(relTime(p.last_opened))}</span></div>`
      + `</button>`;
  }).join("");
  grid.querySelectorAll(".prj").forEach(b=> b.onclick=(e)=>{ if(e.target.closest(".prj-forget")) return; openProject(b.dataset.data); });
  grid.querySelectorAll(".prj-forget").forEach(b=> b.onclick=async (e)=>{ e.stopPropagation();
    try{ await fetch("/api/projects/forget",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({data:b.dataset.forget})}); }catch(_e){}
    renderProjects();
  });
}
async function openProject(data){
  if(!data){ homeError("Enter a path to a data.yaml or a dataset folder."); return; }
  // Persist the open image's edits before swapping projects: resetClientState()
  // zeroes `dirty` and clears boxes/polys, so without this the work is lost silently
  // (not even the beforeunload warning fires). Same guard as backToHome/fixDuplicate.
  if(dirty && idx>=0 && !(await save())){ homeError("Save the current image before switching projects."); return; }
  if(dirty){ homeError("You edited while saving — save before switching projects."); return; }
  homeError("");
  const btn=$("#homeopen"), t=btn.textContent; btn.disabled=true; btn.textContent="Opening…";
  try{
    const r = await fetch("/api/projects/open",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({data})});
    const d = await r.json();
    if(!r.ok || !d.open){ homeError(d.error||"Could not open that dataset"); return; }
    resetClientState();
    await enterLabeler(d);
    const hp=$("#homepath"); if(hp) hp.value="";
  }catch(e){ showHome(); homeError("Could not open that dataset"); }
  finally{ btn.disabled=false; btn.textContent=t; }
}
function resetClientState(){
  loadSeq++;            // invalidate any in-flight load()/poll/stream from the previous project
  clearTimeout(boostTimer); boostTimer=null;
  clearTimeout(autosaveTimer); autosaveTimer=null;
  idx=-1; sel=-1; selPoly=-1; selBoxes.clear(); hover=-1; active=0; boxes=[]; polys=[]; ghosts=[]; radarFindings=[];
  IMAGES=[]; suggestedIds=new Set(); selSet=null; listFilter="all"; radarDeck=[]; mapPoints=[]; mapFit=null; progSig="";
  undoStack=[]; redoStack=[]; gestureSnap=null; dirty=false; imgOk=false; gradData=null; assist=null; assistModel=null;
  setTool("box");
  imgQuery=""; const si=$("#imgsearch"); if(si) si.value="";
  document.querySelectorAll("#filter button").forEach(x=>x.classList.toggle("on", x.dataset.f==="all"));
  const bc=$("#boostchip"); if(bc) bc.className="chip";
  ["#radar","#mapmodal","#insights","#sharepop","#picker"].forEach(s=>{ const m=$(s); if(m) m.classList.remove("show"); });
  const _help=$("#help"); if(_help) _help.style.display="none";
  const b=$("#banner"); if(b) b.style.display="none";
}
function relTime(epoch){
  if(!epoch) return "";
  const s = Math.max(0, (Date.now()/1000) - epoch);
  if(s<60) return "just now";
  if(s<3600) return Math.floor(s/60)+"m ago";
  if(s<86400) return Math.floor(s/3600)+"h ago";
  if(s<2592000) return Math.floor(s/86400)+"d ago";
  return Math.floor(s/2592000)+"mo ago";
}

// ---- teammate share ----
function urlRow(u){ return `<div class="urlrow"><code>${esc(u)}</code><button class="copybtn" data-u="${esc(u)}" title="Copy">${ICO_COPY}</button></div>`; }
async function toggleShare(){
  const pop=$("#sharepop"); if(!pop) return;
  if(pop.classList.contains("show")){ pop.classList.remove("show"); return; }
  pop.classList.add("show"); pop.innerHTML = `<p>Loading…</p>`;
  let s; try{ s=await jget("/api/server"); }catch(e){ pop.innerHTML=`<p>Could not read server info.</p>`; return; }
  if(s.shareable && s.lan_url){
    pop.innerHTML = `<h4>Label together</h4><p>Send this link to teammates on your network — they open the same dataset and you split the images (each save writes its own file).</p>` + urlRow(s.lan_url);
  } else {
    pop.innerHTML = `<h4>Running locally</h4><p>Only this machine can reach it. To let teammates on your network join, restart with <code>--share</code>.</p>` + urlRow(s.local_url);
  }
  pop.querySelectorAll(".copybtn").forEach(b=> b.onclick=()=>{
    try{ navigator.clipboard.writeText(b.dataset.u); }catch(e){}
    b.classList.add("copied"); b.innerHTML=ICO_CHECK;
    setTimeout(()=>{ b.classList.remove("copied"); b.innerHTML=ICO_COPY; }, 1200);
  });
}

init().catch(err=>{ document.body.insertAdjacentHTML("afterbegin",
  `<div style="padding:14px;color:#f5b13d">LibreLabel failed to start: ${err.message}</div>`); });
</script>
</body>
</html>
"""
