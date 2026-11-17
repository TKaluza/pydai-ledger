"""Self-contained HTML report of one ledger.

One file without external resources: charts are inline SVG, fonts are embedded, the page
opens light and has a switch to dark. The cases are embedded as gzip-compressed
JSON and shown by a little JavaScript as a compact table with filter, search, sorting
and paging; large case sets are only unpacked when the reader asks for them. All
figures come from :mod:`pydai_ledger.summary`, shared with the PDF report.
Image previews need Pillow and the QR code needs segno (``pydai-ledger[report]``).
"""

from __future__ import annotations

import base64
import gzip
import json
import os
from collections.abc import Callable, Sequence
from html import escape
from pathlib import Path
from typing import Any

from .casebook import Casebook
from .config import Config
from .header import ReportHeader
from .ledger import Ledger
from .summary import GRID_COLOURS, MAX_SERIES, _curve, _result_grid, build

AUTOLOAD_BYTES = 1_500_000  # larger case payloads are unpacked on request
FONTS = Path(__file__).parent / "fonts"

_CSS = """
:root{color-scheme:light;--page:#f5f4f0;--surface:#fcfcfb;--text:#0b0b0b;--text2:#0b0b0b;--muted:#0b0b0b;
--grid:#e1e0d9;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;
--s6:#008300;--s7:#6250d6;--s8:#e34948;--good:#0ca30c;--warning:#fab219;--critical:#d03b3b;
--warn-bg:#fff4d1;--crit-bg:#fde3e1;--code:#f0efec}
:root[data-theme=dark]{color-scheme:dark;--page:#121211;--surface:#1a1a19;--text:#f0efec;
--text2:#f0efec;--muted:#f0efec;--grid:#2c2c2a;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--warn-bg:#3a3010;--crit-bg:#3d1a19;--code:#262624}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--text);
font:14px/1.5 Lato,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0;font-weight:700}h2{font-size:18px;font-weight:700;margin:44px 0 10px}h3{font-size:14px;margin:0 0 8px}h3.sec{font-size:15px;margin:28px 0 6px}
.theme{display:block;margin:-10px 0 8px auto;font:inherit;font-size:12px;color:var(--text);background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:3px 9px;cursor:pointer}
@media print{.theme{display:none}}
header{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:18px}
.qr{flex:none;text-align:right}.qr svg{width:92px;height:92px;display:block;margin-left:auto;background:#fff}.qr small{color:var(--muted);font-size:10px}
.meta{color:var(--text2);margin:6px 0;font-size:13px}header .meta{margin:0}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin:0}
.facts div{padding:10px 14px 10px 0}.facts dt{font-size:12px;color:var(--text2)}
.facts dd{margin:2px 0 0;font-size:24px;font-weight:700;line-height:1.2;font-variant-numeric:tabular-nums}.facts small{color:var(--muted);font-size:12px}
.notes{list-style:none;margin:14px 0 0;padding:0;font-size:13px;display:grid;grid-template-columns:max-content 1fr;gap:6px 14px}
.notes li{display:contents}.notes b{border-left:4px solid var(--warning);padding-left:8px;font-weight:700}
.notes li.error b{border-left-color:var(--critical)}.notes li.info b{border-left-color:var(--muted)}
.notes details{margin-top:2px;color:var(--text2)}.notes summary{cursor:pointer;color:var(--muted);font-size:12px}.notes details p{margin:4px 0 6px;max-width:70ch}
.mx{width:auto}.mx th{border:0;font-weight:700}.mx td{font-family:"Ubuntu Mono",ui-monospace,Consolas,monospace;text-align:right}
rect[data-c]{cursor:pointer}
.cap{font-size:12px;margin:22px 0 4px}.remark{font-style:italic;font-size:13px;margin:6px 0}
.split{display:grid;grid-template-columns:max-content 1fr;gap:10px 32px;align-items:start;margin-top:10px}
.split .cap{margin-top:0}.split p:not(.cap){margin:0 0 8px;font-size:13px;color:var(--text2);max-width:62ch}
@media(max-width:760px){.split{grid-template-columns:1fr}}
.det{list-style:none;margin:8px 0 0;padding:0;font-size:13px;display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:4px 28px}
.det li{display:grid;grid-template-columns:76px 1fr;gap:8px}.det b{font-weight:400;font-family:"Ubuntu Mono",ui-monospace,Consolas,monospace;color:var(--muted)}
.det li.on b{color:var(--good)}.det li.off b{color:var(--critical)}.det li.mixed b{color:#a86b00}.det em{color:var(--muted);font-style:normal}.det code{color:var(--text2);font-size:12px}
td.c,th.c{text-align:center}
.sub{color:var(--text2);margin:0 0 20px}.muted{color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile .v{font-size:26px;font-weight:700;line-height:1.2}.tile .l{color:var(--text2);font-size:12px}
.curve svg{width:100%;max-width:680px}
figure{margin:0 0 14px}figcaption{color:var(--text2);font-size:13px;margin-top:6px;max-width:75ch}figure svg{max-width:100%}
figure .scroll svg{min-width:640px}
.note{border-radius:10px;padding:10px 14px;margin:8px 0;border:1px solid var(--border);display:flex;gap:10px}
.note.warn{background:var(--warn-bg);border-left:5px solid var(--warning)}
.note.crit{background:var(--crit-bg);border-left:5px solid var(--critical)}
.note b{white-space:nowrap}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:5px 10px;
vertical-align:top}th{font-weight:700;border-bottom:1px solid var(--text2)}tr:hover td{background:var(--code)}
td.n,th.n{text-align:right}td.n,.facts dd{font-family:"Ubuntu Mono",ui-monospace,Consolas,monospace}
.scroll{overflow-x:auto}
code,pre{font:12.5px/1.45 "Ubuntu Mono",ui-monospace,Consolas,monospace;background:var(--code);border-radius:5px}
code{padding:1px 5px}pre{padding:10px;overflow:auto;max-height:340px;margin:6px 0;white-space:pre-wrap;word-break:break-word}
details{margin:6px 0}summary{cursor:pointer;padding:4px 0}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--text2);margin-bottom:4px}
.sw{display:inline-block;width:10px;height:10px;margin-right:5px}
.meter{display:inline-block;width:70px;height:6px;background:var(--grid);border-radius:3px;margin-left:8px;vertical-align:middle}
.meter i{display:block;height:6px;border-radius:3px;background:var(--s1)}
.tag{display:inline-block;border-radius:5px;padding:0 6px;font-size:12px;border:1px solid var(--border);margin-right:4px}
.tag.bad{background:var(--crit-bg)}.tag.warn{background:var(--warn-bg)}
.thumbs img{max-width:160px;max-height:160px;border-radius:6px;border:1px solid var(--border);margin:4px 6px 4px 0}
svg text{font:11px Lato,system-ui,sans-serif;fill:var(--muted)}svg .lab{fill:var(--text2)}
.bar{display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center;margin-bottom:10px}
.bar button,.bar input,.bar select,.load{font:inherit;font-size:12px;color:var(--text);background:var(--surface);
border:1px solid var(--border);border-radius:6px;padding:3px 9px}
.bar button,.load{cursor:pointer}.bar button.on{background:var(--text);color:var(--surface)}
.bar input{flex:1;min-width:160px}.bar .sp{flex:1}.bar button:disabled{opacity:.4;cursor:default}
.ct{table-layout:fixed;min-width:760px;font-size:12px}
.ct th{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:4px 8px}.ct th.so{cursor:pointer}
.ct td{padding:2px 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.5}
.ct tr.r{cursor:pointer}.ct tr.r:hover td{background:var(--code)}
.ct td.o{font-family:"Ubuntu Mono",ui-monospace,Consolas,monospace;font-size:12px}
.ct td.bad{background:var(--crit-bg)}.ct td.dif{background:var(--warn-bg)}
.ct tr.d td{white-space:normal;background:var(--page);padding:10px 14px}
.ct tr.d pre{max-height:200px;margin:3px 0;padding:6px 8px}.ct tr.d table{width:auto;min-width:50%}
.ct tr.d table td{padding:2px 10px 2px 0;background:none}
.ct tr.d h4{margin:8px 0 2px;font-size:12px;color:var(--text2)}
.st{display:inline-block;border-radius:4px;padding:0 4px;margin-right:3px;font-size:11px;border:1px solid var(--border)}
.st.bad{background:var(--crit-bg)}.st.warn{background:var(--warn-bg)}
#tip{position:fixed;pointer-events:none;background:var(--text);color:var(--surface);padding:5px 8px;border-radius:6px;
font-size:12px;display:none;z-index:9;white-space:pre}
"""

_JS = r"""
// The page starts light whatever the system says; the switch is the only way to dark. Print is always light.
const themeBtn=document.querySelector('.theme'),rootEl=document.documentElement;
if(themeBtn){const names=JSON.parse(themeBtn.dataset.labels);let dark=false;
 const set=on=>{if(on)rootEl.dataset.theme='dark';else delete rootEl.dataset.theme};
 themeBtn.onclick=()=>{dark=!dark;set(dark);themeBtn.textContent=names[dark?1:0]};
 addEventListener('beforeprint',()=>set(false));addEventListener('afterprint',()=>set(dark))}
const tip=document.getElementById('tip');
const gridEl=document.getElementById('grid'),G=gridEl?JSON.parse(gridEl.dataset.grid):null;
function cellTip(el){const c=+el.dataset.c,k=window.pydaiKeys&&window.pydaiKeys.get(c);
 return G.case+' '+c+(G.per>1?'–'+(c+G.per-1)+' ('+G.per+' '+G.cases+')':'')+(k!=null?' · '+k:'')+'\n'+G.targets[+el.dataset.t]+': '+G.states[+el.dataset.s]+'\n'+G.hint}
document.addEventListener('mousemove',e=>{const el=e.target.closest?e.target.closest('[data-tip],rect[data-c]'):null;
 if(!el){tip.style.display='none';return}
 tip.textContent=el.dataset.tip||cellTip(el);tip.style.display='block';
 tip.style.left=Math.min(e.clientX+12,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=(e.clientY+14)+'px'});
document.addEventListener('click',e=>{const el=e.target.closest?e.target.closest('rect[data-c]'):null;
 if(el&&window.pydaiShow)window.pydaiShow(+el.dataset.c)});
(function(){
const src=document.getElementById('cases-data'),root=document.getElementById('cases');
if(!src||!root)return;
const T=JSON.parse(root.dataset.targets),Lb=JSON.parse(root.dataset.labels),LOC=Lb.locale,FILTERS=[[Lb.js_all,0],['✖ '+Lb.js_failed,1],['≠ '+Lb.js_differing,2],['✗ '+Lb.js_wrong,4]];
let D,view=[],page=0,per=50,flt=0,q='',sk='i',sd=1,tbody,info,prev,next;const open=new Set();
const $=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e};
const num=(v,d)=>v==null?'–':v.toLocaleString(LOC,{minimumFractionDigits:d,maximumFractionDigits:d});
function best(c,ti){for(let j=c.t.length-1;j>=0;j--){const t=c.t[j],v=t.o.filter(o=>o[0]===ti);
  if(v.length){v.sort((a,b)=>b[1]-a[1]);return{text:v[0][2],more:v.length-1,bad:v.some(o=>o[3]===false)}}}return null}
function hay(c){return (c.i+' '+(c.k??'')+' '+c.t.map(t=>t.p.join(' ')+' '+t.o.map(o=>o[2]).join(' ')).join(' ')+' '+c.x.map(x=>x[1]).join(' ')).toLowerCase()}
function apply(){const needle=q.trim().toLowerCase();
  view=D.cases.filter(c=>(!flt||c.f&flt)&&(!needle||hay(c).includes(needle)));
  view.sort((a,b)=>{const x=a[sk],y=b[sk];return (x==null)-(y==null)||(x<y?-sd:x>y?sd:a.i-b.i)});page=0;render()}
function tags(c){const f=document.createDocumentFragment();
  if(c.f&1){const e=$('span','st bad','✖');e.title=Lb.js_failed;f.append(e)}
  if(c.f&2){const e=$('span','st warn','≠');e.title=Lb.js_differing;f.append(e)}
  if(c.f&4){const e=$('span','st bad','✗');e.title=Lb.js_wrong;f.append(e)}
  if(!c.f)f.append($('span','muted','✓'));return f}
function detail(c){const td=$('td');td.colSpan=6+T.length;
  c.t.forEach((t,j)=>{td.append($('h4',null,Lb.turn+' '+(j+1)));t.p.forEach(p=>td.append($('pre',null,p)));
    if(t.m.length){const box=$('div','thumbs');t.m.forEach(m=>{if(m.u!=null){const im=$('img');im.src=D.thumbs[m.u];im.alt='input image';box.append(im)}
      else box.append($('span','tag',m.y+' '+m.b))});td.append(box)}
    if(t.o.length||t.e!=null){const tb=$('table');
      t.o.forEach(o=>{const tr=$('tr');tr.append($('td',null,T[o[0]]),$('td','n',o[1]+'×'));
        const out=$('td');out.append($('code',null,o[2]));if(o[3]!=null)out.append(o[3]?' ✓':' ✗');tr.append(out);tb.append(tr)});
      if(t.e!=null){const tr=$('tr');tr.append($('td','muted',Lb.expected),$('td'));const out=$('td');out.append($('code',null,t.e));tr.append(out);tb.append(tr)}
      td.append(tb)}});
  c.x.forEach(x=>{const n=$('div','note crit');n.append($('b',null,'✖ '+T[x[0]]),$('span',null,x[1]));td.append(n)});
  const tr=$('tr','d');tr.append(td);return tr}
function render(){tbody.replaceChildren();const start=page*per,rows=view.slice(start,start+per);
  rows.forEach(c=>{const tr=$('tr','r');const a=$('td','n',String(c.i));tr.append(a);
    const st=$('td');st.append(tags(c));tr.append(st);
    const pr=$('td',null,(c.k!=null?c.k+' · ':'')+(c.n?'🖼 ':'')+(c.t[0]?c.t[0].p.join(' '):''));pr.title=pr.textContent;tr.append(pr);
    T.forEach((_,ti)=>{const b=best(c,ti),td=$('td','o'+(b&&b.bad?' bad':(c.f&2?' dif':'')),b?b.text+(b.more?'  +'+b.more:''):'–');
      if(b)td.title=b.text;tr.append(td)});
    tr.append($('td','n',String(c.t.length)),$('td','n',num(c.l,2)),$('td','n',num(c.o,0)));
    tr.onclick=()=>{if(open.has(c.i)){open.delete(c.i);tr.nextSibling.remove()}else{open.add(c.i);tr.after(detail(c))}};
    tbody.append(tr);if(open.has(c.i))tbody.append(detail(c))});
  info.textContent=view.length?num(start+1,0)+'–'+num(start+rows.length,0)+' '+Lb.js_of+' '+num(view.length,0):Lb.js_none;
  prev.disabled=page===0;next.disabled=start+per>=view.length}
function build(){root.replaceChildren();const bar=$('div','bar');
  FILTERS.forEach(([label,bit])=>{const n=bit?D.cases.filter(c=>c.f&bit).length:D.cases.length;
    const b=$('button',bit===flt?'on':'',label+' '+num(n,0));
    b.onclick=()=>{flt=bit;bar.querySelectorAll('button.on').forEach(x=>x.classList.remove('on'));b.classList.add('on');apply()};bar.append(b)});
  const input=$('input');input.type='search';input.placeholder=Lb.js_search;
  let timer;input.oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>{q=input.value;apply()},150)};bar.append(input);
  window.pydaiShow=i=>{const k=window.pydaiKeys.get(i);q=k!=null?String(k):String(i);input.value=q;flt=0;
    bar.querySelectorAll('button.on').forEach(x=>x.classList.remove('on'));bar.querySelector('button').classList.add('on');
    apply();if(view.some(c=>c.i===i)){open.add(i);render()}root.scrollIntoView({behavior:'smooth'})};
  const sel=$('select');[25,50,100,250].forEach(n=>{const o=$('option',null,n+' '+Lb.js_rows);o.value=n;if(n===per)o.selected=true;sel.append(o)});
  sel.onchange=()=>{per=+sel.value;page=0;render()};
  prev=$('button',null,'‹');next=$('button',null,'›');info=$('span','muted');
  prev.onclick=()=>{page--;render()};next.onclick=()=>{page++;render()};bar.append(sel,prev,info,next);
  const table=$('table','ct'),cg=$('colgroup');
  ['56px','62px','26%'].concat(T.map(()=>'auto'),['58px','66px','66px']).forEach(w=>{const c=$('col');c.style.width=w;cg.append(c)});
  const head=$('tr');
  [['#','i','n'],[Lb.js_status,'f',''],[Lb.js_prompt,null,'']].concat(T.map(t=>[t,null,'']),[[Lb.js_turns,null,'n'],[Lb.js_mean,'l','n'],[Lb.js_tokens,'o','n']])
   .forEach(([label,key,cls])=>{const th=$('th',cls+(key?' so':''),label);th.title=label+(key?' ('+Lb.js_sort+')':'');
     if(key)th.onclick=()=>{sd=sk===key?-sd:(key==='i'?1:-1);sk=key;apply()};head.append(th)});
  const thead=$('thead');thead.append(head);tbody=$('tbody');table.append(cg,thead,tbody);
  const wrap=$('div','card scroll');wrap.append(table);root.append(bar,wrap);
  if(D.note)root.append($('p','muted',D.note))}
async function load(){const btn=root.querySelector('.load');if(btn){btn.disabled=true;btn.textContent=Lb.js_unpacking}
  try{const bin=Uint8Array.from(atob(src.textContent),c=>c.charCodeAt(0));
    const text=await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
    D=JSON.parse(text);window.pydaiKeys=new Map(D.cases.map(c=>[c.i,c.k]));build();apply()}
  catch(e){root.replaceChildren($('p','muted',Lb.js_error+e))}}
const btn=root.querySelector('.load');if(btn)btn.onclick=load;else load();
})();
"""


def _font_css() -> str:
    """The bundled fonts as @font-face rules (Lato for text, Ubuntu Mono for code and numbers)."""
    rules = ""
    for family, weight, style, name in (
        ("Lato", 400, "normal", "Lato-Regular"),
        ("Lato", 400, "italic", "Lato-Italic"),
        ("Lato", 700, "normal", "Lato-Bold"),
        ("Ubuntu Mono", 400, "normal", "UbuntuMono-Regular"),
        ("Ubuntu Mono", 400, "italic", "UbuntuMono-Italic"),
    ):
        data = base64.b64encode((FONTS / f"{name}.ttf").read_bytes()).decode("ascii")
        rules += (
            f'@font-face{{font-family:"{family}";font-weight:{weight};font-style:{style};'
            f'src:url(data:font/ttf;base64,{data}) format("truetype")}}'
        )
    return rules


def qr_svg(text: str) -> str | None:
    """Inline SVG of a QR code, or None when segno is not installed."""
    try:
        import segno
    except ImportError:
        return None
    return segno.make(text, error="m").svg_inline(border=2, dark="#111111", light="#ffffff", omitsize=True)


def _table(t: dict[str, Any]) -> str:
    """Render a summary table as HTML."""
    cls = {"l": "", "r": ' class="n"', "c": ' class="c"'}
    head = "".join(f"<th{cls[a]}>{escape(h)}</th>" for h, a in zip(t["head"], t["align"], strict=True))
    rows = "".join(
        "<tr>" + "".join(f"<td{cls[a]}>{escape(c)}</td>" for c, a in zip(row, t["align"], strict=True)) + "</tr>"
        for row in t["rows"]
    )
    return f'<div class="scroll"><table><tr>{head}</tr>{rows}</table></div>'


def _matrix(block: dict[str, Any]) -> str:
    """Reproducibility between targets with one model as a compact table. A target against
    itself and pairs of different models stay empty; a cell turns yellow as the share of
    identical outputs falls below 100 %.
    """
    head = "".join(f"<th>{escape(t)}</th>" for t in block["targets"])
    rows = ""
    for name, texts, values in zip(block["targets"], block["rows"], block["values"], strict=True):
        cells = ""
        for text, value in zip(texts, values, strict=True):
            style = (
                f"background:rgba(250,178,25,{min(1.0, (1 - value) * 4):.2f})"
                if value is not None and value < 1
                else ""
            )
            cells += f'<td style="{style}">{escape(text)}</td>'
        rows += f"<tr><th>{escape(name)}</th>{cells}</tr>"
    return f'<div class="scroll"><table class="mx"><tr><th></th>{head}</tr>{rows}</table></div>'


def _validation(S: dict[str, Any]) -> str:
    """The validation section: setup, precision, trueness and accuracy, block by block."""
    L, out = S["L"], ""
    for block in S["validation"]:
        kind = block["kind"]
        if kind == "h2":
            out += f'<h3 class="sec">{escape(block["text"])}</h3>'
        elif kind == "text":
            out += f'<p class="meta">{escape(block["text"])}</p>'
        elif kind == "det":
            out += (
                '<ul class="det">'
                + "".join(
                    f'<li class="{i["state"]}"><b>{escape(i["word"])}</b><span>{escape(i["label"])}'
                    + (f" <em>({escape(L['d_required'])})</em>" if i["required"] else "")
                    + (f" <code>{escape(i['value'])}</code>" if i["value"] else "")
                    + "</span></li>"
                    for i in block["items"]
                )
                + "</ul>"
            )
        else:
            cap = f'<p class="cap">{escape(block["cap"])}</p>' if block["cap"] else ""
            out += cap + (_matrix(block) if kind == "matrix" else _table(block["table"]))
    return out


def _json_block(obj: Any) -> str:
    """Pretty-printed JSON in a ``<pre>`` block."""
    return f"<pre>{escape(json.dumps(obj, indent=2, ensure_ascii=False, default=str))}</pre>"


def _notes(S: dict[str, Any], section: str) -> str:
    """The notes of one section as a list: level mark and code, text, background on demand."""
    L, marks = S["L"], {"error": "✖", "warning": "⚠", "info": "ℹ"}
    items = []
    for n in (n for n in S["notes"] if n["section"] == section):
        more = "".join(
            f"<p><i>{escape(L[label])}.</i> {escape(n[key])}</p>"
            for key, label in (("background", "background"), ("remark", "remark"))
            if n[key]
        )
        if more:
            more = f"<details><summary>{escape(L['background'])}</summary>{more}</details>"
        items.append(
            f'<li class="{n["level"]}"><b>{marks[n["level"]]} {escape(n["code"])}</b>'
            f"<span>{escape(n['text'])}{more}</span></li>"
        )
    return f'<ul class="notes">{"".join(items)}</ul>' if items else ""


def write_report(
    ledger: Ledger,
    path: str | os.PathLike[str],
    *,
    casebook: Casebook | None = None,
    lang: str = "en",
    max_cases: int | None = None,
    max_images: int = 30,
    thumbnail_px: int = 256,
    header: ReportHeader | None = None,
    sections: Sequence[str] | None = None,
    embed_fonts: bool = True,
    normalize: Callable[[Any], Any] | None = None,
    config: str | os.PathLike[str] | Config | None = None,
) -> Path:
    """Write the HTML report of ``ledger`` to ``path``.

    What the report is follows from what it is given. A run ledger gives the run
    report: precision, trueness and accuracy per target, the attempts with their seeds, the cases.
    A benchmark ledger gives the benchmark report: recommended concurrency, throughput
    and latency per level. Each ledger gets its own report. ``sections`` limits a
    report to some of ``"targets"``, ``"benchmark"``,
    ``"validation"`` and ``"cases"``.

    ``header`` is a :class:`ReportHeader` with title, authors, number and date.
    ``lang`` is ``"en"`` or ``"de"``. With ``casebook`` the report also shows keys,
    prompts, image previews and the match with expected outputs. All cases go into the
    case table unless ``max_cases`` limits it (failed cases first, then differing
    outputs, then outputs that are not as expected). Only the first ``max_images``
    images in that order get a preview of ``thumbnail_px`` pixels; the others are listed
    by type and hash. ``embed_fonts=False`` leaves the bundled fonts out (about 600 KB)
    and falls back to fonts installed on the reader's machine. ``normalize`` is applied to
    outputs before they are compared, see :meth:`Ledger.agreement`. ``config`` is a
    :class:`Config` or the path of a ``pydai-ledger.toml`` that sets the level of each
    note; by default the file in the working directory is used when it exists.
    """
    S = build(
        ledger,
        casebook,
        lang=lang,
        header=header,
        sections=sections,
        max_cases=max_cases,
        max_images=max_images,
        thumbnail_px=thumbnail_px,
        normalize=normalize,
        config=config,
    )
    L, show = S["L"], S["show"]
    h: list[str] = []
    add = h.append

    qr = qr_svg(S["stamp"])
    head = "<br>".join(escape(line) for line in S["head"])
    add(
        f'<header><div><h1>{escape(S["title"])}</h1><p class="meta">{head}</p></div>'
        + (
            f'<div class="qr" data-tip="{escape(S["stamp"], quote=True)}">{qr}<small>{escape(L["stamp"])}</small></div>'
            if qr
            else ""
        )
        + "</header>"
    )
    add(
        '<dl class="facts">'
        + "".join(
            f"<div><dt>{escape(k)}</dt><dd>{escape(v)}</dd><small>{escape(n)}</small></div>" for k, v, n in S["facts"]
        )
        + "</dl>"
    )
    add(_notes(S, ""))
    if S["config_line"]:
        add(f'<p class="meta">{escape(S["config_line"])}</p>')

    if show["targets"]:
        add(f'<h2>{escape(L["models"])}</h2><p class="cap">{escape(L["hardware_title"])}</p>{_table(S["models"])}')
        add(f'<p class="cap">{escape(L["setup_title"])}</p>{_table(S["capabilities"])}')

    figure = 0
    if show["benchmark"]:
        add(f"<h2>{escape(L['throughput'])}</h2>")
        add(_notes(S, "throughput"))
        if S["curve"]:
            figure += 1
            legend = "".join(
                f'<span><i class="sw" style="background:var(--s{i + 1})"></i>{escape(name)}</span>'
                for i, name in enumerate(S["series_names"][:MAX_SERIES])
            )
            add(
                f'<figure><div class="legend">{legend}</div><div class="curve">{_curve(**S["curve"])}</div>'
                f"<figcaption><b>{escape(L['figure'])} {figure}.</b> {escape(S['chart_caption'])}</figcaption></figure>"
            )
        if S["charts_note"]:
            add(f'<p class="remark">{escape(S["charts_note"])}</p>')
        add(_table(S["throughput"]))
        if S["tuning"]:
            add(f'<p class="cap">{escape(L["result_title"])}</p>{_table(S["tuning"])}')
        if S["server"]:
            add(f'<h3 style="margin-top:18px">{escape(L["server_means"])}</h3>{_table(S["server"])}')

    payload = S["cases"]
    packed = base64.b64encode(
        gzip.compress(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0)
    ).decode("ascii")
    if show["validation"]:
        add(f"<h2>{escape(L['validation'])}</h2>{_validation(S)}")
        if S["model_pairs"]:
            add(
                f"<h2>{escape(L['model_comparison'])}</h2>"
                f'<p class="meta">{escape(L["model_comparison_note"])}</p>{_table(S["model_pairs"])}'
            )
        add(f"<h2>{escape(L['execution_title'])}</h2>{_table(S['per_target'])}")
        if S["execution"]:
            add(f'<p class="remark">{escape(S["execution"])}</p>')

        grid, per_cell = _result_grid(S["states"], L["axis_case"], ids=S["case_ids"])
        figure += 1
        legend = "".join(
            f'<span><i class="sw" style="background:{GRID_COLOURS[state]}"></i>{escape(L[key])}</span>'
            for state, key in ((1, "grid_ok"), (2, "grid_dev"), (3, "grid_fail"), (0, "grid_none"))
        )
        caption = L["fig_grid"].format(agg=L["grid_agg"].format(k=per_cell) if per_cell > 1 else "")
        if len(S["targets"]) > 1:
            caption += L["grid_rows"].format(names=", ".join(S["targets"]))
        grid_info = {
            "targets": S["targets"],
            "states": [L["grid_none"], L["grid_ok"], L["grid_dev"], L["grid_fail"]],
            "case": L["case"],
            "cases": L["cases_per_cell"],
            "per": per_cell,
            "hint": L["grid_click"],
        }
        add(
            f'<h2>{escape(L["results"])}</h2><figure id="grid" data-grid="{escape(json.dumps(grid_info, ensure_ascii=False), quote=True)}">'
            f'<div class="legend">{legend}</div><div class="scroll">{grid}</div>'
            f"<figcaption><b>{escape(L['figure'])} {figure}.</b> {escape(caption)}</figcaption></figure>"
        )

    if show["cases"]:
        add(f'<h2>{escape(L["cases"])}</h2><p class="meta">{escape(S["cases_note"])}. {escape(L["cases_hint"])}</p>')
        button = ""
        if len(packed) > AUTOLOAD_BYTES:
            text = L["js_load"].format(n=f"{len(payload['cases']):,}", mb=f"{len(packed) / 1e6:.1f}")
            button = f'<button class="load">{escape(text)}</button>'
        labels = {k: v for k, v in L.items() if k.startswith("js_") or k in ("turn", "expected")}
        labels["locale"] = "de-DE" if S["lang"] == "de" else "en-US"
        add(
            f'<div id="cases" data-targets="{escape(json.dumps(S["targets"]), quote=True)}" '
            f'data-labels="{escape(json.dumps(labels, ensure_ascii=False), quote=True)}">{button}</div>'
        )

    add(f"<h2>{escape(L['run_details'])}</h2>")
    if S["runs"]:
        add(
            f'<div class="card" style="margin-bottom:12px"><h3>{escape(L["runs"])}</h3>'
            + "".join(
                f'<p class="meta">{escape(", ".join(r["targets"]))} · {escape(r["sealed_at"])}</p>'
                f'<pre style="white-space:pre-wrap;word-break:break-all">{escape(r["stamp"])}</pre>'
                for r in S["runs"]
            )
            + "</div>"
        )
    add(
        f'<p class="meta">{escape(L["run"])} <code>{S["run_id"][:12]}</code> · {escape(L["casebook"])} <code>{S["casebook_id"][:12]}</code> · '
        f"{escape(L['sealed'])} {escape(S['sealed_at'][:19].replace('T', ' '))} UTC · {escape(S['signed'])}</p>"
        f"<details><summary>{escape(L['stamp'])}</summary><pre>{escape(S['stamp'])}</pre></details>"
    )
    add(f'<div class="card"><h3>{escape(L["instructions"])}</h3>')
    for text in S["instructions"]:
        add(f"<pre>{escape(text)}</pre>")
    add(f"<details><summary>{escape(L['settings'])}</summary>{_json_block(S['agent'])}</details>")
    if S["request_parameters"]:
        add(
            f"<details><summary>{escape(L['request_parameters'])} ({len(S['request_parameters'])})</summary>{_json_block(S['request_parameters'])}</details>"
        )
    add("</div>")
    for target in S["plan_targets"]:
        snapshot = S["snapshots"].get(target["name"], {})
        add(
            f'<div class="card" style="margin-top:12px"><h3>Target {escape(target["name"])}</h3>'
            f"<details><summary>{escape(L['snapshot'])}</summary>{_json_block({'target': target, 'snapshot': snapshot})}</details></div>"
        )
    add(
        f'<div class="card" style="margin-top:12px"><h3>{escape(L["environment"])}</h3>{_json_block(S["environment"])}</div>'
    )

    html = (
        f'<!doctype html><html lang="{S["lang"]}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(S['title'])} {S['run_id'][:12]}</title><style>{_font_css() if embed_fonts else ''}{_CSS}</style></head>"
        f'<body><main><button class="theme" type="button" '
        f'data-labels="{escape(json.dumps([L["theme_dark"], L["theme_light"]]), quote=True)}">'
        f'{escape(L["theme_dark"])}</button>{"".join(h)}</main><div id="tip"></div>'
        f'<script id="cases-data" type="application/octet-stream">{packed}</script><script>{_JS}</script></body></html>'
    )
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
