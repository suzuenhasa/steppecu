// src/app/pca_html_writer.cpp
//
// The self-contained interactive PCA scatter emitter. write_pca_html serializes a PcaResult
// into ONE HTML file with everything inline (CSS in <style>, the PC coordinates in a JSON
// data literal, the renderer in <script>) and ZERO external references — no CDN, no <link>,
// no remote font, no network fetch — so the artifact opens offline. The renderer is a
// dependency-free canvas-2D scatter (no inline SVG, so no xmlns/DTD URL trips the network
// self-containment grep): PC-axis X/Y selectors, pan (drag) + zoom (wheel), nearest-point
// hover labels, a click-to-toggle population legend, and a scree strip.
#include "app/pca_html_writer.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <ios>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace steppe::app {

namespace {

// JSON-escape a string for embedding inside the inline data literal.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '/': out += "\\/"; break;  // escape '/' so no "</script>" can appear verbatim
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c) & 0xff);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Compact finite-double formatter (10 significant digits); non-finite -> JSON null.
std::string jnum(double v) {
    if (!std::isfinite(v)) return "null";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.10g", v);
    return std::string(buf);
}

// Emit the `const DATA = {...};` literal: axis names, eigen spectrum, populations, and one
// {id, pop-index, coords[]} record per sample. pop is an index into `pops` (keeps it small).
void write_data_literal(std::ostream& os, const PcaResult& r) {
    std::unordered_map<std::string, int> pop_index;
    for (std::size_t p = 0; p < r.pop_labels.size(); ++p)
        pop_index[r.pop_labels[p]] = static_cast<int>(p);

    const int K = r.K;
    const bool has_proj = r.n_ref > 0 && r.n_ref < r.N;
    os << "const DATA = {\n";
    os << "  \"hasProjected\": " << (has_proj ? "true" : "false") << ",\n";
    os << "  \"axisNames\": [";
    for (int k = 0; k < K; ++k) os << (k ? "," : "") << "\"PC" << (k + 1) << "\"";
    os << "],\n";
    os << "  \"eigenvalues\": [";
    for (int k = 0; k < K; ++k)
        os << (k ? "," : "")
           << jnum(static_cast<std::size_t>(k) < r.eigenvalues.size() ? r.eigenvalues[static_cast<std::size_t>(k)] : 0.0);
    os << "],\n";
    os << "  \"varExplained\": [";
    for (int k = 0; k < K; ++k)
        os << (k ? "," : "")
           << jnum(static_cast<std::size_t>(k) < r.var_explained.size() ? r.var_explained[static_cast<std::size_t>(k)] : 0.0);
    os << "],\n";
    os << "  \"pops\": [";
    for (std::size_t p = 0; p < r.pop_labels.size(); ++p)
        os << (p ? "," : "") << "\"" << json_escape(r.pop_labels[p]) << "\"";
    os << "],\n";
    os << "  \"samples\": [\n";
    const int N = r.N;
    for (int i = 0; i < N; ++i) {
        const std::string& id = (static_cast<std::size_t>(i) < r.sample_id.size())
                                    ? r.sample_id[static_cast<std::size_t>(i)] : std::string();
        const std::string& pop = (static_cast<std::size_t>(i) < r.sample_pop.size())
                                     ? r.sample_pop[static_cast<std::size_t>(i)] : std::string();
        const auto it = pop_index.find(pop);
        const int pidx = (it != pop_index.end()) ? it->second : 0;
        const int pj = (static_cast<std::size_t>(i) < r.is_projected.size() &&
                        r.is_projected[static_cast<std::size_t>(i)])
                           ? 1
                           : 0;
        os << "    {\"id\":\"" << json_escape(id) << "\",\"pop\":" << pidx << ",\"pj\":" << pj
           << ",\"c\":[";
        for (int k = 0; k < K; ++k) {
            const std::size_t off = static_cast<std::size_t>(i) * static_cast<std::size_t>(K) +
                                    static_cast<std::size_t>(k);
            os << (k ? "," : "") << jnum(off < r.coords.size() ? r.coords[off] : 0.0);
        }
        os << "]}" << (i + 1 < N ? "," : "") << "\n";
    }
    os << "  ]\n";
    os << "};\n";
}

// The renderer (CSS + DOM + vanilla-JS canvas-2D scatter). No external references, no SVG.
const char* kHtmlHead = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>steppe PCA</title>
<style>
:root{
  --bg:#ffffff; --fg:#1a1a1a; --muted:#666; --grid:#e2e2e2; --panel:#f6f6f7;
  --border:#dcdce0; --accent:#4E79A7; --bar:#c8c8cc;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#14161a; --fg:#e8e8ea; --muted:#9a9aa2; --grid:#2a2d33; --panel:#1c1f24;
         --border:#2e323a; --accent:#79a7d8; --bar:#3a3d44; }
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
  font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{display:flex;flex-direction:column;height:100vh}
header{padding:10px 16px;border-bottom:1px solid var(--border)}
header h1{margin:0;font-size:15px;font-weight:650}
header .sub{color:var(--muted);font-size:12px;margin-top:2px}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:8px 16px;
  border-bottom:1px solid var(--border);font-size:13px}
.controls label{color:var(--muted);margin-right:4px}
select,button{background:var(--panel);color:var(--fg);border:1px solid var(--border);
  border-radius:6px;padding:4px 8px;font-size:13px;cursor:pointer}
.main{display:flex;flex:1;min-height:0}
.plot{position:relative;flex:1;min-width:0}
#scatter{width:100%;height:100%;display:block;cursor:crosshair}
.side{width:220px;border-left:1px solid var(--border);padding:12px;overflow:auto}
.side h2{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);
  margin:0 0 8px}
.legend-item{display:flex;align-items:center;gap:8px;padding:3px 4px;border-radius:5px;
  cursor:pointer;font-size:13px}
.legend-item:hover{background:var(--panel)}
.legend-item.off{opacity:.4}
.swatch{width:12px;height:12px;border-radius:3px;flex:none}
#scree{width:100%;height:70px;display:block;margin-top:6px}
#tooltip{position:fixed;pointer-events:none;background:var(--panel);color:var(--fg);
  border:1px solid var(--border);border-radius:6px;padding:6px 8px;font-size:12px;
  display:none;box-shadow:0 2px 8px rgba(0,0,0,.25);z-index:10}
@media (max-width:640px){ .side{width:150px} }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>steppe &middot; principal component analysis</h1>
    <div class="sub" id="subtitle"></div>
  </header>
  <div class="controls">
    <span><label>X</label><select id="xaxis"></select></span>
    <span><label>Y</label><select id="yaxis"></select></span>
    <button id="reset">Reset view</button>
    <span class="sub" id="counts" style="color:var(--muted)"></span>
  </div>
  <div class="main">
    <div class="plot"><canvas id="scatter"></canvas></div>
    <div class="side">
      <h2>Populations</h2>
      <div id="legend"></div>
      <h2 style="margin-top:14px">Scree</h2>
      <canvas id="scree"></canvas>
    </div>
  </div>
</div>
<div id="tooltip"></div>
<script>
)HTML";

const char* kHtmlScript = R"HTML(
(function(){
  const palette = ["#4E79A7","#F28E2B","#59A14F","#E15759","#B07AA1","#76B7B2",
                   "#EDC948","#FF9DA7","#9C755F","#BAB0AC","#1F77B4","#D62728",
                   "#2CA02C","#9467BD","#8C564B","#E377C2"];
  const canvas=document.getElementById('scatter'), ctx=canvas.getContext('2d');
  const tooltip=document.getElementById('tooltip');
  const xsel=document.getElementById('xaxis'), ysel=document.getElementById('yaxis');
  const legendEl=document.getElementById('legend');
  const K=DATA.axisNames.length;
  const popVisible=DATA.pops.map(()=>true);
  const view={scale:1,tx:0,ty:0};

  document.getElementById('subtitle').textContent =
    DATA.samples.length+" samples · "+DATA.pops.length+" populations · "+K+" PCs";
  const c=document.getElementById('counts');

  function cssvar(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()||'#888';}
  function color(p){return palette[p%palette.length];}
  function esc(s){return String(s).replace(/[&<>]/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[x]));}

  for(let a=0;a<K;a++){
    const pct=DATA.varExplained[a]!=null?" ("+(DATA.varExplained[a]*100).toFixed(1)+"%)":"";
    for(const sel of [xsel,ysel]){const o=document.createElement('option');o.value=a;
      o.textContent=DATA.axisNames[a]+pct;sel.appendChild(o);}
  }
  xsel.value=0; ysel.value=Math.min(1,K-1);
  function axes(){return [parseInt(xsel.value,10),parseInt(ysel.value,10)];}

  function buildLegend(){
    legendEl.innerHTML='';
    DATA.pops.forEach((name,p)=>{
      const item=document.createElement('div');
      item.className='legend-item'+(popVisible[p]?'':' off');
      const sw=document.createElement('span');sw.className='swatch';sw.style.background=color(p);
      const lbl=document.createElement('span');lbl.textContent=name;
      item.appendChild(sw);item.appendChild(lbl);
      item.onclick=()=>{popVisible[p]=!popVisible[p];buildLegend();draw();};
      legendEl.appendChild(item);
    });
  }
  function dataBounds(ax,ay){
    let a=Infinity,b=-Infinity,cc=Infinity,d=-Infinity;
    for(const s of DATA.samples){if(!popVisible[s.pop])continue;
      const x=s.c[ax],y=s.c[ay];
      if(x<a)a=x;if(x>b)b=x;if(y<cc)cc=y;if(y>d)d=y;}
    if(!isFinite(a)){a=-1;b=1;cc=-1;d=1;}
    return{minx:a,maxx:b,miny:cc,maxy:d};
  }
  function resize(){
    const r=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
    canvas.width=Math.max(1,Math.floor(r.width*dpr));
    canvas.height=Math.max(1,Math.floor(r.height*dpr));
    ctx.setTransform(dpr,0,0,dpr,0,0);
  }
  function fit(){
    const [ax,ay]=axes(),bd=dataBounds(ax,ay),r=canvas.getBoundingClientRect(),pad=44;
    const w=r.width-2*pad,h=r.height-2*pad;
    const dx=(bd.maxx-bd.minx)||1,dy=(bd.maxy-bd.miny)||1;
    const s=Math.min(w/dx,h/dy);
    view.scale=s;
    view.tx=pad+(w-s*dx)/2-s*bd.minx;
    view.ty=pad+(h-s*dy)/2+s*bd.maxy;
  }
  function w2s(x,y){return[view.tx+view.scale*x,view.ty-view.scale*y];}
  function s2w(px,py){return[(px-view.tx)/view.scale,(view.ty-py)/view.scale];}

  function draw(){
    const [ax,ay]=axes(),r=canvas.getBoundingClientRect();
    ctx.clearRect(0,0,r.width,r.height);
    ctx.strokeStyle=cssvar('--grid');ctx.lineWidth=1;
    const o=w2s(0,0);
    ctx.beginPath();ctx.moveTo(0,o[1]);ctx.lineTo(r.width,o[1]);
    ctx.moveTo(o[0],0);ctx.lineTo(o[0],r.height);ctx.stroke();
    let shown=0;
    for(const s of DATA.samples){if(!popVisible[s.pop])continue;shown++;
      const p=w2s(s.c[ax],s.c[ay]);
      if(s.pj){
        // Projected (lsqproject) sample: a hollow diamond so it reads apart from the reference.
        const rr=4.2;ctx.beginPath();
        ctx.moveTo(p[0],p[1]-rr);ctx.lineTo(p[0]+rr,p[1]);
        ctx.lineTo(p[0],p[1]+rr);ctx.lineTo(p[0]-rr,p[1]);ctx.closePath();
        ctx.globalAlpha=0.95;ctx.lineWidth=1.6;ctx.strokeStyle=color(s.pop);ctx.stroke();
        ctx.globalAlpha=1;ctx.lineWidth=1;
      }else{
        ctx.beginPath();ctx.arc(p[0],p[1],3.3,0,6.2832);
        ctx.globalAlpha=0.82;ctx.fillStyle=color(s.pop);ctx.fill();ctx.globalAlpha=1;
      }
    }
    ctx.fillStyle=cssvar('--fg');ctx.font='13px system-ui,sans-serif';
    ctx.fillText(DATA.axisNames[ax],r.width-72,o[1]-8);
    ctx.save();ctx.translate(o[0]+8,18);ctx.fillText(DATA.axisNames[ay],0,0);ctx.restore();
    c.textContent=shown+" of "+DATA.samples.length+" points";
    drawScree(ax,ay);
  }
  function drawScree(ax,ay){
    const sc=document.getElementById('scree');if(!sc)return;const g=sc.getContext('2d');
    const r=sc.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
    sc.width=Math.max(1,Math.floor(r.width*dpr));sc.height=Math.max(1,Math.floor(r.height*dpr));
    g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,r.width,r.height);
    const ve=DATA.varExplained,n=ve.length;if(!n)return;
    const mv=Math.max.apply(null,ve.map(v=>v||0))||1,bw=r.width/n;
    for(let i=0;i<n;i++){const hh=(ve[i]||0)/mv*(r.height-4);
      g.fillStyle=(i===ax||i===ay)?cssvar('--accent'):cssvar('--bar');
      g.fillRect(i*bw+1,r.height-hh,Math.max(1,bw-2),hh);}
  }

  let drag=false,lx=0,ly=0;
  canvas.addEventListener('mousedown',e=>{drag=true;lx=e.clientX;ly=e.clientY;});
  window.addEventListener('mouseup',()=>{drag=false;});
  canvas.addEventListener('mousemove',e=>{
    const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    if(drag){view.tx+=e.clientX-lx;view.ty+=e.clientY-ly;lx=e.clientX;ly=e.clientY;
      tooltip.style.display='none';draw();return;}
    hover(mx,my,e.clientX,e.clientY);
  });
  canvas.addEventListener('mouseleave',()=>{tooltip.style.display='none';});
  canvas.addEventListener('wheel',e=>{e.preventDefault();
    const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    const wpt=s2w(mx,my),f=Math.exp(-e.deltaY*0.001);view.scale*=f;
    const ns=w2s(wpt[0],wpt[1]);view.tx+=mx-ns[0];view.ty+=my-ns[1];draw();
  },{passive:false});
  function hover(mx,my,cx,cy){
    const [ax,ay]=axes();let best=null,bd=64;
    for(const s of DATA.samples){if(!popVisible[s.pop])continue;
      const p=w2s(s.c[ax],s.c[ay]),dd=(p[0]-mx)*(p[0]-mx)+(p[1]-my)*(p[1]-my);
      if(dd<bd){bd=dd;best=s;}}
    if(best){tooltip.style.display='block';tooltip.style.left=(cx+12)+'px';tooltip.style.top=(cy+12)+'px';
      tooltip.innerHTML='<b>'+esc(best.id)+'</b><br>'+esc(DATA.pops[best.pop])+'<br>'+
        DATA.axisNames[ax]+' '+best.c[ax].toFixed(4)+'<br>'+DATA.axisNames[ay]+' '+best.c[ay].toFixed(4);
    }else tooltip.style.display='none';
  }
  function reflow(){resize();fit();draw();}
  xsel.onchange=()=>{fit();draw();};ysel.onchange=()=>{fit();draw();};
  document.getElementById('reset').onclick=()=>{fit();draw();};
  window.addEventListener('resize',reflow);
  buildLegend();reflow();
})();
</script>
</body>
</html>
)HTML";

}  // namespace

bool write_pca_html(const std::string& path, const PcaResult& result, const char* prefix) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "steppe %s: cannot open --emit-html file: %s\n", prefix, path.c_str());
        return false;
    }
    out << kHtmlHead;
    write_data_literal(out, result);
    out << kHtmlScript;
    out.flush();
    if (!out.good()) {
        std::fprintf(stderr, "steppe %s: write failed for --emit-html file: %s\n", prefix,
                     path.c_str());
        return false;
    }
    return true;
}

}  // namespace steppe::app
