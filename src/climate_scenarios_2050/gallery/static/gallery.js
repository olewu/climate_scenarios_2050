// Shared code of the comparison pages: data loading (JSON, byte ranges of the export files) and map drawing.
"use strict";
const G = {};
G.RDBU = [[5,48,97],[6,50,100],[7,52,103],[8,54,106],[9,56,109],[10,59,112],[12,61,115],[13,63,118],[14,65,121],[15,67,123],[16,69,126],[17,71,129],[18,73,132],[19,76,135],[20,78,138],[21,80,141],[23,82,144],[24,84,147],[25,86,150],[26,88,153],[27,90,156],[28,92,159],[29,95,162],[30,97,165],[31,99,168],[32,101,171],[34,103,172],[35,105,173],[36,106,174],[38,108,175],[39,110,176],[40,112,177],[42,113,178],[43,115,179],[44,117,180],[46,119,181],[47,121,181],[48,122,182],[50,124,183],[51,126,184],[52,128,185],[54,129,186],[55,131,187],[56,133,188],[58,135,189],[59,136,190],[60,138,190],[62,140,191],[63,142,192],[64,143,193],[66,145,194],[67,147,195],[70,149,196],[73,151,197],[76,153,198],[79,155,199],[82,157,200],[86,159,201],[89,161,202],[92,163,203],[95,165,205],[98,167,206],[101,169,207],[104,171,208],[107,172,209],[110,174,210],[113,176,211],[117,178,212],[120,180,213],[123,182,214],[126,184,215],[129,186,216],[132,188,217],[135,190,218],[138,192,219],[141,194,220],[144,196,221],[147,198,222],[150,199,223],[152,200,224],[155,201,224],[157,203,225],[160,204,226],[162,205,227],[165,206,227],[167,208,228],[169,209,229],[172,210,229],[174,211,230],[177,213,231],[179,214,232],[182,215,232],[184,216,233],[187,218,234],[189,219,234],[192,220,235],[194,221,236],[197,223,236],[199,224,237],[202,225,238],[204,226,239],[207,228,239],[209,229,240],[210,230,240],[212,230,241],[213,231,241],[215,232,241],[216,233,241],[218,233,242],[219,234,242],[221,235,242],[222,235,242],[224,236,243],[225,237,243],[227,237,243],[228,238,244],[230,239,244],[231,240,244],[233,240,244],[234,241,245],[236,242,245],[237,242,245],[239,243,245],[240,244,246],[242,245,246],[243,245,246],[245,246,247],[246,247,247],[247,246,246],[247,245,244],[248,244,242],[248,243,240],[248,242,239],[248,241,237],[249,240,235],[249,239,233],[249,238,231],[249,237,229],[249,235,227],[250,234,225],[250,233,223],[250,232,222],[250,231,220],[251,230,218],[251,229,216],[251,228,214],[251,227,212],[252,226,210],[252,224,208],[252,223,207],[252,222,205],[253,221,203],[253,220,201],[253,219,199],[253,217,196],[252,215,194],[252,213,191],[252,211,188],[251,208,185],[251,206,183],[251,204,180],[250,202,177],[250,200,175],[249,198,172],[249,196,169],[249,194,167],[248,191,164],[248,189,161],[248,187,158],[247,185,156],[247,183,153],[247,181,150],[246,179,148],[246,177,145],[246,175,142],[245,172,139],[245,170,137],[245,168,134],[244,166,131],[243,164,129],[242,161,127],[241,158,125],[240,156,123],[239,153,121],[238,150,119],[236,147,116],[235,145,114],[234,142,112],[233,139,110],[232,137,108],[230,134,106],[229,131,104],[228,128,102],[227,126,100],[226,123,98],[225,120,96],[223,118,94],[222,115,92],[221,112,89],[220,110,87],[219,107,85],[218,104,83],[216,101,81],[215,99,79],[214,96,77],[213,93,76],[211,90,74],[210,88,73],[208,85,72],[207,82,70],[206,79,69],[204,76,68],[203,73,66],[201,71,65],[200,68,64],[198,65,62],[197,62,61],[196,59,60],[194,56,58],[193,54,57],[191,51,56],[190,48,54],[189,45,53],[187,42,52],[186,40,50],[184,37,49],[183,34,48],[182,31,46],[180,28,45],[179,25,44],[177,24,43],[174,23,42],[171,22,42],[168,21,41],[165,20,41],[162,19,40],[159,18,40],[156,17,39],[153,16,39],[150,15,39],[147,14,38],[144,13,38],[141,12,37],[138,11,37],[135,10,36],[132,9,36],[129,8,35],[127,8,35],[124,7,34],[121,6,34],[118,5,33],[115,4,33],[112,3,32],[109,2,32],[106,1,31],[103,0,31]];   // matplotlib RdBu_r, 256 colours

G._json = new Map();
G.json = function (url) {
  if (!G._json.has(url)) {
    G._json.set(url, fetch(url, {cache: "no-cache"}).then(r => {
      if (!r.ok) throw new Error(url + ": " + r.status);
      return r.json();
    }));
  }
  return G._json.get(url);
};

// one map (or mask) of an export file: only its bytes are fetched (HTTP range request), results are kept
G._ranges = new Map();
G.range = function (url, offset, length) {
  const key = url + "#" + offset;
  if (!G._ranges.has(key)) {
    G._ranges.set(key, fetch(url, {headers: {Range: "bytes=" + offset + "-" + (offset + length - 1)}}).then(r => {
      if (!(r.status === 206 || r.status === 200)) throw new Error(url + ": " + r.status);
      return r.arrayBuffer().then(buf => r.status === 200 ? buf.slice(offset, offset + length) : buf);
    }));
  }
  return G._ranges.get(key);
};

// int16 export -> Float32Array (NaN for missing)
G.decode = function (buf, scale, missing) {
  const raw = new Int16Array(buf), out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw[i] === missing ? NaN : raw[i] * scale;
  return out;
};

G.unpackBits = function (buf, n) {
  const bytes = new Uint8Array(buf), out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (bytes[i >> 3] >> (7 - (i & 7))) & 1;
  return out;
};

// cos-lat weighted mean over the valid points
G.fieldMean = function (values, lat, nlon) {
  let s = 0, w = 0;
  for (let j = 0; j < lat.length; j++) {
    const cw = Math.cos(lat[j] * Math.PI / 180);
    for (let i = 0; i < nlon; i++) {
      const v = values[j * nlon + i];
      if (!Number.isNaN(v)) { s += cw * v; w += cw; }
    }
  }
  return w > 0 ? s / w : NaN;
};

// geometry of a regular lat/lon grid drawn in PlateCarree (equal degrees in x and y)
G.geometry = function (lat, lon, width) {
  const dlon = (lon[lon.length - 1] - lon[0]) / (lon.length - 1), dlat = (lat[lat.length - 1] - lat[0]) / (lat.length - 1);
  const west = Math.min(lon[0], lon[lon.length - 1]) - Math.abs(dlon) / 2, east = Math.max(lon[0], lon[lon.length - 1]) + Math.abs(dlon) / 2;
  const south = Math.min(lat[0], lat[lat.length - 1]) - Math.abs(dlat) / 2, north = Math.max(lat[0], lat[lat.length - 1]) + Math.abs(dlat) / 2;
  const height = Math.round(width * (north - south) / (east - west));
  return {west, east, south, north, width, height, dlon, dlat,
          x: lo => (lo - west) / (east - west) * width, y: la => (north - la) / (north - south) * height};
};

// draws a map: values (Float32Array, lat-major), colour limit, optional significance mask and coastlines
G.drawMap = function (canvas, grid, values, opts) {
  const {lat, lon} = grid, nlat = lat.length, nlon = lon.length;
  const geo = G.geometry(lat, lon, opts.width);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = geo.width * dpr; canvas.height = geo.height * dpr;
  canvas.style.width = geo.width + "px"; canvas.style.height = geo.height + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#e6e6e6"; ctx.fillRect(0, 0, geo.width, geo.height);
  // field at native resolution, then scaled up without smoothing (one rectangle per grid cell)
  const img = new ImageData(nlon, nlat), cmap = opts.cmap || G.RDBU, lim = opts.limit;
  const north_up = lat[0] > lat[nlat - 1];
  for (let j = 0; j < nlat; j++) {
    const row = north_up ? j : nlat - 1 - j;
    for (let i = 0; i < nlon; i++) {
      const v = values[j * nlon + i], p = 4 * (row * nlon + i);
      if (Number.isNaN(v)) { img.data[p + 3] = 0; continue; }
      const k = Math.max(0, Math.min(255, Math.floor((v + lim) / (2 * lim) * 256)));
      const c = cmap[k];
      img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255;
    }
  }
  const off = document.createElement("canvas"); off.width = nlon; off.height = nlat;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, geo.width, geo.height);
  if (opts.mask) G.drawDots(ctx, geo, opts.mask, nlat, nlon, north_up);
  if (opts.coast) G.drawCoast(ctx, geo, opts.coast);
  return geo;
};

// stippling of the significant points: a dot pattern clipped to the mask
G.drawDots = function (ctx, geo, mask, nlat, nlon, north_up) {
  const m = new ImageData(nlon, nlat);
  let any = false;
  for (let j = 0; j < nlat; j++) {
    const row = north_up ? j : nlat - 1 - j;
    for (let i = 0; i < nlon; i++) if (mask[j * nlon + i]) { m.data[4 * (row * nlon + i) + 3] = 255; any = true; }
  }
  if (!any) return;
  const mc = document.createElement("canvas"); mc.width = nlon; mc.height = nlat;
  mc.getContext("2d").putImageData(m, 0, 0);
  const layer = document.createElement("canvas"); layer.width = geo.width; layer.height = geo.height;
  const lc = layer.getContext("2d");
  lc.fillStyle = "rgba(0,0,0,0.75)";
  const step = 5;
  for (let y = 2; y < geo.height; y += step) for (let x = 2 + ((y / step) % 2) * step / 2; x < geo.width; x += step) lc.fillRect(x, y, 1.3, 1.3);
  lc.globalCompositeOperation = "destination-in";
  lc.imageSmoothingEnabled = false;
  lc.drawImage(mc, 0, 0, geo.width, geo.height);
  ctx.drawImage(layer, 0, 0, geo.width, geo.height);
};

G.drawCoast = function (ctx, geo, coast) {
  ctx.strokeStyle = "#222"; ctx.lineWidth = 0.6;
  ctx.beginPath();
  for (const line of coast.lines) {
    ctx.moveTo(geo.x(line[0][0]), geo.y(line[0][1]));
    for (let k = 1; k < line.length; k++) ctx.lineTo(geo.x(line[k][0]), geo.y(line[k][1]));
  }
  ctx.stroke();
};

G.drawColorbar = function (canvas, limit, label, width) {
  const h = 42, dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr; canvas.height = h * dpr; canvas.style.width = width + "px"; canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, h);
  const x0 = 30, x1 = width - 30;
  for (let x = x0; x < x1; x++) {
    const c = G.RDBU[Math.floor((x - x0) / (x1 - x0) * 256)];
    ctx.fillStyle = "rgb(" + c + ")"; ctx.fillRect(x, 2, 1, 10);
  }
  ctx.strokeStyle = "#444"; ctx.strokeRect(x0, 2, x1 - x0, 10);
  ctx.fillStyle = "#222"; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
  for (const t of [-1, -0.5, 0, 0.5, 1]) {
    const x = x0 + (t + 1) / 2 * (x1 - x0);
    ctx.fillRect(x, 12, 1, 3);
    ctx.fillText(+(t * limit).toPrecision(3), x, 24);
  }
  ctx.fillText(label, width / 2, 39);
};

// grid index under the mouse (or -1)
G.pick = function (geo, grid, ev) {
  const r = ev.target.getBoundingClientRect();
  const lo = geo.west + (ev.clientX - r.left) / geo.width * (geo.east - geo.west);
  const la = geo.north - (ev.clientY - r.top) / geo.height * (geo.north - geo.south);
  const i = Math.round((lo - grid.lon[0]) / geo.dlon), j = Math.round((la - grid.lat[0]) / geo.dlat);
  if (i < 0 || j < 0 || i >= grid.lon.length || j >= grid.lat.length) return null;
  return {index: j * grid.lon.length + i, lat: grid.lat[j], lon: grid.lon[i]};
};

// page state in the URL (#key=value&...), so a view can be bookmarked or sent
G.readHash = function () {
  const out = {};
  for (const part of location.hash.replace(/^#/, "").split("&")) {
    if (!part) continue;
    const [k, v] = part.split("=");
    out[decodeURIComponent(k)] = decodeURIComponent(v || "");
  }
  return out;
};
G.writeHash = function (state) {
  const s = Object.entries(state).filter(([, v]) => v !== undefined && v !== "").map(([k, v]) => encodeURIComponent(k) + "=" + encodeURIComponent(v)).join("&");
  history.replaceState(null, "", "#" + s);
};

G.select = function (el, options, value) {
  el.innerHTML = "";
  for (const o of options) {
    const opt = document.createElement("option");
    if (Array.isArray(o)) { opt.value = o[0]; opt.textContent = o[1]; } else { opt.value = opt.textContent = o; }
    el.appendChild(opt);
  }
  const values = options.map(o => Array.isArray(o) ? o[0] : o);
  el.value = values.includes(value) ? value : values[0];
  return el.value;
};

//---------------line charts (SVG) for the index pages--------------//
G.COLORS = {analogue: "#1f77b4", cmip6_all: "#000000", ar1_fair: "#ff7f0e", ar1_all: "#d62728", ar1_rm_fair: "#9467bd",
            sine_trend: "#2ca02c", persistence: "#8c564b", null_p95: "#7f7f7f"};
G.PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"];

G.niceTicks = function (lo, hi, n = 5) {
  const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
};

// series: [{label, color, dash, width, y: [..], lo: [..], hi: [..]}]; x: numbers or category labels
G.lineChart = function (el, opts) {
  const cat = typeof opts.x[0] === "string";
  const W = opts.width || 480, H = opts.height || 300, m = {l: 46, r: 10, t: opts.title ? 22 : 8, b: cat ? 58 : 40};
  const xs = cat ? opts.x.map((_, i) => i) : opts.x;
  const x0 = cat ? -0.5 : Math.min(...xs), x1 = cat ? xs.length - 0.5 : Math.max(...xs);
  const [y0, y1] = opts.ylim;
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r), Y = v => m.t + (y1 - v) / (y1 - y0) * (H - m.t - m.b);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("width", W); svg.setAttribute("height", H); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const add = (tag, attrs, text) => {
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    svg.appendChild(e); return e;
  };
  add("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "#fff", stroke: "#bbb"});
  for (const t of G.niceTicks(y0, y1)) {
    add("line", {x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: t === 0 ? "#888" : "#eee"});
    add("text", {x: m.l - 4, y: Y(t) + 3, "text-anchor": "end", "font-size": 10, fill: "#333"}, +t.toPrecision(3));
  }
  const xticks = cat ? xs : G.niceTicks(x0, x1, 8);
  xticks.forEach((t, i) => {
    const lab = cat ? opts.x[i] : t;
    const tx = add("text", {x: X(t), y: H - m.b + 13, "text-anchor": cat ? "end" : "middle", "font-size": 10, fill: "#333"}, lab);
    if (cat) tx.setAttribute("transform", `rotate(-40 ${X(t)} ${H - m.b + 13})`);
  });
  if (opts.xlabel) add("text", {x: (m.l + W - m.r) / 2, y: H - 3, "text-anchor": "middle", "font-size": 11, fill: "#333"}, opts.xlabel);
  if (opts.ylabel) add("text", {x: 11, y: (m.t + H - m.b) / 2, "text-anchor": "middle", "font-size": 11, fill: "#333", transform: `rotate(-90 11 ${(m.t + H - m.b) / 2})`}, opts.ylabel);
  if (opts.title) add("text", {x: m.l, y: 15, "font-size": 12, "font-weight": 600, fill: "#222"}, opts.title);
  for (const v of opts.vlines || []) if (!cat && v >= x0 && v <= x1) add("line", {x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: "#999", "stroke-dasharray": "2 3"});
  const clip = v => Math.max(y0, Math.min(y1, v));
  for (const s of opts.series) {
    if (s.lo && s.hi) {   // band, one polygon per run of valid points
      let run = [];
      const flush = () => {
        if (run.length > 1) {
          const pts = run.map(i => `${X(xs[i])},${Y(clip(s.hi[i]))}`).concat(run.slice().reverse().map(i => `${X(xs[i])},${Y(clip(s.lo[i]))}`));
          add("polygon", {points: pts.join(" "), fill: s.color, "fill-opacity": 0.13, stroke: "none"});
        }
        run = [];
      };
      xs.forEach((_, i) => (s.lo[i] === null || s.hi[i] === null) ? flush() : run.push(i));
      flush();
    }
  }
  for (const s of opts.series) {
    let d = "", pen = false;
    xs.forEach((x, i) => {
      const v = s.y[i];
      if (v === null || v === undefined || Number.isNaN(v)) { pen = false; return; }
      d += (pen ? "L" : "M") + X(x).toFixed(1) + "," + Y(clip(v)).toFixed(1); pen = true;
    });
    add("path", {d, fill: "none", stroke: s.color, "stroke-width": s.width || 1.6, "stroke-dasharray": s.dash || ""});
    if (cat) xs.forEach((x, i) => { const v = s.y[i]; if (v !== null && v !== undefined) add("circle", {cx: X(x), cy: Y(clip(v)), r: 2.4, fill: s.color}); });
  }
  el.appendChild(svg);
  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = opts.series.map(s => `<span><svg width="22" height="8"><line x1="0" x2="22" y1="4" y2="4" stroke="${s.color}" stroke-width="2" stroke-dasharray="${s.dash || ""}"/></svg>${s.label}</span>`).join(" ");
  el.appendChild(legend);
};

// stacked areas (SVG): layers [{label, color, y: [..]}] from the bottom, numeric x; gaps where any layer is missing
G.stackChart = function (el, opts) {
  const W = opts.width || 480, H = opts.height || 300, m = {l: 46, r: 10, t: 8, b: 40};
  const xs = opts.x, x0 = opts.xlim ? opts.xlim[0] : Math.min(...xs), x1 = opts.xlim ? opts.xlim[1] : Math.max(...xs);
  const [y0, y1] = opts.ylim || [0, 1];
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r), Y = v => m.t + (y1 - v) / (y1 - y0) * (H - m.t - m.b);
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
  svg.setAttribute("width", W); svg.setAttribute("height", H); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const add = (tag, attrs, text) => {
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    svg.appendChild(e); return e;
  };
  add("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "#fff", stroke: "#bbb"});
  const ok = i => xs[i] >= x0 && xs[i] <= x1 && opts.layers.every(l => l.y[i] !== null && l.y[i] !== undefined && !Number.isNaN(l.y[i]));
  const base = xs.map(() => 0);
  for (const layer of opts.layers) {
    const top = base.map((b, i) => b + (ok(i) ? layer.y[i] : 0));
    let run = [];
    const flush = () => {
      if (run.length > 1) {
        const pts = run.map(i => `${X(xs[i])},${Y(Math.min(y1, top[i]))}`).concat(run.slice().reverse().map(i => `${X(xs[i])},${Y(Math.min(y1, base[i]))}`));
        add("polygon", {points: pts.join(" "), fill: layer.color, "fill-opacity": 0.85, stroke: "none"}).appendChild(document.createElementNS(ns, "title")).textContent = layer.label;
      }
      run = [];
    };
    xs.forEach((_, i) => ok(i) ? run.push(i) : flush());
    flush();
    top.forEach((v, i) => { base[i] = v; });
  }
  for (const t of G.niceTicks(y0, y1)) {
    add("line", {x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: "#fff", "stroke-opacity": 0.5});
    add("text", {x: m.l - 4, y: Y(t) + 3, "text-anchor": "end", "font-size": 10, fill: "#333"}, +t.toPrecision(3));
  }
  for (const t of G.niceTicks(x0, x1, 8)) add("text", {x: X(t), y: H - m.b + 13, "text-anchor": "middle", "font-size": 10, fill: "#333"}, t);
  for (const v of opts.vlines || []) if (v >= x0 && v <= x1) add("line", {x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: "#222", "stroke-dasharray": "2 3"});
  if (opts.xlabel) add("text", {x: (m.l + W - m.r) / 2, y: H - 3, "text-anchor": "middle", "font-size": 11, fill: "#333"}, opts.xlabel);
  if (opts.ylabel) add("text", {x: 11, y: (m.t + H - m.b) / 2, "text-anchor": "middle", "font-size": 11, fill: "#333", transform: `rotate(-90 11 ${(m.t + H - m.b) / 2})`}, opts.ylabel);
  el.appendChild(svg);
  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = opts.layers.map(l => `<span><svg width="14" height="10"><rect width="14" height="10" fill="${l.color}" fill-opacity="0.85"/></svg>${l.label}</span>`).join(" ");
  el.appendChild(legend);
};

// heat map of a (rows x cols) array with NaN, rows from the bottom (e.g. init years), colour limit symmetric
G.heatmap = function (canvas, values, nrow, ncol, limit, width, height) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr; canvas.height = height * dpr; canvas.style.width = width + "px"; canvas.style.height = height + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#d9d9d9"; ctx.fillRect(0, 0, width, height);
  const img = new ImageData(ncol, nrow);
  for (let r = 0; r < nrow; r++) for (let c = 0; c < ncol; c++) {
    const v = values[r * ncol + c], p = 4 * ((nrow - 1 - r) * ncol + c);
    if (v === null || Number.isNaN(v)) continue;
    const k = Math.max(0, Math.min(255, Math.floor((v + limit) / (2 * limit) * 256))), col = G.RDBU[k];
    img.data[p] = col[0]; img.data[p + 1] = col[1]; img.data[p + 2] = col[2]; img.data[p + 3] = 255;
  }
  const off = document.createElement("canvas"); off.width = ncol; off.height = nrow;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, width, height);
};

// a pasted or edited address (#...) in an open tab: reload with the new view (writeHash uses replaceState, no event)
window.addEventListener("hashchange", () => location.reload());
