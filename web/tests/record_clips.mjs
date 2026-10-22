// Records deterministic 3D clips of the simulator (frame-by-frame, independent of real-time speed).
//   FFMPEG=path node tests/record_clips.mjs outDir [maxSim] [name1,name2 ...]
import http from "node:http";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const root = fileURLToPath(new URL("..", import.meta.url));
const out = process.argv[2] ?? "media";
const maxSim = Number(process.argv[3] ?? 700);
const only = process.argv[4]?.split(",");
const ffmpeg = process.env.FFMPEG ?? "ffmpeg";
const CLIPS = [
  { name: "hack_vs_fix", left: "delta_fix", right: "hover_hack", seed: 1 },
  { name: "baseline_vs_hurry", left: "baseline", right: "time_penalty", seed: 1 },
  { name: "improvement_variants", left: "delta_no_time", right: "tilt_only_hack", seed: 1 },
  { name: "tradeoffs", left: "fuel_efficient", right: "soft_landing", seed: 1 },
].filter((c) => !only || only.includes(c.name));

const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".png": "image/png" };
const server = http.createServer(async (req, res) => {
  try {
    const p = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^[\\/]+/, "");
    res.writeHead(200, { "content-type": types[extname(p)] ?? "application/octet-stream" });
    res.end(await readFile(join(root, p || "index.html")));
  } catch { res.writeHead(404); res.end(); }
}).listen(0);
const chrome = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].find(existsSync);
const browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox"] });
await mkdir(out, { recursive: true });

for (const clip of CLIPS) {
  const t0 = Date.now();
  const page = await browser.newPage();
  await page.setViewport({ width: 1320, height: 900 });
  await page.goto(`http://localhost:${server.address().port}/sim.html`, { waitUntil: "networkidle0" });
  await page.addStyleTag({ content: ".resultcard,.countdown,.rule{display:none !important} .panel-head{min-height:0 !important}" });
  await page.select("#sel-left", clip.left);
  await page.select("#sel-right", clip.right);
  await page.evaluate((seed) => { window.__lab.state.seed = seed; document.querySelector("#btn-restart").click(); }, clip.seed);
  await page.evaluate(() => { window.__lab.manual = true; window.__lab.fixedDt = 0.04; });
  const rect = await page.evaluate(() => {
    const els = [...document.querySelectorAll(".panel-head,.viewport")].map((e) => e.getBoundingClientRect());
    const x = Math.min(...els.map((r) => r.left)), y = Math.min(...els.map((r) => r.top));
    return { x, y, width: Math.max(...els.map((r) => r.right)) - x, height: Math.max(...els.map((r) => r.bottom)) - y };
  });
  const dir = join(out, `_${clip.name}`);
  await rm(dir, { recursive: true, force: true });
  await mkdir(dir, { recursive: true });
  let n = 0, hold = 0, sim = 0;
  for (;;) {
    const finished = await page.evaluate(() => window.__lab.state.done.every(Boolean));
    if (!finished && sim < maxSim) { await page.evaluate(() => window.__lab.advance(2)); sim += 2; }
    else if (++hold > 45) break;
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(r)));
    await writeFile(join(dir, `f${String(n++).padStart(4, "0")}.png`), await page.screenshot({ clip: rect }));
  }
  const mp4 = join(out, `${clip.name}.mp4`);
  execFileSync(ffmpeg, ["-y", "-loglevel", "error", "-framerate", "25", "-i", join(dir, "f%04d.png"),
    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27", "-preset", "slow", "-movflags", "+faststart", mp4]);
  await writeFile(join(out, `${clip.name}.jpg`), await page.screenshot({ clip: rect, type: "jpeg", quality: 82 }));
  await rm(dir, { recursive: true, force: true });
  console.log(`${clip.name}: ${n} frames in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  await page.close();
}
await browser.close();
server.close();
