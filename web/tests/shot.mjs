// Headless-browser smoke test: serves web/, loads the page, logs console errors, saves screenshots.
//   node tests/shot.mjs [outDir] [seconds,...] [leftPilot] [rightPilot] [speed]
import http from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const root = fileURLToPath(new URL("..", import.meta.url));
const out = process.argv[2] ?? "tests/shots";
const times = (process.argv[3] ?? "30,200").split(",").map(Number);
const left = process.argv[4], right = process.argv[5], speed = process.argv[6];
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".png": "image/png" };
const server = http.createServer(async (req, res) => {
  try {
    const p = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^[\\/]+/, "");
    const data = await readFile(join(root, p || "index.html"));
    res.writeHead(200, { "content-type": types[extname(p)] ?? "application/octet-stream" });
    res.end(data);
  } catch { res.writeHead(404); res.end("nf"); }
}).listen(0);
const port = server.address().port;
await mkdir(out, { recursive: true });

const chrome = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"];
const { existsSync } = await import("node:fs");
const browser = await puppeteer.launch({
  executablePath: chrome.find(existsSync), headless: true,
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--no-sandbox"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 1000 });
const errors = [];
page.on("console", (m) => { if (["error", "warning"].includes(m.type())) errors.push(`[${m.type()}] ${m.text()}`); });
page.on("pageerror", (e) => errors.push(`[pageerror] ${e.message}`));
await page.goto(`http://localhost:${port}/index.html`, { waitUntil: "networkidle0" });
if (left) await page.select("#sel-left", left);
if (right) await page.select("#sel-right", right);
if (speed) await page.select("#sel-speed", speed);
let t0 = Date.now();
// times are simulation frames: advance deterministically, let the renderer catch up, then shoot
let done = 0;
for (const f of times) {
  await page.evaluate((n) => window.__lab.advance(n), f - done);
  done = f;
  await new Promise((r) => setTimeout(r, 900));
  await page.screenshot({ path: join(out, `f${f}.png`) });
  console.log("shot frame", f);
}
console.log(errors.length ? errors.slice(0, 15).join("\n") : "no console errors/warnings");
await browser.close();
server.close();
