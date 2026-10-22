// Interaction test: keyboard thrust changes the human lander; Results modal renders.
import http from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import puppeteer from "puppeteer-core";

const root = fileURLToPath(new URL("..", import.meta.url));
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".png": "image/png" };
const server = http.createServer(async (req, res) => {
  try {
    const p = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^[\/]+/, "");
    res.writeHead(200, { "content-type": types[extname(p)] ?? "application/octet-stream" });
    res.end(await readFile(join(root, p || "index.html")));
  } catch { res.writeHead(404); res.end(); }
}).listen(0);
const chrome = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].find(existsSync);
const browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox"] });
const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 1000 });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
await page.goto(`http://localhost:${server.address().port}/index.html`, { waitUntil: "networkidle0" });
await page.select("#sel-left", "human");
const fuel = () => page.evaluate(() => window.__lab.state.envs[0].returns.fuel);
await page.evaluate(() => window.__lab.advance(10));
const f0 = await fuel();
await page.keyboard.down("ArrowUp");
await page.evaluate(() => window.__lab.advance(30));
await page.keyboard.up("ArrowUp");
const f1 = await fuel();
await page.keyboard.down("a");
await page.evaluate(() => window.__lab.advance(20));
await page.keyboard.up("a");
const f2 = await fuel();
console.log(`fuel after idle ${f0.toFixed(2)}, +30 frames main ${f1.toFixed(2)} (expect +9.0), +20 frames left ${f2.toFixed(2)} (expect +0.6)`);
await page.click("#btn-results");
await new Promise((r) => setTimeout(r, 500));
await page.screenshot({ path: "tests/shots/results.png" });
console.log(errors.length ? errors.join("\n") : "no errors");
await browser.close(); server.close();
