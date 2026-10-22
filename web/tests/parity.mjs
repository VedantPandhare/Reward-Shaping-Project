// JS policy forward pass must pick the same action as SB3 on recorded observations.
import { readFileSync } from "node:fs";
import { act } from "../src/policy.js";

const pol = JSON.parse(readFileSync(new URL("../models/policies.json", import.meta.url)));
const par = JSON.parse(readFileSync(new URL("../models/parity.json", import.meta.url)));
let bad = 0, n = 0;
for (const [variant, samples] of Object.entries(par)) {
  for (const s of samples) { n++; if (act(pol[variant], s.obs) !== s.action) bad++; }
}
console.log(`parity: ${n - bad}/${n} actions match SB3`);
if (bad > 0) process.exit(1);
