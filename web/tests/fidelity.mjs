// Run exported policies in the JS physics port; compare outcome rates with the Python results.
import { readFileSync } from "node:fs";
import { LunarLanderEnv } from "../src/lander_env.js";
import { act } from "../src/policy.js";

const pol = JSON.parse(readFileSync(new URL("../models/policies.json", import.meta.url)));
const episodes = Number(process.argv[2] ?? 40);
const wind = process.argv[4] ? { windPower: Number(process.argv[4]), turbulencePower: Number(process.argv[5] ?? 0) } : {};
const variants = process.argv[3] ? process.argv[3].split(",") : ["baseline", "hover_hack", "delta_fix"];
for (const v of variants) {
  const c = { landed: 0, crashed: 0, timeout: 0 };
  let ret = 0, len = 0;
  for (let ep = 0; ep < episodes; ep++) {
    const env = new LunarLanderEnv({ variant: v, ...wind });
    let obs = env.reset(10000 + ep);
    let r;
    do { r = env.step(act(pol[v], obs)); obs = r.obs; } while (!env.done);
    c[r.outcome]++; ret += env.returns.raw; len += env.t;
  }
  console.log(v.padEnd(16), `landed ${(c.landed / episodes).toFixed(2)} crashed ${(c.crashed / episodes).toFixed(2)}`,
    `timeout ${(c.timeout / episodes).toFixed(2)} return ${(ret / episodes).toFixed(0)} len ${(len / episodes).toFixed(0)}`);
}
