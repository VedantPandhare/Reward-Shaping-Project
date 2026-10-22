import { LunarLanderEnv } from "../src/lander_env.js";
const pilot = (s) => { const at=Math.max(-0.4,Math.min(0.4,s[0]*0.5+s[2])); const ht=0.55*Math.abs(s[0]);
  let a=(at-s[4])*0.5-s[5], h=(ht-s[1])*0.5-s[3]*0.5; if(s[6]||s[7]){a=0;h=-s[3]*0.5;}
  if(h>Math.abs(a)&&h>0.05)return 2; if(a<-0.05)return 3; if(a>0.05)return 1; return 0; };
function run(cfg,d,h,n=24){ let L=0,C=0,T=0; for(let ep=0;ep<n;ep++){ const env=new LunarLanderEnv(cfg); let o=env.reset(20000+ep); const hist=[o]; let a=0,t=0,r;
  do{ if(t%h===0)a=pilot(hist[Math.max(0,hist.length-1-d)]); r=env.step(a); hist.push(r.obs); t++; }while(!env.done);
  if(r.outcome==="landed")L++; else if(r.outcome==="crashed")C++; else T++; } return `L${(L/n).toFixed(2)} C${(C/n).toFixed(2)} T${(T/n).toFixed(2)}`; }
for (const [name,cfg,d,h] of [
  ["easy -4 @0.5x (delay4 hold2)",{gravity:-4,assist:0.9},4,2],
  ["easy -4 @1x (delay8 hold3)",{gravity:-4,assist:0.9},8,3],
  ["medium -6 @1x",{gravity:-6,assist:0.9},8,3],
  ["medium -6 @0.75x (delay6 hold2)",{gravity:-6,assist:0.9},6,2],
  ["medium -10 @0.75x",{gravity:-10,assist:0.9},6,2],
  ["real -10 @0.5x (delay4 hold2)",{gravity:-10,assist:0},4,2],
]) console.log(name.padEnd(36), run(cfg,d,h));
