module.exports={content:["./app/**/*.tsx","./components/**/*.tsx"],theme:{extend:{
colors:{bg:"#07090c",panel:"#0e1116",raised:"#141920",line:"#1f252e",accent:"#4f9dff",risk:"#f0504a",ok:"#35c46a",warn:"#f2a93b"},
fontFamily:{sans:["var(--font-sans)","system-ui","sans-serif"],mono:["var(--font-mono)","ui-monospace","monospace"]},
keyframes:{rise:{from:{opacity:0,transform:"translateY(6px)"},to:{opacity:1,transform:"none"}},slideIn:{from:{opacity:0,transform:"translateX(28px)"},to:{opacity:1,transform:"none"}},shimmer:{to:{backgroundPosition:"-200% 0"}}},
animation:{"slide-in":"slideIn .5s cubic-bezier(.2,.7,.2,1) both",rise:"rise .35s ease-out both",shimmer:"shimmer 1.6s linear infinite"}}},plugins:[]}
