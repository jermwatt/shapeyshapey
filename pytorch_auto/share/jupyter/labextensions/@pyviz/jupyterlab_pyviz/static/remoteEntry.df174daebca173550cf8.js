var _JUPYTERLAB;(()=>{"use strict";var e,r,t,n,o,a,i,u,l,s,f,p,d,c,v,h,y,b,g,m,j,w={281:(e,r,t)=>{var n={"./index":()=>t.e(445).then((()=>()=>t(445))),"./extension":()=>t.e(445).then((()=>()=>t(445))),"./style":()=>t.e(747).then((()=>()=>t(747)))},o=(e,r)=>(t.R=r,r=t.o(n,e)?n[e]():Promise.resolve().then((()=>{throw new Error('Module "'+e+'" does not exist in container.')})),t.R=void 0,r),a=(e,r)=>{if(t.S){var n="default",o=t.S[n];if(o&&o!==e)throw new Error("Container initialization failed as it has already been initialized with a different share scope");return t.S[n]=e,t.I(n,r)}};t.d(r,{get:()=>o,init:()=>a})}},S={};function _(e){var r=S[e];if(void 0!==r)return r.exports;var t=S[e]={id:e,exports:{}};return w[e](t,t.exports,_),t.exports}_.m=w,_.c=S,_.n=e=>{var r=e&&e.__esModule?()=>e.default:()=>e;return _.d(r,{a:r}),r},_.d=(e,r)=>{for(var t in r)_.o(r,t)&&!_.o(e,t)&&Object.defineProperty(e,t,{enumerable:!0,get:r[t]})},_.f={},_.e=e=>Promise.all(Object.keys(_.f).reduce(((r,t)=>(_.f[t](e,r),r)),[])),_.u=e=>e+"."+{445:"e0d338852f1bcbe1aa57",747:"872484f6c86b06b5c3f4"}[e]+".js?v="+{445:"e0d338852f1bcbe1aa57",747:"872484f6c86b06b5c3f4"}[e],_.g=function(){if("object"==typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"==typeof window)return window}}(),_.o=(e,r)=>Object.prototype.hasOwnProperty.call(e,r),e={},r="@pyviz/jupyterlab_pyviz:",_.l=(t,n,o,a)=>{if(e[t])e[t].push(n);else{var i,u;if(void 0!==o)for(var l=document.getElementsByTagName("script"),s=0;s<l.length;s++){var f=l[s];if(f.getAttribute("src")==t||f.getAttribute("data-webpack")==r+o){i=f;break}}i||(u=!0,(i=document.createElement("script")).charset="utf-8",i.timeout=120,_.nc&&i.setAttribute("nonce",_.nc),i.setAttribute("data-webpack",r+o),i.src=t),e[t]=[n];var p=(r,n)=>{i.onerror=i.onload=null,clearTimeout(d);var o=e[t];if(delete e[t],i.parentNode&&i.parentNode.removeChild(i),o&&o.forEach((e=>e(n))),r)return r(n)},d=setTimeout(p.bind(null,void 0,{type:"timeout",target:i}),12e4);i.onerror=p.bind(null,i.onerror),i.onload=p.bind(null,i.onload),u&&document.head.appendChild(i)}},_.r=e=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},(()=>{_.S={};var e={},r={};_.I=(t,n)=>{n||(n=[]);var o=r[t];if(o||(o=r[t]={}),!(n.indexOf(o)>=0)){if(n.push(o),e[t])return e[t];_.o(_.S,t)||(_.S[t]={});var a=_.S[t],i="@pyviz/jupyterlab_pyviz",u=[];return"default"===t&&((e,r,t,n)=>{var o=a[e]=a[e]||{},u=o[r];(!u||!u.loaded&&(1!=!u.eager?n:i>u.from))&&(o[r]={get:()=>_.e(445).then((()=>()=>_(445))),from:i,eager:!1})})("@pyviz/jupyterlab_pyviz","2.2.1"),e[t]=u.length?Promise.all(u).then((()=>e[t]=1)):1}}})(),(()=>{var e;_.g.importScripts&&(e=_.g.location+"");var r=_.g.document;if(!e&&r&&(r.currentScript&&(e=r.currentScript.src),!e)){var t=r.getElementsByTagName("script");t.length&&(e=t[t.length-1].src)}if(!e)throw new Error("Automatic publicPath is not supported in this browser");e=e.replace(/#.*$/,"").replace(/\?.*$/,"").replace(/\/[^\/]+$/,"/"),_.p=e})(),t=e=>{var r=e=>e.split(".").map((e=>+e==e?+e:e)),t=/^([^-+]+)?(?:-([^+]+))?(?:\+(.+))?$/.exec(e),n=t[1]?r(t[1]):[];return t[2]&&(n.length++,n.push.apply(n,r(t[2]))),t[3]&&(n.push([]),n.push.apply(n,r(t[3]))),n},n=(e,r)=>{e=t(e),r=t(r);for(var n=0;;){if(n>=e.length)return n<r.length&&"u"!=(typeof r[n])[0];var o=e[n],a=(typeof o)[0];if(n>=r.length)return"u"==a;var i=r[n],u=(typeof i)[0];if(a!=u)return"o"==a&&"n"==u||"s"==u||"u"==a;if("o"!=a&&"u"!=a&&o!=i)return o<i;n++}},o=e=>{var r=e[0],t="";if(1===e.length)return"*";if(r+.5){t+=0==r?">=":-1==r?"<":1==r?"^":2==r?"~":r>0?"=":"!=";for(var n=1,a=1;a<e.length;a++)n--,t+="u"==(typeof(u=e[a]))[0]?"-":(n>0?".":"")+(n=2,u);return t}var i=[];for(a=1;a<e.length;a++){var u=e[a];i.push(0===u?"not("+l()+")":1===u?"("+l()+" || "+l()+")":2===u?i.pop()+" "+i.pop():o(u))}return l();function l(){return i.pop().replace(/^\((.+)\)$/,"$1")}},a=(e,r)=>{if(0 in e){r=t(r);var n=e[0],o=n<0;o&&(n=-n-1);for(var i=0,u=1,l=!0;;u++,i++){var s,f,p=u<e.length?(typeof e[u])[0]:"";if(i>=r.length||"o"==(f=(typeof(s=r[i]))[0]))return!l||("u"==p?u>n&&!o:""==p!=o);if("u"==f){if(!l||"u"!=p)return!1}else if(l)if(p==f)if(u<=n){if(s!=e[u])return!1}else{if(o?s>e[u]:s<e[u])return!1;s!=e[u]&&(l=!1)}else if("s"!=p&&"n"!=p){if(o||u<=n)return!1;l=!1,u--}else{if(u<=n||f<p!=o)return!1;l=!1}else"s"!=p&&"n"!=p&&(l=!1,u--)}}var d=[],c=d.pop.bind(d);for(i=1;i<e.length;i++){var v=e[i];d.push(1==v?c()|c():2==v?c()&c():v?a(v,r):!c())}return!!c()},i=(e,r)=>{var t=_.S[e];if(!t||!_.o(t,r))throw new Error("Shared module "+r+" doesn't exist in shared scope "+e);return t},u=(e,r)=>{var t=e[r];return(r=Object.keys(t).reduce(((e,r)=>!e||n(e,r)?r:e),0))&&t[r]},l=(e,r)=>{var t=e[r];return Object.keys(t).reduce(((e,r)=>!e||!t[e].loaded&&n(e,r)?r:e),0)},s=(e,r,t,n)=>"Unsatisfied version "+t+" from "+(t&&e[r][t].from)+" of shared singleton module "+r+" (required "+o(n)+")",f=(e,r,t,n)=>{var o=l(e,t);return a(n,o)||"undefined"!=typeof console&&console.warn&&console.warn(s(e,t,o,n)),v(e[t][o])},p=(e,r,t)=>{var o=e[r];return(r=Object.keys(o).reduce(((e,r)=>!a(t,r)||e&&!n(e,r)?e:r),0))&&o[r]},d=(e,r,t,n)=>{var a=e[t];return"No satisfying version ("+o(n)+") of shared module "+t+" found in shared scope "+r+".\nAvailable versions: "+Object.keys(a).map((e=>e+" from "+a[e].from)).join(", ")},c=(e,r,t,n)=>{"undefined"!=typeof console&&console.warn&&console.warn(d(e,r,t,n))},v=e=>(e.loaded=1,e.get()),y=(h=e=>function(r,t,n,o){var a=_.I(r);return a&&a.then?a.then(e.bind(e,r,_.S[r],t,n,o)):e(r,_.S[r],t,n,o)})(((e,r,t,n)=>(i(e,t),v(p(r,t,n)||c(r,e,t,n)||u(r,t))))),b=h(((e,r,t,n)=>(i(e,t),f(r,0,t,n)))),g={},m={19:()=>b("default","@jupyterlab/coreutils",[1,5,4,5]),46:()=>y("default","@jupyterlab/docregistry",[1,3,4,5]),74:()=>b("default","@jupyterlab/mainmenu",[1,3,4,5]),271:()=>b("default","react",[1,17,0,1]),292:()=>b("default","@jupyter-widgets/jupyterlab-manager",[1,3,0,0]),484:()=>b("default","@jupyterlab/docmanager",[1,3,4,5]),493:()=>b("default","@jupyterlab/application",[1,3,4,5]),526:()=>b("default","@lumino/coreutils",[1,1,11,0]),840:()=>b("default","@lumino/signaling",[1,1,10,0]),912:()=>b("default","@jupyterlab/ui-components",[1,3,4,5]),921:()=>b("default","@jupyterlab/settingregistry",[1,3,4,5]),923:()=>b("default","@lumino/disposable",[1,1,10,0]),972:()=>b("default","@jupyterlab/notebook",[1,3,4,5]),991:()=>b("default","@jupyterlab/apputils",[1,3,4,5]),992:()=>b("default","@lumino/widgets",[1,1,33,0])},j={445:[19,46,74,271,292,484,493,526,840,912,921,923,972,991,992]},_.f.consumes=(e,r)=>{_.o(j,e)&&j[e].forEach((e=>{if(_.o(g,e))return r.push(g[e]);var t=r=>{g[e]=0,_.m[e]=t=>{delete _.c[e],t.exports=r()}},n=r=>{delete g[e],_.m[e]=t=>{throw delete _.c[e],r}};try{var o=m[e]();o.then?r.push(g[e]=o.then(t).catch(n)):t(o)}catch(e){n(e)}}))},(()=>{var e={924:0};_.f.j=(r,t)=>{var n=_.o(e,r)?e[r]:void 0;if(0!==n)if(n)t.push(n[2]);else{var o=new Promise(((t,o)=>n=e[r]=[t,o]));t.push(n[2]=o);var a=_.p+_.u(r),i=new Error;_.l(a,(t=>{if(_.o(e,r)&&(0!==(n=e[r])&&(e[r]=void 0),n)){var o=t&&("load"===t.type?"missing":t.type),a=t&&t.target&&t.target.src;i.message="Loading chunk "+r+" failed.\n("+o+": "+a+")",i.name="ChunkLoadError",i.type=o,i.request=a,n[1](i)}}),"chunk-"+r,r)}};var r=(r,t)=>{var n,o,[a,i,u]=t,l=0;if(a.some((r=>0!==e[r]))){for(n in i)_.o(i,n)&&(_.m[n]=i[n]);u&&u(_)}for(r&&r(t);l<a.length;l++)o=a[l],_.o(e,o)&&e[o]&&e[o][0](),e[o]=0},t=self.webpackChunk_pyviz_jupyterlab_pyviz=self.webpackChunk_pyviz_jupyterlab_pyviz||[];t.forEach(r.bind(null,0)),t.push=r.bind(null,t.push.bind(t))})(),_.nc=void 0;var k=_(281);(_JUPYTERLAB=void 0===_JUPYTERLAB?{}:_JUPYTERLAB)["@pyviz/jupyterlab_pyviz"]=k})();