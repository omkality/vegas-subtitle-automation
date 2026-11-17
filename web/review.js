let data, states=new Map(), dirty=false, page=0, saveTimer, revision=0, saveChain=Promise.resolve();
const size=20;
const el=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e};
const status=document.querySelector('#status');
function serialize(){return {schema:2,dataset_sha256:data.dataset_sha256,decisions:[...states.values()]}}
function initialState(r){
  const previous=r.decision||{};
  const state={id:r.id,text:r.text,speaker:r.speaker,trim_start:r.trim_start,trim_end:r.trim_end,
    alignment_reviewed:false,training_approved:false,rejected:false,coverage:'full',
    text_reviewed:!!previous.training_approved,...previous};
  if(!state.coverage||state.coverage==='unreviewed')state.coverage='full';
  return state;
}
function approve(s){s.training_approved=!!(s.text_reviewed&&s.alignment_reviewed&&s.coverage==='full'&&!s.rejected)}
function changed(){revision++;dirty=true;status.textContent='Сохраняю…';clearTimeout(saveTimer);saveTimer=setTimeout(save,800);update()}
async function save(){
  clearTimeout(saveTimer);
  const savedRevision=revision, body=JSON.stringify(serialize());
  const operation=saveChain.catch(()=>{}).then(async()=>{const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body});const result=await r.json();if(!r.ok)throw Error(result.error)});
  saveChain=operation;
  try{await operation;if(savedRevision===revision){dirty=false;status.textContent='Все изменения сохранены';status.classList.remove('error')}return true}
  catch(e){status.textContent='Не удалось сохранить: '+e.message;status.classList.add('error');return false}
}
function update(){
  const rows=[...states.values()], text=rows.filter(s=>s.text_reviewed).length, complete=rows.filter(s=>s.training_approved).length, rejected=rows.filter(s=>s.rejected).length;
  document.querySelector('#progress').textContent=`Текст: ${text}/${rows.length} · С границами: ${complete} · Отклонено: ${rejected}`;
}
function cardFor(r,index){
  const s=states.get(r.id), card=el('article'), badge=el('div');badge.className='badge';
  function paint(){approve(s);card.classList.toggle('approved',s.training_approved);card.classList.toggle('text-done',s.text_reviewed);card.classList.toggle('rejected',s.rejected);badge.textContent=s.rejected?'Пример отклонён':s.training_approved?'Текст и границы проверены':s.text_reviewed?'Текст проверен. Границы можно проверить позже.':'Ожидает проверки'}
  const meta=el('div',`${index+1}. ${r.project} · ${r.start.toFixed(2)}–${r.end.toFixed(2)} сек`);meta.className='meta';card.append(meta);
  const source=el('div','Исходный титр: '+r.text);source.className='source';card.append(source);
  const audio=el('audio');audio.controls=true;audio.preload='metadata';audio.src=`/audio/${r.id}.wav`;card.append(audio);
  const canvas=el('canvas');canvas.width=1000;canvas.height=140;canvas.setAttribute('aria-label','Звуковая волна. Перетащи голубые края или нажми для перехода.');card.append(canvas);
  const selection=el('div');selection.className='selection';card.append(selection);
  let peaks=[], rangePlaying=false, dragging=null, rangeTimer;
  const duration=r.context_duration;
  const x=t=>Math.max(0,Math.min(1000,t/duration*1000));
  function draw(){
    const c=canvas.getContext('2d');c.clearRect(0,0,1000,140);c.fillStyle='#183f69';c.fillRect(x(s.trim_start),0,x(s.trim_end)-x(s.trim_start),140);
    c.strokeStyle='#9eb6d2';c.lineWidth=1;c.beginPath();for(let i=0;i<peaks.length;i++){const h=peaks[i]*48;c.moveTo(i+0.5,65-h);c.lineTo(i+0.5,65+h)}c.stroke();
    for(const [time,label] of [[s.trim_start,'Начало'],[s.trim_end,'Конец']]){const pos=x(time);c.strokeStyle='#75c8ff';c.lineWidth=3;c.beginPath();c.moveTo(pos,0);c.lineTo(pos,118);c.stroke();c.fillStyle='#75c8ff';c.fillRect(pos-5,42,10,42);c.font='13px system-ui';c.fillText(label+' '+Number(time).toFixed(2),Math.min(875,Math.max(5,pos+8)),17)}
    const cursor=x(audio.currentTime||0);c.strokeStyle='#ffbd75';c.lineWidth=2;c.beginPath();c.moveTo(cursor,23);c.lineTo(cursor,118);c.stroke();
    c.fillStyle='#9bb0cd';c.font='12px system-ui';for(let i=0;i<=4;i++)c.fillText((duration*i/4).toFixed(2)+' с',Math.min(950,i*250+3),134);
    selection.textContent=`Выделено: ${Number(s.trim_start).toFixed(2)}–${Number(s.trim_end).toFixed(2)} с · позиция: ${(audio.currentTime||0).toFixed(2)} с`;
  }
  function boundary(which,value){
    audio.pause();rangePlaying=false;clearTimeout(rangeTimer);
    if(which==='trim_start')s.trim_start=Math.max(0,Math.min(value,Number(s.trim_end)-.04));
    else s.trim_end=Math.min(duration,Math.max(value,Number(s.trim_start)+.04));
    s.alignment_reviewed=false;aligned.checked=false;paint();draw();changed();
  }
  const mouseTime=e=>Math.max(0,Math.min(duration,(e.clientX-canvas.getBoundingClientRect().left)/canvas.getBoundingClientRect().width*duration));
  canvas.onpointerdown=e=>{const t=mouseTime(e),threshold=duration*16/canvas.getBoundingClientRect().width;const ds=Math.abs(t-s.trim_start),de=Math.abs(t-s.trim_end);if(Math.min(ds,de)<=threshold){dragging=ds<=de?'trim_start':'trim_end';canvas.setPointerCapture(e.pointerId)}else{rangePlaying=false;audio.currentTime=t;draw()}};
  canvas.onpointermove=e=>{if(dragging)boundary(dragging,mouseTime(e))};canvas.onpointerup=()=>dragging=null;canvas.onpointercancel=()=>dragging=null;
  audio.ontimeupdate=()=>{if(rangePlaying&&audio.currentTime>=s.trim_end){audio.pause();rangePlaying=false;audio.currentTime=s.trim_end}draw()};audio.addEventListener('pointerdown',()=>{rangePlaying=false;clearTimeout(rangeTimer)});audio.onpause=()=>clearTimeout(rangeTimer);
  fetch(audio.src).then(r=>r.arrayBuffer()).then(buffer=>{
    const v=new DataView(buffer);let offset=12,start=0,length=0;
    while(offset+8<=v.byteLength){const name=String.fromCharCode(...new Uint8Array(buffer,offset,4)),n=v.getUint32(offset+4,true);if(name==='data'){start=offset+8;length=n;break}offset+=8+n+(n%2)}
    const count=Math.floor(length/2);if(!count)throw Error('Нет PCM-данных');peaks=Array(1000).fill(0);
    for(let i=0;i<count;i++)peaks[Math.min(999,Math.floor(i/count*1000))]=Math.max(peaks[Math.min(999,Math.floor(i/count*1000))],Math.abs(v.getInt16(start+i*2,true))/32768);
    const maximum=Math.max(...peaks,.02);peaks=peaks.map(p=>p/maximum);draw();
  }).catch(()=>{selection.textContent='Не удалось загрузить волну. Аудиоплеер остаётся доступен.'});
  const buttons=el('div');buttons.className='row';
  for(const [label,action] of [
    ['▶ Прослушать выделение',async()=>{audio.pause();rangePlaying=true;audio.currentTime=s.trim_start;try{await audio.play();clearTimeout(rangeTimer);rangeTimer=setTimeout(()=>{if(rangePlaying){audio.pause();rangePlaying=false;audio.currentTime=s.trim_end;draw()}},Math.max(0,(s.trim_end-audio.currentTime)*1000))}catch(e){status.textContent=e.message}}],
    ['▶ Весь фрагмент',()=>{rangePlaying=false;clearTimeout(rangeTimer);audio.currentTime=0;audio.play().catch(e=>status.textContent=e.message)}],
    ['Начало здесь',()=>boundary('trim_start',audio.currentTime)],['Конец здесь',()=>boundary('trim_end',audio.currentTime)],
    ['Сбросить границы',()=>{s.trim_start=r.trim_start;s.trim_end=r.trim_end;s.alignment_reviewed=false;aligned.checked=false;paint();draw();changed()}]
  ]){const b=el('button',label);b.onclick=action;buttons.append(b)}card.append(buttons);
  const text=el('textarea');text.value=s.text;text.setAttribute('aria-label','Дословная расшифровка');text.oninput=()=>{s.text=text.value;s.text_reviewed=false;checked.checked=false;s.alignment_reviewed=false;aligned.checked=false;paint();changed()};card.append(text);
  const row=el('div');row.className='row';const speaker=el('select');for(const [value,label] of [['self','Я'],['friend','Товарищ'],['unknown','Не уверен']]){const option=el('option',label);option.value=value;speaker.append(option)}speaker.value=s.speaker;speaker.onchange=()=>{s.speaker=speaker.value;s.text_reviewed=false;checked.checked=false;paint();changed()};const sl=el('label','Говорящий');sl.append(speaker);row.append(sl);
  const coverage=el('select');for(const [value,label]of [['unreviewed','Ещё не оценивал'],['full','Все слова слышны в контексте'],['cut_start','Обрезано начало'],['cut_end','Обрезан конец'],['cut_both','Обрезаны оба края'],['unsure','Не уверен']]){const option=el('option',label);option.value=value;coverage.append(option)}coverage.value=s.coverage;coverage.onchange=()=>{s.coverage=coverage.value;if(s.coverage!=='full'){s.alignment_reviewed=false;aligned.checked=false}paint();changed()};const cl=el('label','Полнота звука');cl.append(coverage);row.append(cl);card.append(row);
  const checks=el('div');checks.className='checks';const checked=el('input');checked.type='checkbox';checked.checked=s.text_reviewed;
  checked.onchange=()=>{if(checked.checked&&(!s.text.trim()||s.text.includes('*')||s.speaker==='unknown')){checked.checked=false;alert('Впиши слышимые слова без звёздочек и выбери говорящего.');return}s.text_reviewed=checked.checked;if(s.text_reviewed){s.rejected=false;reject.checked=false}paint();changed()};const tl=el('label');tl.append(checked,el('span','Текст и говорящий проверены'));checks.append(tl);
  const aligned=el('input');aligned.type='checkbox';aligned.checked=s.alignment_reviewed;aligned.onchange=()=>{if(aligned.checked&&s.coverage!=='full'){aligned.checked=false;alert('Для подтверждения границ выбери «Все слова слышны в контексте».');return}s.alignment_reviewed=aligned.checked;paint();changed()};const al=el('label');al.append(aligned,el('span','Границы выделения проверены'));checks.append(al);
  const reject=el('input');reject.type='checkbox';reject.checked=s.rejected;reject.onchange=()=>{s.rejected=reject.checked;if(s.rejected){s.text_reviewed=false;checked.checked=false;s.alignment_reviewed=false;aligned.checked=false}paint();changed()};const rl=el('label');rl.append(reject,el('span','Отклонить пример'));checks.append(rl);card.append(checks,badge);
  const advanced=el('details');advanced.append(el('summary','Точные значения границ'));for(const [key,label]of [['trim_start','Начало'],['trim_end','Конец']]){const l=el('label',label);const n=el('input');n.type='number';n.min=0;n.max=duration;n.step=.01;n.value=Number(s[key]).toFixed(3);n.onchange=()=>boundary(key,Number(n.value));advanced.ontoggle=()=>advanced.querySelectorAll('input').forEach((v,i)=>v.value=Number(s[i?'trim_end':'trim_start']).toFixed(3));l.append(n);advanced.append(l)}card.append(advanced);
  if(Object.keys(r.prediction).length){const d=el('details');d.append(el('summary','Подсказки моделей'));d.append(el('pre',Object.entries(r.prediction).map(([k,v])=>`${k}: ${v}`).join('\n')));card.append(d)}
  const note=el('small',`ID: ${r.id} · контекст ${duration.toFixed(2)} с`);card.append(note);paint();draw();return card;
}
function render(){const area=document.querySelector('#items');area.replaceChildren();data.items.slice(page*size,(page+1)*size).forEach((r,i)=>area.append(cardFor(r,page*size+i)));const pages=Math.max(1,Math.ceil(data.items.length/size));document.querySelector('#page').textContent=`${page+1} / ${pages}`;document.querySelector('#page-number').value=page+1;document.querySelector('#page-number').max=pages;document.querySelector('#prev').disabled=page===0;for(const id of ['next','next-bottom'])document.querySelector('#'+id).disabled=(page+1)*size>=data.items.length;history.replaceState(null,'',`#batch=${data.dataset_sha256.slice(0,12)}&page=${page+1}`);update()}
async function goTo(next){if(!Number.isInteger(next)||next<0||next>=Math.ceil(data.items.length/size))return;if(dirty&&!await save())return;document.querySelectorAll('audio').forEach(a=>a.pause());page=next;render();window.scrollTo({top:0,behavior:'smooth'})}
async function turn(delta){await goTo(page+delta)}
document.querySelector('#save').onclick=save;document.querySelector('#prev').onclick=()=>turn(-1);document.querySelector('#next').onclick=document.querySelector('#next-bottom').onclick=()=>turn(1);
document.querySelector('#jump').onclick=()=>goTo(Number(document.querySelector('#page-number').value)-1);
document.querySelector('#resume').onclick=()=>{const index=data.items.findIndex(r=>{const s=states.get(r.id);return !s.training_approved&&!s.rejected});if(index>=0)goTo(Math.floor(index/size));else status.textContent='Все примеры проверены или отклонены.'};
document.querySelector('#backup').onclick=()=>{const blob=new Blob([JSON.stringify(serialize(),null,2)],{type:'application/json'});const a=el('a');a.href=URL.createObjectURL(blob);a.download='vegas-review.json';a.click();URL.revokeObjectURL(a.href)};
window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue=''}};
fetch('/api/data').then(r=>r.json()).then(d=>{data=d;document.querySelector('#title').textContent=d.title;for(const r of d.items)states.set(r.id,initialState(r));const location=new URLSearchParams(window.location.hash.slice(1));const requested=Number(location.get('page'))-1;const first=data.items.findIndex(r=>{const s=states.get(r.id);return !s.training_approved&&!s.rejected});page=location.get('batch')===d.dataset_sha256.slice(0,12)&&Number.isInteger(requested)&&requested>=0&&requested<Math.ceil(d.items.length/size)?requested:Math.max(0,Math.floor(first/size));render();status.textContent='Готово. Изменения сохраняются автоматически.'}).catch(e=>status.textContent=e.message);
