"""정확한 한 프레임씩 보면서 접촉을 확인하는 코칭 검수 화면."""
from html import escape
from string import Template


def pull_page(result):
    review = result.pull_review
    labels = {"ready": "이륙 · 원판이 바닥에서 올라가기 시작하는 순간",
              "transition": "무릎 통과 · 원판 중심이 가까운 무릎 높이를 지나는 순간",
              "drive": "접촉 · 바와 몸이 닿는 순간"}
    last = result.frame_count - 1
    controls = []
    for key, label in labels.items():
        value = review.events.get(key, "")
        candidate = review.suggested[key]
        suggested = f"자동 후보 {candidate}번" if candidate is not None else "자동 후보 보류"
        controls.append(f'''
<fieldset><legend>{key}</legend><small class="meaning">{label}</small>
<label>{key} 프레임 <input name="{key}" type="number" min="0" max="{last}" step="1" value="{value}" required></label>
<button type="button" data-show="{key}">해당 장면 보기</button>
<button type="button" data-use="{key}">지금 보는 프레임 사용</button>
<small>{suggested} · {escape(review.reasons[key])}</small></fieldset>''')
    def number(value, unit, digits):
        return "산출 보류" if value is None else f"{value:,.{digits}f} {unit}"
    bounds = result.pull_speed_range
    range_line = ""
    if bounds:
        low, high = bounds.late_over_early
        range_line = f"<p>시점 ±1프레임에서 판정 변경 · 뒤/앞 {low:.2f}~{high:.2f}배</p>"
    first = next((review.events[key] for key in ("drive", "transition", "ready") if key in review.events), 0)
    return PAGE.substitute(
        status="접촉 확인 완료" if review.status == "confirmed" else "접촉 확인 전 · 잠정 결과",
        early=number(result.lift_to_knee_speed_px_s, "px/s", 1),
        late=number(result.knee_to_contact_speed_px_s, "px/s", 1),
        factor=number(result.pull_speed_late_over_early, "배", 3), verdict=escape(result.pull_speed_verdict),
        reason=escape(result.pull_speed_reason), bounds=range_line, fields="".join(controls), first=first, last=last,
        url=escape(result.pull_review_url, quote=True), overlay=escape(result.files.get("overlay.mp4", ""), quote=True))


PAGE = Template("""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>당기기 구간 검수</title><style>
*{box-sizing:border-box}body{margin:0;background:#f6f7f4;color:#192825;font:16px/1.5 system-ui,sans-serif}
main{margin:auto;max-width:1000px;padding:20px}h1{font-size:1.6rem}h2{font-size:1.15rem;margin:0}
.layout{display:grid;grid-template-columns:1fr 1fr;gap:24px}.viewer,form{min-width:0}img{width:100%;height:min(55vh,520px);object-fit:contain;background:#192825}
button,input{font:inherit}button{cursor:pointer;padding:5px 9px}input[type=range]{width:100%;margin:12px 0}
fieldset{min-width:0;border:1px solid #bbc5c1;border-radius:6px;margin:6px 0 12px;padding:8px}legend{font-weight:700;font-size:1.15rem}.meaning{margin:0 0 7px;font-size:.85rem;color:#40544e}
input[type=number]{width:5.3em;padding:3px}small{display:block;color:#40544e;margin-top:8px}
fieldset button{font-size:.86rem;margin:3px 0}a{color:#185544}.metrics{margin:18px 0;background:white;padding:12px 16px;border-left:4px solid #1c6955}
.metrics p{margin:4px 0}button[type=submit]{color:white;background:#195e4c;border:0;border-radius:4px;padding:9px 14px}
@media(max-width:700px){.layout{grid-template-columns:1fr}} </style>
<main><h1>당기기 구간 검수</h1>
<section class="metrics"><h2>$status</h2><p>ready → transition <strong>$early</strong> · transition → drive <strong>$late</strong></p>
<small>이륙에서 무릎 통과까지 · 무릎 통과에서 바와 몸의 접촉까지의 평균 수직속도</small>
<p>뒤/앞 <strong>$factor</strong> · $verdict</p>$bounds<small>$reason</small></section>
<div class="layout"><section class="viewer">
<img id="frame" src="$url/frames/$first" alt="원판과 가까운 관절을 겹친 프레임"><div>
<button type="button" id="prev">이전 장</button> <strong id="number">프레임 $first</strong> <button type="button" id="next">다음 장</button></div>
<input type="range" id="seek" min="0" max="$last" value="$first" aria-label="볼 프레임"><a href="$overlay" target="_blank">전체 오버레이 재생</a>
</section><form id="review"><p>바와 몸이 닿는 장면을 확인하세요. 후보가 맞으면 바로 저장하고, 다르면 원하는 장면에서 <b>지금 보는 프레임 사용</b>을 누르세요.</p>
$fields<button type="submit">접촉 확인 · 다시 계산</button><p aria-live="polite" id="message"></p></form></div></main>
<script>
const form=document.querySelector('form'), slide=document.querySelector('#seek'), frame=document.querySelector('#frame');
const number=document.querySelector('#number'), msg=document.querySelector('#message');
let loaded=null, generation=0;
function go(n){
  n=Math.min($last,Math.max(0,Number(n))); slide.value=n;
  const version=++generation, image=new Image(); loaded=null; number.textContent='프레임 '+n+' 불러오는 중';
  image.onload=()=>{if(version!==generation)return; frame.src=image.src; loaded=n; number.textContent='프레임 '+n;};
  image.onerror=()=>{if(version===generation)number.textContent='프레임을 읽을 수 없습니다.';};
  image.src='$url/frames/'+n;
}
slide.oninput=()=>go(slide.value);
document.querySelector('#prev').onclick=()=>go(Number(slide.value)-1);
document.querySelector('#next').onclick=()=>go(Number(slide.value)+1);
document.querySelectorAll('[data-show]').forEach(b=>b.onclick=()=>{
  const value=form.elements[b.dataset.show].value; if(value!=='')go(value);
});
document.querySelectorAll('[data-use]').forEach(b=>b.onclick=()=>{
  if(loaded===null){msg.textContent='장면이 표시된 뒤 선택하세요.';return;}
  form.elements[b.dataset.use].value=loaded; msg.textContent='';
});
form.onsubmit=async e=>{
  e.preventDefault(); const submit=form.querySelector('[type=submit]'); submit.disabled=true;
  const body=new FormData(form);
  for(const name of ['ready','transition']){
    if(form.elements[name].value===form.elements[name].defaultValue)body.delete(name);
  }
  try{
    const response=await fetch('$url',{method:'POST',body});
    const data=await response.json(); if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'프레임 번호를 확인하세요.');
    location.reload();
  } catch(err){msg.textContent=err.message;submit.disabled=false;}
};
go($first);
</script></html>""")
