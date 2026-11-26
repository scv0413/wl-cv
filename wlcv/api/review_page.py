"""오버레이에서 두 시점을 선택하는 로컬 판독 화면."""
from html import escape
from string import Template


def review_page(overlay_url, chart_url, frame_count, fps, events, has_chart):
    return PAGE.substitute(
        video=escape(overlay_url, quote=True), chart=escape(chart_url, quote=True),
        last=frame_count - 1, fps=float(fps), catch=events.get("catch", ""),
        finish=events.get("finish", ""), hidden="" if has_chart else "hidden",
    )


PAGE = Template("""<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>캐치·피니시 판독</title>
<style>
* { box-sizing: border-box } body { margin: 0; background: #fafaf7; color: #172723; font: 16px/1.55 system-ui, sans-serif }
main { max-width: 900px; margin: auto; padding: 20px } h1 { font-size: 1.5rem } p { margin: 8px 0 }
.layout { display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start }
video { display: block; width: min(100%, 480px); max-height: 66vh; background: #142523 }
section, form { flex: 1 1 260px; min-width: 0 } input, button { font: inherit; padding: 5px 10px }
input { width: 6.3em } label { display: block; margin-top: 12px; font-weight: 600 }
button { cursor: pointer } .steps { margin: 7px 0 } button[type=submit] { margin-top: 18px; color: white; background: #165a4d; border: 0; border-radius: 4px; padding: 9px 16px }
button:disabled { opacity: .55 } a { color: #155b4c } img { max-width: 100% }
</style>
<main><h1>캐치·피니시 판독</h1>
<div class="layout"><section>
<video id="overlay" controls playsinline preload="auto" src="$video" aria-label="판독용 오버레이 영상"></video>
<div class="steps"><button type="button" id="prev">이전 프레임</button> <button type="button" id="next">다음 프레임</button></div>
</section>
<form><p>영상을 멈추고 화면에 표시된 Frame 번호를 선택하세요. 번호를 직접 입력해도 됩니다.</p>
<label for="catch">캐치 · 바벨을 받은 프레임</label>
<input id="catch" name="catch" type="number" min="0" max="$last" step="1" value="$catch" required> <button type="button" data-point="catch">현재 프레임 선택</button>
<label for="finish">피니시 · 일어서기를 마친 프레임</label>
<input id="finish" name="finish" type="number" min="0" max="$last" step="1" value="$finish" required> <button type="button" data-point="finish">현재 프레임 선택</button>
<p><button type="submit">그래프 만들기</button></p><p role="status" id="status"></p>
</form></div><div id="result" $hidden><a id="png" href="$chart" download="catch_to_finish.png">그래프 PNG 저장</a>
<img id="preview" alt="캐치부터 피니시까지 원판과 어깨의 앞뒤 거리 및 체간 기울기"></div></main>
<script>
const video = document.querySelector('video');
const form = document.querySelector('form');
const statusLine = document.querySelector('#status');
const last = $last, fps = $fps;
function currentFrame() { return Math.min(last, Math.max(0, Math.floor(video.currentTime * fps + 0.00001))); }
function step(delta) {
  if (video.readyState < 2) return;
  video.pause();
  const n = Math.max(0, Math.min(last, currentFrame() + delta));
  video.currentTime = (n + 0.5) / fps;
}
document.querySelector('#prev').onclick = () => step(-1);
document.querySelector('#next').onclick = () => step(1);
document.querySelectorAll('[data-point]').forEach(button => button.onclick = () => {
  if (video.readyState < 2 || video.seeking) { statusLine.textContent = '영상 프레임이 표시된 뒤 선택하세요.'; return; }
  video.pause();
  document.getElementById(button.dataset.point).value = currentFrame();
  statusLine.textContent = '';
});
const preview = document.querySelector('#preview');
const result = document.querySelector('#result');
const png = document.querySelector('#png');
if (!result.hidden) preview.src = png.href;
form.onsubmit = async event => {
  event.preventDefault();
  const submit = form.querySelector('[type=submit]');
  submit.disabled = true;
  result.hidden = true;
  statusLine.textContent = '그래프 생성 중…';
  try {
    const response = await fetch(location.pathname, {method: 'POST', body: new FormData(form)});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '두 프레임 번호를 확인하세요.');
    const measurements = data.catch_to_finish;
    const total = measurements.finish_frame - measurements.catch_frame + 1;
    statusLine.textContent = '측정한 프레임 · 원판과 어깨 ' + measurements.distance_frames + '/' + total + '장, 체간 ' + measurements.angle_frames + '/' + total + '장';
    png.href = data.files['catch_to_finish.png'] + '?v=' + Date.now();
    preview.src = png.href;
    result.hidden = false;
  } catch (error) { statusLine.textContent = error.message; }
  finally { submit.disabled = false; }
};
</script></html>""")
