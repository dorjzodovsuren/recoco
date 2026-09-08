/*
 * Загварын математикийн шалгуур сорилууд.
 * Ажиллуулах:  node tests/model.test.js
 */
'use strict';
require('../assets/js/corpus.js');
require('../assets/js/model.js');

var corpus = globalThis.TE.corpus;
var model = globalThis.TE.model;

var passed = 0;
var failed = 0;

function check(name, condition, extra) {
  if (condition) {
    passed++;
    console.log('  ✓ ' + name);
  } else {
    failed++;
    console.error('  ✗ ' + name + (extra ? '  → ' + extra : ''));
  }
}

function close(a, b, eps) {
  return Math.abs(a - b) < (eps === undefined ? 1e-6 : eps);
}

console.log('\n1. Токенчлогч');
var toks = corpus.tokenize('Монгол улсын нийслэл хот бол');
check('эхлэлийн токеноор эхэлнэ', toks[0].token === corpus.BOS);
check('"улсын" нь үндэс + дагавар болж задарна',
  toks.map(function (t) { return t.token; }).join(' ').indexOf('улс ##ын') !== -1,
  toks.map(function (t) { return t.token; }).join(' '));
check('толь бичигт байхгүй үг <үл-мэдэгдэх> болно',
  corpus.tokenize('пингвин').slice(1)[0].unknown === true);
check('бүх токен хүчинтэй дугаартай',
  toks.every(function (t) { return Number.isInteger(t.id) && t.id >= 0 && t.id < corpus.vocab.length; }));
check('толь бичиг давхардалгүй',
  new Set(corpus.vocab).size === corpus.vocab.length);

console.log('\n2. LayerNorm');
var ln = model.layerNorm(Float64Array.from([1, 2, 3, 4, 10]));
var mean = ln.reduce(function (a, b) { return a + b; }, 0) / ln.length;
var vari = ln.reduce(function (a, b) { return a + b * b; }, 0) / ln.length;
check('дундаж ≈ 0', close(mean, 0, 1e-9), String(mean));
check('дисперс ≈ 1', close(vari, 1, 1e-4), String(vari));

console.log('\n3. Софтмакс ба GELU');
var sm = model.softmax(Float64Array.from([1, 2, 3]));
check('нийлбэр = 1', close(sm.reduce(function (a, b) { return a + b; }, 0), 1));
check('дараалал хадгалагдана', sm[2] > sm[1] && sm[1] > sm[0]);
check('GELU(0) = 0', close(model.gelu(0), 0));
check('GELU(x) ≈ x том утганд', close(model.gelu(6), 6, 1e-3));
check('GELU сөрөг утгад бага', model.gelu(-3) < 0 && model.gelu(-3) > -0.1);

console.log('\n4. Урагш дамжлага ба анхаарал');
var ids = toks.map(function (t) { return t.id; });
var trace = model.forward(ids);
check('блокуудын тоо', trace.blocks.length === model.CFG.N_BLOCKS);
check('толгойн тоо', trace.blocks[0].heads.length === model.CFG.N_HEADS);
check('нуугдмал векторын хэмжээс', trace.x0[0].length === model.CFG.D_MODEL);

var maskOk = true, sumOk = true;
trace.blocks.forEach(function (b) {
  b.heads.forEach(function (h) {
    h.attn.forEach(function (row, t) {
      var s = 0;
      for (var j = 0; j < row.length; j++) {
        if (j > t && row[j] !== 0) maskOk = false;
        s += row[j];
      }
      if (!close(s, 1, 1e-9)) sumOk = false;
    });
  });
});
check('шалтгаант маск (ирээдүй рүү харахгүй)', maskOk);
check('анхаарлын мөр бүрийн нийлбэр = 1', sumOk);

function argmax(arr) {
  var best = 0;
  for (var i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}
var last = ids.length - 1;
check('1-р толгой өмнөх токенд анхаарна',
  argmax(trace.blocks[0].heads[0].attn[last]) === last - 1,
  'argmax=' + argmax(trace.blocks[0].heads[0].attn[last]));
check('2-р толгой эхний токенд анхаарна',
  argmax(trace.blocks[0].heads[1].attn[last]) === 0,
  'argmax=' + argmax(trace.blocks[0].heads[1].attn[last]));

console.log('\n5. Тодорхой байдал (determinism)');
var trace2 = model.forward(ids);
var same = true;
for (var i = 0; i < trace.logits.length; i++) {
  if (!close(trace.logits[i], trace2.logits[i], 1e-12)) same = false;
}
check('ижил оролт → ижил логит', same);

console.log('\n6. Температур ба түүвэрлэлт');
var p1 = model.probabilities(trace.logits, 0.3, 0, 1).probs;
var p2 = model.probabilities(trace.logits, 1.5, 0, 1).probs;
check('бага температур → илүү тодорхой тархалт', Math.max.apply(null, Array.from(p1)) > Math.max.apply(null, Array.from(p2)));
check('магадлалын нийлбэр = 1', close(Array.from(p2).reduce(function (a, b) { return a + b; }, 0), 1, 1e-9));

var topk = model.probabilities(trace.logits, 1, 5, 1);
var nonZero = Array.from(topk.probs).filter(function (v) { return v > 0; }).length;
check('top-k = 5 нь 5 хувилбар үлдээнэ', nonZero === 5, 'nonZero=' + nonZero);
check('top-k магадлал дахин нормчлогдоно',
  close(Array.from(topk.probs).reduce(function (a, b) { return a + b; }, 0), 1, 1e-9));

var topp = model.probabilities(trace.logits, 1, 0, 0.5);
var nzp = Array.from(topp.probs).filter(function (v) { return v > 0; }).length;
check('top-p = 0.5 нь хувилбарыг хязгаарлана', nzp > 0 && nzp < nonZero + 50);

console.log('\n7. Таамаглалын утга учир');
function predict(text) {
  var t = corpus.tokenize(text);
  var tr = model.forward(t.map(function (x) { return x.id; }));
  var pr = model.probabilities(tr.logits, 1, 0, 1);
  return model.topTokens(pr.probs, 3).map(function (x) { return x.token; });
}
check('"Багш хичээл" → "заадаг"', predict('Багш хичээл')[0] === 'заадаг', predict('Багш хичээл').join(','));
check('"Гол ус тунгалаг" → "урсдаг"', predict('Гол ус тунгалаг')[0] === 'урсдаг', predict('Гол ус тунгалаг').join(','));
check('"Би ном" → "уншиж"/"унших"', predict('Би ном').some(function (t) { return t === 'уншиж' || t === 'унших'; }), predict('Би ном').join(','));

console.log('\n8. Эмбеддинг');
check('эмбеддингийн хэмжээс', model.E[0].length === model.CFG.D_SEM);
var normOk = model.E.every(function (row) {
  var n = Math.sqrt(row.reduce(function (a, b) { return a + b * b; }, 0));
  return n === 0 || close(n, model.CFG.SEM_AMP * Math.sqrt(model.CFG.D_SEM), 1e-6);
});
check('эмбеддинг бүр ижил урттай', normOk);
var pos0 = model.positionalVector(0);
var pos5 = model.positionalVector(5);
check('байрлалын вектор байрлалаас хамаарна', pos0[0] !== pos5[0]);

console.log('\n─────────────────────────────');
console.log('Амжилттай: ' + passed + ',  Алдаа: ' + failed);
process.exit(failed === 0 ? 0 : 1);
