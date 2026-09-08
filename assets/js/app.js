/*
 * app.js — Хэрэглэгчийн интерфэйс ба төлөвийн удирдлага.
 */
(function (global) {
  'use strict';
  var TE = global.TE;
  var corpus = TE.corpus;
  var model = TE.model;
  var viz = TE.viz;
  var CFG = model.CFG;

  var MAX_TOKENS = 24;

  var EXAMPLES = [
    'Монгол улсын нийслэл хот бол',
    'Багш хичээл',
    'Хиймэл оюун ухаан',
    'Гол ус тунгалаг',
    'Анхаарлын механизм',
    'Өвөл цас их'
  ];

  var el = {};
  var state = {
    text: '',
    temperature: 1,
    topK: 0,
    topP: 1,
    block: 0,
    head: 0,
    selected: -1,   // сонгосон токены индекс (-1 = сүүлийнх)
    tokens: [],
    trace: null,
    probs: null
  };

  function $(id) { return document.getElementById(id); }

  function collectElements() {
    ['prompt', 'generate', 'reset', 'examples', 'temperature', 'topk', 'topp',
     'temp-out', 'topk-out', 'topp-out', 'tokens', 'token-count', 'embed-canvas',
     'embed-tooltip', 'block-tabs', 'head-tabs', 'head-desc', 'q-canvas', 'k-canvas',
     'v-canvas', 'attn-matrix', 'attn-bars', 'mlp-canvas', 'resid-canvas', 'probs',
     'entropy-note', 'd-model-1', 'd-sem-1', 'd-pos-1', 'ff-dims', 'vocab-size-note'
    ].forEach(function (id) {
      el[id] = $(id);
    });
  }

  /* ------------------------------------------------------------------ */
  /* Тусламжийн функцууд                                                */
  /* ------------------------------------------------------------------ */
  function displayToken(token) {
    if (token === corpus.BOS) return '⟨эхлэл⟩';
    if (token === corpus.UNK) return '⟨үл мэдэгдэх⟩';
    return token;
  }

  function shortLabel(t) {
    var s = displayToken(t.token);
    return s.length > 9 ? s.slice(0, 8) + '…' : s;
  }

  function selectedIndex() {
    var T = state.tokens.length;
    if (state.selected < 0 || state.selected >= T) return T - 1;
    return state.selected;
  }

  function entropyOf(probs) {
    var h = 0;
    for (var i = 0; i < probs.length; i++) {
      if (probs[i] > 0) h -= probs[i] * Math.log2(probs[i]);
    }
    return h;
  }

  /* ------------------------------------------------------------------ */
  /* Тооцоолол                                                          */
  /* ------------------------------------------------------------------ */
  function compute() {
    var tokens = corpus.tokenize(state.text);
    if (tokens.length > MAX_TOKENS) tokens = tokens.slice(0, MAX_TOKENS);
    if (tokens.length === 0) tokens = corpus.tokenize('');
    state.tokens = tokens;
    state.trace = model.forward(tokens.map(function (t) { return t.id; }));
    state.probs = model.probabilities(state.trace.logits, state.temperature, state.topK, state.topP);
  }

  /* ------------------------------------------------------------------ */
  /* Дүрслэл                                                            */
  /* ------------------------------------------------------------------ */
  function renderTokens() {
    var host = el.tokens;
    host.textContent = '';
    var sel = selectedIndex();
    state.tokens.forEach(function (t, i) {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'token' +
        (i === sel ? ' selected' : '') +
        (t.special ? ' special' : '') +
        (t.unknown ? ' unknown' : '') +
        (t.continuation ? ' continuation' : '');
      btn.setAttribute('role', 'option');
      btn.setAttribute('aria-selected', String(i === sel));
      btn.title = 'Токен #' + i + ' · толь бичгийн дугаар ' + t.id +
        (t.unknown ? ' · толь бичигт байхгүй' : '');
      var label = document.createElement('span');
      label.textContent = displayToken(t.token);
      var idx = document.createElement('span');
      idx.className = 'idx';
      idx.textContent = String(i);
      btn.appendChild(label);
      btn.appendChild(idx);
      btn.addEventListener('click', function () {
        state.selected = i;
        render();
      });
      host.appendChild(btn);
    });
    el['token-count'].textContent =
      state.tokens.length + ' токен · толь бичиг ' + corpus.vocab.length + ' токентой' +
      (state.tokens.length >= MAX_TOKENS ? ' · дээд хязгаар ' + MAX_TOKENS : '');
  }

  function renderEmbeddings() {
    viz.drawMatrix(el['embed-canvas'], state.trace.x0, { highlight: selectedIndex() });
  }

  function renderBlockTabs() {
    var host = el['block-tabs'];
    host.textContent = '';
    for (var b = 0; b < CFG.N_BLOCKS; b++) {
      (function (index) {
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'tab';
        btn.setAttribute('role', 'tab');
        btn.setAttribute('aria-selected', String(index === state.block));
        btn.textContent = 'Блок ' + (index + 1);
        btn.addEventListener('click', function () {
          state.block = index;
          render();
        });
        host.appendChild(btn);
      })(b);
    }
  }

  function renderHeadTabs() {
    var host = el['head-tabs'];
    host.textContent = '';
    model.HEAD_INFO.forEach(function (info, index) {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'tab';
      btn.setAttribute('role', 'tab');
      btn.setAttribute('aria-selected', String(index === state.head));
      btn.textContent = info.name + ' · ' + info.role;
      btn.addEventListener('click', function () {
        state.head = index;
        render();
      });
      host.appendChild(btn);
    });
    var current = model.HEAD_INFO[state.head];
    el['head-desc'].textContent = current.role + ' — ' + current.desc;
  }

  function renderQKV() {
    var head = state.trace.blocks[state.block].heads[state.head];
    var hi = selectedIndex();
    viz.drawMatrix(el['q-canvas'], head.q, { highlight: hi });
    viz.drawMatrix(el['k-canvas'], head.k, { highlight: hi });
    viz.drawMatrix(el['v-canvas'], head.v, { highlight: hi });
  }

  function renderAttention() {
    var head = state.trace.blocks[state.block].heads[state.head];
    var labels = state.tokens.map(shortLabel);
    viz.drawAttention(el['attn-matrix'], head.attn, labels, {
      selected: selectedIndex(),
      onSelect: function (q) {
        state.selected = q;
        render();
      }
    });

    var row = head.attn[selectedIndex()];
    var host = el['attn-bars'];
    host.textContent = '';
    var pairs = [];
    for (var i = 0; i < row.length; i++) if (row[i] > 0) pairs.push({ i: i, v: row[i] });
    pairs.sort(function (a, b) { return b.v - a.v; });
    pairs.slice(0, 10).forEach(function (p) {
      var line = document.createElement('div');
      line.className = 'attn-bar-row';
      var tok = document.createElement('span');
      tok.className = 'tok';
      tok.textContent = '#' + p.i + ' ' + displayToken(state.tokens[p.i].token);
      var track = document.createElement('span');
      track.className = 'track';
      var fill = document.createElement('span');
      fill.className = 'fill';
      fill.style.width = (p.v * 100).toFixed(1) + '%';
      track.appendChild(fill);
      var val = document.createElement('span');
      val.className = 'val';
      val.textContent = (p.v * 100).toFixed(1) + '%';
      line.appendChild(tok);
      line.appendChild(track);
      line.appendChild(val);
      host.appendChild(line);
    });
  }

  function renderMLPAndResidual() {
    var block = state.trace.blocks[state.block];
    var hi = selectedIndex();
    // Далд давхаргын идэвхжилтийг харагдахуйц болгож 4 мөрөнд нугалж харуулна.
    var act = block.mlpAct[hi];
    var lanes = 4;
    var per = Math.ceil(act.length / lanes);
    var folded = [];
    for (var c = 0; c < per; c++) {
      var col = new Float64Array(lanes);
      for (var r = 0; r < lanes; r++) {
        var idx = r * per + c;
        col[r] = idx < act.length ? act[idx] : 0;
      }
      folded.push(col);
    }
    viz.drawMatrix(el['mlp-canvas'], folded);
    // Блокын өмнөх (x) ба дараах (resid2) төлөвийг дээд/доод мөрөнд харьцуулна.
    drawTwoRows(el['resid-canvas'], blockInput(hi), block.resid2[hi]);
  }

  /** Сонгосон блок руу орж буй вектор: эхний блокт x0, бусад блокт өмнөх блокын гаралт. */
  function blockInput(tokenIndex) {
    return state.block === 0
      ? state.trace.x0[tokenIndex]
      : state.trace.blocks[state.block - 1].resid2[tokenIndex];
  }

  /** Хоёр векторыг дээд/доод мөр болгон зурна. */
  function drawTwoRows(canvas, top, bottom) {
    var columns = [];
    var maxAbs = Math.max(viz.maxAbsOf([top]), viz.maxAbsOf([bottom]));
    for (var i = 0; i < top.length; i++) {
      var col = new Float64Array(2);
      col[0] = top[i];
      col[1] = bottom[i];
      columns.push(col);
    }
    viz.drawMatrix(canvas, columns, { maxAbs: maxAbs });
  }

  function renderProbs() {
    var host = el.probs;
    host.textContent = '';
    var top = model.topTokens(state.probs.probs, 12);
    top.forEach(function (item, i) {
      var line = document.createElement('div');
      line.className = 'prob-row' + (i === 0 ? ' top' : '');
      var tok = document.createElement('span');
      tok.className = 'tok';
      tok.textContent = displayToken(item.token);
      tok.title = item.token === corpus.BOS
        ? 'Өгүүлбэрийн зааг — GPT-2 дэх <|endoftext|>-тэй утга нэг.'
        : 'Толь бичгийн дугаар ' + item.id;
      var track = document.createElement('span');
      track.className = 'track';
      var fill = document.createElement('span');
      fill.className = 'fill';
      fill.style.width = Math.max(1, item.prob * 100).toFixed(1) + '%';
      track.appendChild(fill);
      var val = document.createElement('span');
      val.className = 'val';
      val.textContent = (item.prob * 100).toFixed(1) + '%';
      line.appendChild(tok);
      line.appendChild(track);
      line.appendChild(val);
      host.appendChild(line);
    });

    var candidates = 0;
    for (var i2 = 0; i2 < state.probs.probs.length; i2++) if (state.probs.probs[i2] > 0) candidates++;
    el['entropy-note'].textContent =
      'Боломжит хувилбар: ' + candidates + ' · энтропи: ' +
      entropyOf(state.probs.probs).toFixed(2) + ' бит · температур: ' +
      state.temperature.toFixed(2);
  }

  function render() {
    compute();
    renderTokens();
    renderEmbeddings();
    renderBlockTabs();
    renderHeadTabs();
    renderQKV();
    renderAttention();
    renderMLPAndResidual();
    renderProbs();
  }

  /* ------------------------------------------------------------------ */
  /* Үйл явдлууд                                                        */
  /* ------------------------------------------------------------------ */
  function attachTooltip(canvas, describe) {
    var tip = el['embed-tooltip'];
    canvas.addEventListener('mousemove', function (ev) {
      var cell = viz.cellAt(canvas, ev.clientX, ev.clientY);
      if (!cell) { tip.hidden = true; return; }
      tip.hidden = false;
      tip.textContent = describe(cell);
      tip.style.left = (ev.clientX + 14) + 'px';
      tip.style.top = (ev.clientY + 14) + 'px';
    });
    canvas.addEventListener('mouseleave', function () { tip.hidden = true; });
  }

  function appendSampledToken() {
    var id = model.sample(state.probs.probs);
    var token = corpus.vocab[id];
    if (token === corpus.BOS) {
      el['entropy-note'].textContent = 'Загвар өгүүлбэрийн төгсгөлийг сонголоо (⟨эхлэл⟩).';
      return;
    }
    var text = state.text.trim();
    if (token.indexOf('##') === 0) {
      text += token.slice(2);
    } else if (/^[.,!?;:]$/.test(token)) {
      text += token;
    } else {
      text += (text ? ' ' : '') + token;
    }
    state.text = text;
    el.prompt.value = text;
    state.selected = -1;
    render();
  }

  function bindEvents() {
    el.prompt.addEventListener('input', function () {
      state.text = el.prompt.value;
      state.selected = -1;
      render();
    });

    el.generate.addEventListener('click', appendSampledToken);

    el.reset.addEventListener('click', function () {
      state.text = EXAMPLES[0];
      el.prompt.value = state.text;
      state.temperature = 1; state.topK = 0; state.topP = 1;
      state.block = 0; state.head = 0; state.selected = -1;
      el.temperature.value = '1'; el.topk.value = '0'; el.topp.value = '1';
      syncOutputs();
      render();
    });

    el.temperature.addEventListener('input', function () {
      state.temperature = parseFloat(el.temperature.value);
      syncOutputs();
      render();
    });
    el.topk.addEventListener('input', function () {
      state.topK = parseInt(el.topk.value, 10);
      syncOutputs();
      render();
    });
    el.topp.addEventListener('input', function () {
      state.topP = parseFloat(el.topp.value);
      syncOutputs();
      render();
    });

    var resizeTimer = null;
    global.addEventListener('resize', function () {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(render, 120);
    });

    attachTooltip(el['embed-canvas'], function (cell) {
      var tok = state.tokens[cell.col];
      var kind = cell.row < CFG.D_SEM ? 'утгын хэмжээс' : 'байрлалын хэмжээс';
      return displayToken(tok.token) + '\n' + kind + ' #' + cell.row +
        '\nутга: ' + cell.value.toFixed(3);
    });

    [['q-canvas', 'Q'], ['k-canvas', 'K'], ['v-canvas', 'V']].forEach(function (pair) {
      attachTooltip(el[pair[0]], function (cell) {
        return pair[1] + ' вектор · токен ' + displayToken(state.tokens[cell.col].token) +
          '\nхэмжээс #' + cell.row + '\nутга: ' + cell.value.toFixed(3);
      });
    });

    attachTooltip(el['mlp-canvas'], function (cell) {
      return 'MLP идэвхжилт\nутга: ' + cell.value.toFixed(3);
    });
    attachTooltip(el['resid-canvas'], function (cell) {
      return (cell.row === 0 ? 'Блок руу орох' : 'Блокоос гарах') +
        '\nхэмжээс #' + cell.col + '\nутга: ' + cell.value.toFixed(3);
    });
  }

  function syncOutputs() {
    el['temp-out'].textContent = state.temperature.toFixed(2);
    el['topk-out'].textContent = state.topK === 0 ? '0 (хязгааргүй)' : String(state.topK);
    el['topp-out'].textContent = state.topP.toFixed(2);
  }

  function renderExamples() {
    var host = el.examples;
    host.textContent = '';
    EXAMPLES.forEach(function (text) {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'example-btn';
      btn.textContent = text;
      btn.addEventListener('click', function () {
        state.text = text;
        el.prompt.value = text;
        state.selected = -1;
        render();
      });
      host.appendChild(btn);
    });
  }

  function fillStaticNumbers() {
    el['d-model-1'].textContent = String(CFG.D_MODEL);
    el['d-sem-1'].textContent = String(CFG.D_SEM);
    el['d-pos-1'].textContent = String(CFG.D_POS);
    el['ff-dims'].textContent = CFG.D_MODEL + ' → ' + CFG.D_FF + ' → ' + CFG.D_MODEL;
    el['vocab-size-note'].textContent = String(corpus.vocab.length);
  }

  function init() {
    collectElements();
    state.text = el.prompt.value || EXAMPLES[0];
    fillStaticNumbers();
    renderExamples();
    syncOutputs();
    bindEvents();
    render();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(typeof window !== 'undefined' ? window : globalThis);
