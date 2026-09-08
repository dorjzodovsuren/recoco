/*
 * viz.js — Матриц ба анхаарлын дүрслэл (canvas + SVG, гадаад сан ашиглаагүй).
 */
(function (global) {
  'use strict';
  var TE = (global.TE = global.TE || {});

  var SVG_NS = 'http://www.w3.org/2000/svg';

  /* ---------------- Өнгөний хуваарь ---------------- */

  /** Тэгш хэмт (сөрөг → цэнхэр, эерэг → улаан) хуваарь. */
  function diverging(value, maxAbs) {
    var t = Math.max(-1, Math.min(1, maxAbs > 0 ? value / maxAbs : 0));
    var a = Math.abs(t);
    var eased = Math.pow(a, 0.7);
    if (t >= 0) {
      return 'rgb(' + Math.round(255 - 40 * eased) + ',' +
        Math.round(255 - 175 * eased) + ',' + Math.round(255 - 190 * eased) + ')';
    }
    return 'rgb(' + Math.round(255 - 200 * eased) + ',' +
      Math.round(255 - 130 * eased) + ',' + Math.round(255 - 25 * eased) + ')';
  }

  /** 0..1 утгын дараалсан хуваарь (цайвараас индиго хүртэл). */
  function sequential(value) {
    var t = Math.max(0, Math.min(1, value));
    var eased = Math.pow(t, 0.55);
    return 'rgb(' + Math.round(247 - 171 * eased) + ',' +
      Math.round(248 - 153 * eased) + ',' + Math.round(253 - 38 * eased) + ')';
  }

  function maxAbsOf(rows) {
    var m = 0;
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      for (var j = 0; j < r.length; j++) {
        var v = Math.abs(r[j]);
        if (v > m) m = v;
      }
    }
    return m || 1;
  }

  /* ---------------- Матрицын зураглал ---------------- */
  /**
   * Векторуудын жагсаалтыг дулааны зураглал болгон зурна.
   * Х тэнхлэг = вектор (токен), Y тэнхлэг = хэмжээс.
   *
   * @param {HTMLCanvasElement} canvas
   * @param {Array<Float64Array|number[]>} vectors
   * @param {object} [opts] {highlight:number, maxAbs:number}
   */
  function drawMatrix(canvas, vectors, opts) {
    var options = opts || {};
    var dpr = global.devicePixelRatio || 1;
    var cssW = canvas.clientWidth || 300;
    var cssH = canvas.clientHeight || 120;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);

    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    var cols = vectors.length;
    if (!cols) return null;
    var rows = vectors[0].length;
    var maxAbs = options.maxAbs || maxAbsOf(vectors);
    var cw = cssW / cols;
    var ch = cssH / rows;

    for (var c = 0; c < cols; c++) {
      var vec = vectors[c];
      for (var r = 0; r < rows; r++) {
        ctx.fillStyle = diverging(vec[r], maxAbs);
        ctx.fillRect(c * cw, r * ch, Math.ceil(cw) + 0.5, Math.ceil(ch) + 0.5);
      }
    }

    if (options.highlight !== undefined && options.highlight >= 0 && options.highlight < cols) {
      ctx.strokeStyle = 'rgba(76, 95, 215, .95)';
      ctx.lineWidth = 2;
      ctx.strokeRect(options.highlight * cw + 1, 1, cw - 2, cssH - 2);
    }

    var grid = { cols: cols, rows: rows, cw: cw, ch: ch, maxAbs: maxAbs, vectors: vectors };
    canvas._grid = grid;
    return grid;
  }

  /** Хулганы байрлалаас нүдний индексийг олно. */
  function cellAt(canvas, clientX, clientY) {
    var grid = canvas._grid;
    if (!grid) return null;
    var rect = canvas.getBoundingClientRect();
    var col = Math.floor((clientX - rect.left) / grid.cw);
    var row = Math.floor((clientY - rect.top) / grid.ch);
    if (col < 0 || row < 0 || col >= grid.cols || row >= grid.rows) return null;
    return { col: col, row: row, value: grid.vectors[col][row] };
  }

  /* ---------------- Анхаарлын матриц (SVG) ---------------- */
  /**
   * @param {HTMLElement} container
   * @param {Array<Float64Array>} attn  attn[query][key]
   * @param {string[]} labels
   * @param {object} opts {selected:number, onSelect:function, onHover:function}
   */
  function drawAttention(container, attn, labels, opts) {
    var options = opts || {};
    var T = attn.length;
    var pad = { left: 74, top: 66, right: 8, bottom: 8 };
    var cell = Math.max(16, Math.min(30, Math.floor(360 / Math.max(T, 1))));
    var w = pad.left + T * cell + pad.right;
    var h = pad.top + T * cell + pad.bottom;

    container.textContent = '';
    var svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('width', String(w));
    svg.setAttribute('height', String(h));
    svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Анхаарлын жингийн матриц');

    function text(x, y, str, cls, anchor, rotate) {
      var el = document.createElementNS(SVG_NS, 'text');
      el.setAttribute('x', String(x));
      el.setAttribute('y', String(y));
      el.setAttribute('class', cls);
      el.setAttribute('text-anchor', anchor || 'start');
      if (rotate) el.setAttribute('transform', 'rotate(-55 ' + x + ' ' + y + ')');
      el.textContent = str;
      return el;
    }

    // Түлхүүрүүд (баганын гарчиг)
    for (var k = 0; k < T; k++) {
      svg.appendChild(text(
        pad.left + k * cell + cell / 2, pad.top - 6, labels[k],
        'attn-axis-label' + (options.selectedKey === k ? ' active' : ''), 'start', true));
    }
    // Асуулгууд (мөрийн гарчиг)
    for (var q = 0; q < T; q++) {
      svg.appendChild(text(
        pad.left - 8, pad.top + q * cell + cell / 2 + 3, labels[q],
        'attn-axis-label' + (options.selected === q ? ' active' : ''), 'end'));
    }

    for (q = 0; q < T; q++) {
      for (k = 0; k < T; k++) {
        var rect = document.createElementNS(SVG_NS, 'rect');
        rect.setAttribute('x', String(pad.left + k * cell));
        rect.setAttribute('y', String(pad.top + q * cell));
        rect.setAttribute('width', String(cell - 1));
        rect.setAttribute('height', String(cell - 1));
        rect.setAttribute('class', 'attn-cell');
        if (k > q) {
          // Шалтгаант маскаар хаагдсан нүд
          rect.setAttribute('fill', 'transparent');
          rect.setAttribute('stroke', 'rgba(138,144,168,.28)');
          rect.setAttribute('stroke-dasharray', '2 2');
        } else {
          rect.setAttribute('fill', sequential(attn[q][k]));
        }
        rect.setAttribute('data-q', String(q));
        rect.setAttribute('data-k', String(k));
        var title = document.createElementNS(SVG_NS, 'title');
        title.textContent = k > q
          ? labels[q] + ' → ' + labels[k] + ': маскаар хаагдсан'
          : labels[q] + ' → ' + labels[k] + ': ' + (attn[q][k] * 100).toFixed(1) + '%';
        rect.appendChild(title);
        svg.appendChild(rect);
      }
    }

    if (options.selected !== undefined && options.selected >= 0) {
      var hl = document.createElementNS(SVG_NS, 'rect');
      hl.setAttribute('x', String(pad.left - 2));
      hl.setAttribute('y', String(pad.top + options.selected * cell - 2));
      hl.setAttribute('width', String(T * cell + 2));
      hl.setAttribute('height', String(cell + 2));
      hl.setAttribute('fill', 'none');
      hl.setAttribute('stroke', 'var(--accent)');
      hl.setAttribute('stroke-width', '2');
      hl.setAttribute('rx', '3');
      svg.appendChild(hl);
    }

    if (options.onSelect) {
      svg.addEventListener('click', function (ev) {
        var target = ev.target;
        if (target && target.hasAttribute && target.hasAttribute('data-q')) {
          options.onSelect(+target.getAttribute('data-q'), +target.getAttribute('data-k'));
        }
      });
    }

    container.appendChild(svg);
  }

  TE.viz = {
    diverging: diverging,
    sequential: sequential,
    drawMatrix: drawMatrix,
    drawAttention: drawAttention,
    cellAt: cellAt,
    maxAbsOf: maxAbsOf
  };
})(typeof window !== 'undefined' ? window : globalThis);
