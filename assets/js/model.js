/*
 * model.js — Жижиг GPT маягийн трансформер загварын бүрэн урагш дамжлага.
 *
 * Энэ нь GPT-2 биш, харин ҮЗҮҮЛЭН ЗАГВАР юм: бүх матрицын үйлдэл
 * (эмбеддинг, байрлалын кодчилол, LayerNorm, олон толгойт өөрийг-анхаарах
 * механизм, MLP, үлдэгдэл холболт, софтмакс) нь бодит бөгөөд хөтчид тооцоологддог.
 * Жингүүд нь сургалтаар биш, корпусын статистикаас (PPMI) болон
 * үртэй санамсаргүй тооноос гаралтай тул үр дүн нь заавал утга төгс биш
 * ч архитектурын ажиллагааг үнэн зөв харуулна.
 */
(function (global) {
  'use strict';
  var TE = (global.TE = global.TE || {});
  var corpus = TE.corpus;

  /* ------------------------------------------------------------------ */
  /* Гиперпараметрүүд                                                    */
  /* ------------------------------------------------------------------ */
  var CFG = {
    D_SEM: 96,          // утгын хэмжээс
    D_POS: 16,          // байрлалын хэмжээс
    D_MODEL: 112,       // нийт нуугдмал хэмжээс (96 + 16)
    N_HEADS: 4,
    HEAD_DIM: 28,       // 112 / 4
    N_BLOCKS: 4,
    D_FF: 448,          // MLP-ийн далд давхарга (4 × d_model)
    MAX_POS: 64,
    SEM_AMP: 0.5,       // утгын эмбеддингийн далайц
    POS_AMP: 0.7,       // байрлалын эмбеддингийн далайц
    ATTN_OUT_SCALE: 0.1,
    MLP_OUT_SCALE: 0.06,
    LOGIT_SCALE: 1.6,
    PRIOR_WEIGHT: 0.35,
    SEED: 20240804
  };

  var HEAD_INFO = [
    { name: 'Толгой 1', role: 'Өмнөх токен', desc: 'Байрлалын кодчилолыг ашиглан яг өмнөх токен руу анхаарна.' },
    { name: 'Толгой 2', role: 'Эхний токен', desc: 'Дарааллын эхний токен руу татагдана (анхаарлын "тогтоор").' },
    { name: 'Толгой 3', role: 'Утгын төстэй байдал', desc: 'Утгын хувьд төстэй эмбеддингтэй токенууд руу анхаарна.' },
    { name: 'Толгой 4', role: 'Холимог хэв маяг', desc: 'Байрлал ба утгыг хольсон, тархсан анхаарал.' }
  ];

  /* ------------------------------------------------------------------ */
  /* Үртэй санамсаргүй тоо (mulberry32)                                  */
  /* ------------------------------------------------------------------ */
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rand) {
    var u = 1 - rand(), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function randMatrix(rows, cols, std, rand) {
    var m = new Array(rows);
    for (var i = 0; i < rows; i++) {
      var row = new Float64Array(cols);
      for (var j = 0; j < cols; j++) row[j] = gaussian(rand) * std;
      m[i] = row;
    }
    return m;
  }

  function zeroMatrix(rows, cols) {
    var m = new Array(rows);
    for (var i = 0; i < rows; i++) m[i] = new Float64Array(cols);
    return m;
  }

  /* ------------------------------------------------------------------ */
  /* Матрицын үндсэн үйлдлүүд                                            */
  /* ------------------------------------------------------------------ */
  function matVec(vec, mat, outCols) {
    // vec (n) × mat (n × outCols) → (outCols)
    var out = new Float64Array(outCols);
    for (var i = 0; i < vec.length; i++) {
      var x = vec[i];
      if (x === 0) continue;
      var row = mat[i];
      for (var j = 0; j < outCols; j++) out[j] += x * row[j];
    }
    return out;
  }

  function dot(a, b) {
    var s = 0;
    for (var i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function addInto(target, other, scale) {
    var s = scale === undefined ? 1 : scale;
    for (var i = 0; i < target.length; i++) target[i] += other[i] * s;
    return target;
  }

  function layerNorm(vec) {
    var n = vec.length, mean = 0, i;
    for (i = 0; i < n; i++) mean += vec[i];
    mean /= n;
    var varsum = 0;
    for (i = 0; i < n; i++) { var d = vec[i] - mean; varsum += d * d; }
    var std = Math.sqrt(varsum / n + 1e-5);
    var out = new Float64Array(n);
    for (i = 0; i < n; i++) out[i] = (vec[i] - mean) / std;
    return out;
  }

  function softmax(arr, from, to) {
    var start = from || 0, end = to === undefined ? arr.length : to;
    var max = -Infinity, i;
    for (i = start; i < end; i++) if (arr[i] > max) max = arr[i];
    var sum = 0;
    var out = new Float64Array(arr.length);
    for (i = start; i < end; i++) { out[i] = Math.exp(arr[i] - max); sum += out[i]; }
    for (i = start; i < end; i++) out[i] /= sum;
    return out;
  }

  function gelu(x) {
    return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
  }

  /* ------------------------------------------------------------------ */
  /* 1. Корпусаас PPMI матриц                                            */
  /* ------------------------------------------------------------------ */
  var V = corpus.vocab.length;

  function buildStatistics() {
    var counts = [];
    var i;
    for (i = 0; i < V; i++) counts.push(Object.create(null));
    var rowTotal = new Float64Array(V);
    var colTotal = new Float64Array(V);
    var unigram = new Float64Array(V);
    var total = 0;

    corpus.sentences.forEach(function (sent) {
      var ids = sent.map(function (t) { return corpus.tokenToId[t]; });
      ids.push(corpus.tokenToId[corpus.BOS]); // өгүүлбэрийн төгсгөл → шинэ эхлэл
      for (var k = 0; k < ids.length; k++) {
        unigram[ids[k]] += 1;
        if (k + 1 < ids.length) {
          var a = ids[k], b = ids[k + 1];
          counts[a][b] = (counts[a][b] || 0) + 1;
          rowTotal[a] += 1;
          colTotal[b] += 1;
          total += 1;
        }
      }
    });

    // PPMI = max(0, log( p(a,b) / (p(a)·p(b)) ))
    var ppmi = [];
    for (i = 0; i < V; i++) {
      var row = new Float64Array(V);
      var keys = Object.keys(counts[i]);
      for (var q = 0; q < keys.length; q++) {
        var b2 = +keys[q];
        var pab = counts[i][b2] / total;
        var pa = rowTotal[i] / total;
        var pb = colTotal[b2] / total;
        if (pa > 0 && pb > 0) {
          var val = Math.log(pab / (pa * pb));
          if (val > 0) row[b2] = val;
        }
      }
      ppmi.push(row);
    }

    var uniSum = 0;
    for (i = 0; i < V; i++) uniSum += unigram[i];
    var logPrior = new Float64Array(V);
    for (i = 0; i < V; i++) logPrior[i] = Math.log((unigram[i] + 0.5) / (uniSum + 0.5 * V));

    return { ppmi: ppmi, logPrior: logPrior, counts: counts };
  }

  var stats = buildStatistics();

  /* ------------------------------------------------------------------ */
  /* 2. Эмбеддинг ба зайлуулах (unembedding) матриц                      */
  /* ------------------------------------------------------------------ */
  // R: V × D_SEM санамсаргүй проекц. Johnson–Lindenstrauss-ийн лемм ёсоор
  // E[a]·R[b] ≈ PPMI[a][b] тул зайлуулах матрицаар дараагийн үгийн утга
  // төгс тархалт гарч ирнэ.
  var randMain = rng(CFG.SEED);
  var R = randMatrix(V, CFG.D_SEM, 1 / Math.sqrt(CFG.D_SEM), randMain);

  var E = zeroMatrix(V, CFG.D_SEM);
  (function buildEmbeddings() {
    for (var a = 0; a < V; a++) {
      var row = stats.ppmi[a];
      var acc = E[a];
      for (var b = 0; b < V; b++) {
        var w = row[b];
        if (w === 0) continue;
        var rb = R[b];
        for (var d = 0; d < CFG.D_SEM; d++) acc[d] += w * rb[d];
      }
      // Мөр бүрийг нэгж уртад хүргэж, дараа нь далайцаар нь томруулна.
      var norm = Math.sqrt(dot(acc, acc));
      if (norm > 1e-9) {
        var scale = (CFG.SEM_AMP * Math.sqrt(CFG.D_SEM)) / norm;
        for (var d2 = 0; d2 < CFG.D_SEM; d2++) acc[d2] *= scale;
      }
    }
  })();

  /* ------------------------------------------------------------------ */
  /* 3. Байрлалын синусоид кодчилол                                      */
  /* ------------------------------------------------------------------ */
  var OMEGA = [];
  (function () {
    var pairs = CFG.D_POS / 2;
    for (var j = 0; j < pairs; j++) {
      OMEGA.push(1 / Math.pow(100, j / (pairs - 1)));
    }
  })();

  function positionalVector(pos) {
    var out = new Float64Array(CFG.D_POS);
    for (var j = 0; j < OMEGA.length; j++) {
      out[2 * j] = Math.sin(pos * OMEGA[j]) * CFG.POS_AMP;
      out[2 * j + 1] = Math.cos(pos * OMEGA[j]) * CFG.POS_AMP;
    }
    return out;
  }

  /* ------------------------------------------------------------------ */
  /* 4. Блокуудын жин                                                    */
  /* ------------------------------------------------------------------ */
  function buildHeadWeights(headIndex, rand) {
    var D = CFG.D_MODEL, H = CFG.HEAD_DIM, S = CFG.D_SEM;
    var wq, wk, bq = new Float64Array(H), scale;

    if (headIndex === 0) {
      // «Өмнөх токен»: асуулгыг байрлалын векторыг нэг алхам эргүүлж үүсгэнэ.
      wq = zeroMatrix(D, H);
      wk = zeroMatrix(D, H);
      for (var j = 0; j < OMEGA.length; j++) {
        var w = OMEGA[j], c = Math.cos(w), s = Math.sin(w);
        var ds = S + 2 * j, dc = S + 2 * j + 1;
        wq[ds][2 * j] = c;  wq[dc][2 * j] = -s;
        wq[ds][2 * j + 1] = s; wq[dc][2 * j + 1] = c;
        wk[ds][2 * j] = 1;
        wk[dc][2 * j + 1] = 1;
      }
      scale = 9;
    } else if (headIndex === 1) {
      // «Эхний токен»: асуулга нь тогтмол (0-р байрлалын вектор).
      wq = zeroMatrix(D, H);
      wk = zeroMatrix(D, H);
      // (−sin, +cos) хэлбэрийн тогтмол асуулга нь p = 0 үед хамгийн их
      // утга өгнө. Нийлбэр нь тэг тул LayerNorm-ийн дундажийн шилжилт
      // яг таг хасагдаж, толгойн зан төлөв тогтвортой болно.
      for (var j2 = 0; j2 < OMEGA.length; j2++) {
        wk[S + 2 * j2][2 * j2] = 1;
        wk[S + 2 * j2 + 1][2 * j2 + 1] = 1;
        bq[2 * j2] = -1;
        bq[2 * j2 + 1] = 1;
      }
      scale = 7;
    } else if (headIndex === 2) {
      // «Утгын төстэй байдал»: Q ба K нэг ижил проекц ашиглана.
      var shared = zeroMatrix(D, H);
      for (var i = 0; i < S; i++) {
        for (var h = 0; h < H; h++) shared[i][h] = gaussian(rand) / Math.sqrt(S);
      }
      wq = shared;
      wk = shared;
      scale = 3.5;
    } else {
      wq = randMatrix(D, H, 1 / Math.sqrt(D), rand);
      wk = randMatrix(D, H, 1 / Math.sqrt(D), rand);
      scale = 3;
    }

    // W_V нь санамсаргүй проекц. W_O-г үүний шилжүүлсэн хэлбэрээр
    // байгуулснаар "хуулах хэлхээ" (copy circuit) үүсэж, анхаарал нь
    // контекстээс мэдээллийг одоогийн байрлал руу зөөдөг.
    return {
      wq: wq,
      wk: wk,
      bq: bq,
      wv: randMatrix(D, H, 1 / Math.sqrt(H), rand),
      scale: scale,
      info: HEAD_INFO[headIndex]
    };
  }

  /**
   * Гаралтын проекц W_O-г толгой бүрийн W_V-ийн шилжүүлсэн хэлбэрээс
   * угсарна: ингэснээр Σ_h (анхаарлаар жинлэгдсэн утгууд) нь эх векторын
   * ойролцоо сэргээлт болж, анхаарал контекстийг үнэхээр "хуулна".
   */
  function buildOutputProjection(heads) {
    var D = CFG.D_MODEL, H = CFG.HEAD_DIM;
    var wo = zeroMatrix(D, D);
    for (var h = 0; h < heads.length; h++) {
      var wv = heads[h].wv;
      for (var i = 0; i < H; i++) {
        var row = wo[h * H + i];
        for (var d = 0; d < D; d++) row[d] = wv[d][i];
      }
    }
    return wo;
  }

  var BLOCKS = [];
  (function buildBlocks() {
    for (var b = 0; b < CFG.N_BLOCKS; b++) {
      var rand = rng(CFG.SEED + 977 * (b + 1));
      var heads = [];
      for (var h = 0; h < CFG.N_HEADS; h++) heads.push(buildHeadWeights(h, rand));
      BLOCKS.push({
        heads: heads,
        wo: buildOutputProjection(heads),
        w1: randMatrix(CFG.D_MODEL, CFG.D_FF, 1 / Math.sqrt(CFG.D_MODEL), rand),
        w2: randMatrix(CFG.D_FF, CFG.D_MODEL, 1 / Math.sqrt(CFG.D_FF), rand)
      });
    }
  })();

  /* ------------------------------------------------------------------ */
  /* 5. Урагш дамжлага                                                   */
  /* ------------------------------------------------------------------ */
  /**
   * @param {number[]} ids токенуудын дугаар
   * @returns {object} бүх завсрын утгыг агуулсан ул мөр (trace)
   */
  function forward(ids) {
    var T = ids.length, D = CFG.D_MODEL, S = CFG.D_SEM, H = CFG.HEAD_DIM;
    var i, t, h, j;

    // 5.1 Эмбеддинг + байрлалын кодчилол
    var semantic = [], positional = [], x = [];
    for (t = 0; t < T; t++) {
      var sem = E[ids[t]];
      var pos = positionalVector(t);
      var vec = new Float64Array(D);
      for (i = 0; i < S; i++) vec[i] = sem[i];
      for (i = 0; i < CFG.D_POS; i++) vec[S + i] = pos[i];
      semantic.push(sem);
      positional.push(pos);
      x.push(vec);
    }

    var trace = {
      ids: ids,
      T: T,
      semantic: semantic,
      positional: positional,
      x0: x.map(function (v) { return Float64Array.from(v); }),
      blocks: []
    };

    // 5.2 Трансформер блокууд
    for (var bi = 0; bi < CFG.N_BLOCKS; bi++) {
      var block = BLOCKS[bi];
      var ln1 = x.map(layerNorm);
      var headTraces = [];
      var concat = [];
      for (t = 0; t < T; t++) concat.push(new Float64Array(D));

      for (h = 0; h < CFG.N_HEADS; h++) {
        var hw = block.heads[h];
        var Q = [], K = [], Vv = [];
        for (t = 0; t < T; t++) {
          var q = matVec(ln1[t], hw.wq, H);
          for (i = 0; i < H; i++) q[i] += hw.bq[i];
          Q.push(q);
          K.push(matVec(ln1[t], hw.wk, H));
          Vv.push(matVec(ln1[t], hw.wv, H));
        }

        var scores = [], attn = [], outs = [];
        for (t = 0; t < T; t++) {
          var raw = new Float64Array(T).fill(-Infinity);
          for (j = 0; j <= t; j++) {
            raw[j] = (dot(Q[t], K[j]) / Math.sqrt(H)) * hw.scale;
          }
          var w = softmax(raw, 0, t + 1);
          var o = new Float64Array(H);
          for (j = 0; j <= t; j++) addInto(o, Vv[j], w[j]);
          scores.push(raw);
          attn.push(w);
          outs.push(o);
          // толгойн гаралтыг нийлүүлсэн вектор дотор байрлуулна
          for (i = 0; i < H; i++) concat[t][h * H + i] = o[i];
        }

        headTraces.push({ q: Q, k: K, v: Vv, scores: scores, attn: attn, out: outs, info: hw.info });
      }

      var attnProj = [], resid1 = [];
      for (t = 0; t < T; t++) {
        var proj = matVec(concat[t], block.wo, D);
        for (i = 0; i < D; i++) proj[i] *= CFG.ATTN_OUT_SCALE;
        var r1 = Float64Array.from(x[t]);
        addInto(r1, proj);
        attnProj.push(proj);
        resid1.push(r1);
      }

      var ln2 = resid1.map(layerNorm);
      var hidden = [], act = [], mlpOut = [], resid2 = [];
      for (t = 0; t < T; t++) {
        var hpre = matVec(ln2[t], block.w1, CFG.D_FF);
        var hact = new Float64Array(CFG.D_FF);
        for (i = 0; i < CFG.D_FF; i++) hact[i] = gelu(hpre[i]);
        var mo = matVec(hact, block.w2, D);
        for (i = 0; i < D; i++) mo[i] *= CFG.MLP_OUT_SCALE;
        var r2 = Float64Array.from(resid1[t]);
        addInto(r2, mo);
        hidden.push(hpre); act.push(hact); mlpOut.push(mo); resid2.push(r2);
      }

      trace.blocks.push({
        index: bi,
        ln1: ln1,
        heads: headTraces,
        concat: concat,
        attnProj: attnProj,
        resid1: resid1,
        ln2: ln2,
        mlpHidden: hidden,
        mlpAct: act,
        mlpOut: mlpOut,
        resid2: resid2
      });

      x = resid2;
    }

    // 5.3 Эцсийн LayerNorm ба логит
    var lnf = x.map(layerNorm);
    var last = lnf[T - 1];
    var logits = new Float64Array(V);
    for (var b2 = 0; b2 < V; b2++) {
      var s2 = 0;
      var rb = R[b2];
      for (i = 0; i < S; i++) s2 += last[i] * rb[i];
      logits[b2] = s2 * CFG.LOGIT_SCALE + stats.logPrior[b2] * CFG.PRIOR_WEIGHT;
    }

    trace.lnf = lnf;
    trace.logits = logits;
    return trace;
  }

  /* ------------------------------------------------------------------ */
  /* 6. Температур, top-k, top-p бүхий түүвэрлэлт                        */
  /* ------------------------------------------------------------------ */
  function probabilities(logits, temperature, topK, topP) {
    var temp = Math.max(0.05, temperature || 1);
    var scaled = new Float64Array(logits.length);
    for (var i = 0; i < logits.length; i++) scaled[i] = logits[i] / temp;

    var order = [];
    for (i = 0; i < logits.length; i++) order.push(i);
    order.sort(function (a, b) { return scaled[b] - scaled[a]; });

    var allowed = Object.create(null);
    var limit = topK && topK > 0 ? Math.min(topK, order.length) : order.length;
    var probsFull = softmax(scaled);
    var cumulative = 0, kept = 0;
    for (i = 0; i < limit; i++) {
      var id = order[i];
      allowed[id] = true;
      kept++;
      cumulative += probsFull[id];
      if (topP && topP < 1 && cumulative >= topP) break;
    }

    var sum = 0;
    var out = new Float64Array(logits.length);
    for (i = 0; i < logits.length; i++) {
      if (allowed[i]) { out[i] = probsFull[i]; sum += out[i]; }
    }
    for (i = 0; i < logits.length; i++) out[i] = sum > 0 ? out[i] / sum : 0;
    return { probs: out, order: order, kept: kept, full: probsFull };
  }

  function sample(probs, rand) {
    var r = (rand || Math.random)();
    var acc = 0;
    for (var i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r <= acc) return i;
    }
    return probs.length - 1;
  }

  function topTokens(probs, n) {
    var idx = [];
    for (var i = 0; i < probs.length; i++) if (probs[i] > 0) idx.push(i);
    idx.sort(function (a, b) { return probs[b] - probs[a]; });
    return idx.slice(0, n).map(function (i2) {
      return { id: i2, token: corpus.vocab[i2], prob: probs[i2] };
    });
  }

  TE.model = {
    CFG: CFG,
    HEAD_INFO: HEAD_INFO,
    V: V,
    E: E,
    R: R,
    OMEGA: OMEGA,
    stats: stats,
    forward: forward,
    probabilities: probabilities,
    sample: sample,
    topTokens: topTokens,
    layerNorm: layerNorm,
    softmax: softmax,
    gelu: gelu,
    positionalVector: positionalVector,
    rng: rng
  };
})(typeof window !== 'undefined' ? window : globalThis);
