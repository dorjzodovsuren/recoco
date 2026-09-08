/*
 * Хөтөч дээрх интеграцийн сорил (Playwright + Chromium).
 * Ажиллуулах:  NODE_PATH=/opt/node22/lib/node_modules node tests/browser.test.js
 */
'use strict';
var path = require('path');
var chromium = require('playwright').chromium;

var passed = 0, failed = 0;
function check(name, cond, extra) {
  if (cond) { passed++; console.log('  ✓ ' + name); }
  else { failed++; console.error('  ✗ ' + name + (extra ? ' → ' + extra : '')); }
}

(async function () {
  var browser = await chromium.launch();
  var page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });

  var errors = [];
  page.on('pageerror', function (e) { errors.push(String(e)); });
  page.on('console', function (m) { if (m.type() === 'error') errors.push(m.text()); });

  var file = 'file://' + path.resolve(__dirname, '..', 'index.html');
  await page.goto(file, { waitUntil: 'load' });
  await page.waitForTimeout(400);

  console.log('\n1. Хуудас ачаалагдах');
  check('JS алдаагүй', errors.length === 0, errors.join(' | '));
  check('гарчиг монголоор', (await page.title()).indexOf('Трансформер') === 0);
  check('токен харагдана', (await page.locator('.token').count()) > 3);
  check('блокын таб 4', (await page.locator('#block-tabs .tab').count()) === 4);
  check('толгойн таб 4', (await page.locator('#head-tabs .tab').count()) === 4);
  check('анхаарлын матриц зурагдсан', (await page.locator('#attn-matrix rect.attn-cell').count()) > 4);
  check('магадлалын мөрүүд', (await page.locator('.prob-row').count()) === 12);
  check('жишээ товчнууд', (await page.locator('.example-btn').count()) === 6);

  console.log('\n2. Canvas зурагдсан эсэх');
  var painted = await page.evaluate(function () {
    var out = {};
    ['embed-canvas', 'q-canvas', 'k-canvas', 'v-canvas', 'mlp-canvas', 'resid-canvas']
      .forEach(function (id) {
        var c = document.getElementById(id);
        var ctx = c.getContext('2d');
        var d = ctx.getImageData(0, 0, c.width, c.height).data;
        var distinct = new Set();
        for (var i = 0; i < d.length; i += 4 * 97) distinct.add(d[i] + ',' + d[i + 1] + ',' + d[i + 2]);
        out[id] = { w: c.width, h: c.height, colors: distinct.size };
      });
    return out;
  });
  Object.keys(painted).forEach(function (id) {
    check(id + ' олон өнгөтэй зурагдсан', painted[id].colors > 3,
      JSON.stringify(painted[id]));
  });

  console.log('\n3. Диаграммын баганууд');
  var barWidths = await page.evaluate(function () {
    function widthOf(sel) {
      var e = document.querySelector(sel);
      return e ? e.getBoundingClientRect().width : -1;
    }
    return { prob: widthOf('.prob-row .fill'), attn: widthOf('.attn-bar-row .fill') };
  });
  check('магадлалын багана харагдана', barWidths.prob > 1, 'w=' + barWidths.prob);
  check('анхаарлын багана харагдана', barWidths.attn > 1, 'w=' + barWidths.attn);

  console.log('\n4. Харилцан үйлдэл');
  var firstTop = await page.locator('.prob-row .tok').first().textContent();
  await page.locator('.example-btn', { hasText: 'Багш хичээл' }).click();
  await page.waitForTimeout(200);
  check('жишээ дарахад оролт солигдоно',
    (await page.locator('#prompt').inputValue()) === 'Багш хичээл');
  var newTop = await page.locator('.prob-row .tok').first().textContent();
  check('таамаглал шинэчлэгдсэн', newTop !== firstTop, firstTop + ' → ' + newTop);
  check('"заадаг" тэргүүлнэ', newTop.trim() === 'заадаг', newTop);

  await page.locator('#head-tabs .tab').nth(1).click();
  await page.waitForTimeout(150);
  check('толгой солигдоно',
    (await page.locator('#head-tabs .tab').nth(1).getAttribute('aria-selected')) === 'true');
  check('толгойн тайлбар шинэчлэгдэнэ',
    (await page.locator('#head-desc').textContent()).indexOf('Эхний токен') === 0);

  await page.locator('#block-tabs .tab').nth(2).click();
  await page.waitForTimeout(150);
  check('блок солигдоно',
    (await page.locator('#block-tabs .tab').nth(2).getAttribute('aria-selected')) === 'true');

  await page.locator('.token').nth(1).click();
  await page.waitForTimeout(150);
  check('токен сонгогдоно', (await page.locator('.token.selected').count()) === 1);

  await page.locator('#temperature').fill('0.2');
  await page.dispatchEvent('#temperature', 'input');
  await page.waitForTimeout(200);
  var lowTemp = parseFloat((await page.locator('.prob-row .val').first().textContent()).replace('%', ''));
  await page.locator('#temperature').fill('1.9');
  await page.dispatchEvent('#temperature', 'input');
  await page.waitForTimeout(200);
  var highTemp = parseFloat((await page.locator('.prob-row .val').first().textContent()).replace('%', ''));
  check('бага температур → өндөр магадлал', lowTemp > highTemp, lowTemp + '% vs ' + highTemp + '%');

  await page.locator('#topk').fill('5');
  await page.dispatchEvent('#topk', 'input');
  await page.waitForTimeout(200);
  check('top-k = 5 → 5 мөр', (await page.locator('.prob-row').count()) === 5,
    String(await page.locator('.prob-row').count()));

  var before = await page.locator('#prompt').inputValue();
  await page.locator('#generate').click();
  await page.waitForTimeout(250);
  var after = await page.locator('#prompt').inputValue();
  check('токен нэмэгдэнэ', after.length > before.length, before + ' → ' + after);

  await page.locator('#reset').click();
  await page.waitForTimeout(250);
  check('шинэчлэх товч ажиллана',
    (await page.locator('#prompt').inputValue()) === 'Монгол улсын нийслэл хот бол');

  console.log('\n5. Толь бичигт байхгүй үг');
  await page.locator('#prompt').fill('Пингвин нисдэггүй');
  await page.dispatchEvent('#prompt', 'input');
  await page.waitForTimeout(250);
  check('үл мэдэгдэх токен тэмдэглэгдэнэ', (await page.locator('.token.unknown').count()) > 0);

  console.log('\n6. Гар утасны хэмжээ');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(400);
  var overflow = await page.evaluate(function () {
    return document.documentElement.scrollWidth - document.documentElement.clientWidth;
  });
  check('хэвтээ гүйлгэлт үүсэхгүй', overflow <= 1, 'overflow=' + overflow + 'px');

  await page.setViewportSize({ width: 1400, height: 1000 });
  await page.locator('#reset').click();
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.resolve(__dirname, '..', 'preview.png'), fullPage: false });

  check('ажиллах явцад JS алдаа гараагүй', errors.length === 0, errors.join(' | '));

  await browser.close();
  console.log('\n─────────────────────────────');
  console.log('Амжилттай: ' + passed + ',  Алдаа: ' + failed);
  process.exit(failed === 0 ? 0 : 1);
})();
