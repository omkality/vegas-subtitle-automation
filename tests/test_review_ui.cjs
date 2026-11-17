const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');

function fixture() {
  const elements = new Map();
  const requests = [];
  const context = vm.createContext({
    URLSearchParams, setTimeout, clearTimeout,
    document: {
      querySelector(selector) {
        if (!elements.has(selector)) elements.set(selector, { classList: { add() {}, remove() {} } });
        return elements.get(selector);
      },
      querySelectorAll() { return []; },
    },
    window: { location: { hash: '' }, scrollTo() {} },
    fetch(url, options) {
      if (url === '/api/data') return new Promise(() => {});
      return new Promise(resolve => requests.push({ body: JSON.parse(options.body), resolve }));
    },
  });
  vm.runInContext(fs.readFileSync('web/review.js', 'utf8'), context);
  vm.runInContext("data={dataset_sha256:'sample',items:Array(40).fill({id:'one'})};states.set('one',{id:'one',text:'first'})", context);
  return { context, requests };
}
const tick = () => new Promise(resolve => setImmediate(resolve));
const ok = { ok: true, json: async () => ({ saved: 1 }) };

test('overlapping saves preserve newer edits and send snapshots in order', async () => {
  const { context, requests } = fixture();
  vm.runInContext('changed()', context);
  const first = vm.runInContext('save()', context);
  await tick();
  assert.equal(requests.length, 1);
  vm.runInContext("states.get('one').text='second';changed()", context);
  const second = vm.runInContext('save()', context);
  await tick();
  assert.equal(requests.length, 1);
  assert.equal(requests[0].body.decisions[0].text, 'first');
  requests[0].resolve(ok);
  await first;
  assert.equal(vm.runInContext('dirty', context), true);
  await tick();
  assert.equal(requests.length, 2);
  assert.equal(requests[1].body.decisions[0].text, 'second');
  requests[1].resolve(ok);
  await second;
  assert.equal(vm.runInContext('dirty', context), false);
});

test('failed save keeps edits dirty and prevents page navigation', async () => {
  const { context, requests } = fixture();
  vm.runInContext('changed()', context);
  const turn = vm.runInContext('goTo(1)', context);
  await tick();
  requests[0].resolve({ ok: false, json: async () => ({ error: 'Disk unavailable' }) });
  await turn;
  assert.equal(vm.runInContext('page', context), 0);
  assert.equal(vm.runInContext('dirty', context), true);
});
