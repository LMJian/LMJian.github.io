import assert from 'node:assert/strict';
import { quote } from './owned-app/app.js';

assert.equal(quote(9899).shipping, 800);
assert.equal(quote(9900).shipping, 0);
for (const invalid of [-1, 0.5, Number.MAX_SAFE_INTEGER + 1]) {
  assert.throws(() => quote(invalid), /INVALID_TOTAL/);
}

console.log('通过：运费阈值、负数、小数和安全整数边界检查');
