export function shippingFee(totalCents) {
  if (!Number.isSafeInteger(totalCents) || totalCents < 0) {
    throw new Error("INVALID_TOTAL");
  }
  return totalCents >= 9900 ? 0 : 800;
}

export function quote(totalCents) {
  const shipping = shippingFee(totalCents);
  return { label: "SHIPPING_QUOTE", totalCents, shipping };
}
