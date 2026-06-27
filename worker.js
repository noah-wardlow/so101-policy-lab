// Serves the static app from Workers Assets and the large ACT ONNX from R2.
// Asset requests are handled by the assets server directly (COOP/COEP applied
// via dist/_headers); only the ONNX path falls through to this handler.
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === '/models/act/act.onnx') {
      const obj = await env.MODELS.get('act/act.onnx');
      if (!obj) return new Response('model not found', { status: 404 });
      const headers = new Headers();
      obj.writeHttpMetadata(headers);
      headers.set('content-type', 'application/octet-stream');
      headers.set('etag', obj.httpEtag);
      // Same-origin fetch, but be explicit so COEP:require-corp is satisfied.
      headers.set('cross-origin-resource-policy', 'same-origin');
      headers.set('cache-control', 'public, max-age=31536000, immutable');
      return new Response(obj.body, { headers });
    }
    return env.ASSETS.fetch(request);
  },
};
