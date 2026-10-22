/* Site navigation controls and independent desktop layout preferences. */
(() => {
  'use strict';

  for (const [selector, label] of [
    ['.toc-trigger:not(#toc-solo-trigger)', '文章目录'],
    ['#back-to-top', '返回顶部']
  ]) {
    for (const button of document.querySelectorAll(selector)) {
      if (!button.hasAttribute('aria-label')) button.setAttribute('aria-label', label);
      button.querySelector('i')?.setAttribute('aria-hidden', 'true');
    }
  }

  const root = document.documentElement;
  const controls = [
    { id: 'nav-toggle', attribute: 'data-nav-hidden', key: 'mason.layout.nav.hidden',
      label: '左侧导航', collapseIcon: 'left', expandIcon: 'right' },
    { id: 'panel-toggle', attribute: 'data-panel-hidden', key: 'mason.layout.panel.hidden',
      label: '右侧栏', collapseIcon: 'right', expandIcon: 'left' }
  ];

  for (const control of controls) {
    const button = document.getElementById(control.id);
    if (!button || !document.getElementById(button.getAttribute('aria-controls'))) continue;

    const render = (collapsed) => {
      root.toggleAttribute(control.attribute, collapsed);
      button.setAttribute('aria-expanded', String(!collapsed));
      const action = `${collapsed ? '展开' : '收起'}${control.label}`;
      button.setAttribute('aria-label', action);
      button.title = action;
      const direction = collapsed ? control.expandIcon : control.collapseIcon;
      button.querySelector('i').className = `fa-solid fa-angles-${direction} fa-fw`;
    };

    let collapsed = false;
    try {
      collapsed = localStorage.getItem(control.key) === 'true';
    } catch {
      // Layout controls also work when the browser blocks local storage.
    }
    render(collapsed);
    button.hidden = false;

    button.addEventListener('click', () => {
      const next = !root.hasAttribute(control.attribute);
      render(next);
      try {
        localStorage.setItem(control.key, String(next));
      } catch {
        // Keep the current page usable even when preferences cannot be saved.
      }
    });
  }

  const desktop = window.matchMedia('(min-width: 850px)');
  desktop.addEventListener('change', ({ matches }) => {
    if (matches && document.body.hasAttribute('sidebar-display')) {
      // Use Chirpy's trigger so its internal drawer state and mask stay in sync.
      document.getElementById('sidebar-trigger').click();
    }
  });

  const toc = window.tocbot;
  const heading = document.querySelector('.content h2');
  if (toc && heading) {
    const wideToc = window.matchMedia('(min-width: 1200px)');
    const alignToc = () => {
      const container = document.querySelector(
        wideToc.matches ? '#toc' : '#toc-popup-content'
      );
      if (!container) return;
      const handlers = new Map(
        [...container.querySelectorAll('.toc-link')]
          .filter((link) => typeof link.onclick === 'function')
          .map((link) => [link.getAttribute('href'), link.onclick])
      );
      const margin = parseFloat(getComputedStyle(heading).scrollMarginTop) || 0;
      toc.refresh({
        tocSelector: wideToc.matches ? '#toc' : '#toc-popup-content',
        contentSelector: '.content',
        ignoreSelector: '[data-toc-skip]',
        headingSelector: 'h2, h3, h4',
        orderedList: false,
        scrollSmooth: false,
        collapseDepth: wideToc.matches ? 0 : 4,
        headingsOffset: margin + 1
      });
      // Refresh rebuilds links; retain Chirpy's popup-close callbacks.
      for (const link of container.querySelectorAll('.toc-link')) {
        const handler = handlers.get(link.getAttribute('href'));
        if (handler) link.onclick = handler;
      }
      document.dispatchEvent(new Event('scroll', { bubbles: true }));
    };
    // Chirpy rebuilds the TOC on breakpoint changes; align it afterward.
    wideToc.addEventListener('change', () => requestAnimationFrame(alignToc));
    alignToc();
  }
})();
