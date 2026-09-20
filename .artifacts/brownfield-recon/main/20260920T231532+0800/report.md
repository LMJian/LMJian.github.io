# Recon: Jekyll/Chirpy existing logic audit and repair

- Repository: `LMJian/LMJian.github.io` (`origin` is the public GitHub Pages repository).
- Task: audit the existing repository logic, identify concrete defects, repair them, and verify the result. Target source is the user's 2026-09-20 request plus the Chirpy 7.6 configuration/layout contracts.
- Scope: `Gemfile`, `_config.yml`, `index.html`, `_layouts/home.html`, `_includes/favicons.html`, `_tabs/`, `_posts/`, the custom SCSS entrypoint, and the GitHub Pages workflow. No remote deployment state was available for inspection.
- Branch: `main`.
- Baseline: `1635c41a2cd36fc952f007f9b37c96ec7c6e37a2`; clean working tree before this report was created.
- Captured at: `2026-09-20T23:15:32+0800`.
- Compatibility: preserve the public GitHub Pages paths and the current Chirpy 7.x theme integration; do not change authored post/page content or the custom visual design.
- Next phase: implementation — **READY_WITH_GAPS**. Local code and upstream contracts are sufficient for the scoped repairs. A production Jekyll build still requires Ruby 3.1+ and installed gems; this machine currently has Ruby 2.6.10 with no project gems.

## Baseline behavior and change disposition

| Current behavior and evidence | Required target | Retain / replace / remove | Affected files | Validation scenario |
| --- | --- | --- | --- | --- |
| The post declares category `随笔` and three tags (`1635c41:_posts/2026-09-15-hello-world.md:4-5`), but `_config.yml` has no `jekyll-archives` generation settings (`1635c41:_config.yml:131-188`). Chirpy's category/tag links target generated archive pages. | Generate category and tag archive pages with Chirpy's `category` and `tag` layouts at `/categories/:name/` and `/tags/:name/`. | Add the missing archive contract; retain the existing category/tag tabs and authored metadata. | `_config.yml` | A production build contains the category and tag detail pages linked from the post and taxonomy tabs. |
| The `tabs` collection is emitted, but its defaults omit Chirpy's root-level `permalink: /:title/` contract (`1635c41:_config.yml:144-167`). | Keep stable public routes `/categories/`, `/tags/`, `/archives/`, and `/about/`. | Add the permalink default; retain tab order, icons, layouts, and content. | `_config.yml`, `_tabs/*` | Generated tab URLs are rooted at the site root, not under `/tabs/` or with collection-specific paths. |
| The custom home layout computes `pinned_num = pinned.size - offset` without capping it to `paginator.per_page` (`1635c41:_layouts/home.html:46-72`). More than one page of pinned posts can therefore overfill a page and displace normal-post indexing. | Match Chirpy 7.6's bounded visible-window algorithm while preserving the custom hero and post-card markup. | Replace only the pagination selection block. | `_layouts/home.html` | With more pinned posts than one page, each page contains at most `per_page` entries and normal posts resume at the correct index. |
| The image block prepends `/` whenever a source lacks `//`, even when no `media_subpath` exists, and emits a raw LQIP value (`1635c41:_layouts/home.html:79-91`). This diverges from the theme's media contract and mishandles relative/data sources. | Apply `media_subpath` only when configured, recognize absolute URLs using `://`, and normalize LQIP through the same media path. | Replace the stale image-path block; retain card structure and alt fallback. | `_layouts/home.html` | Absolute URLs remain unchanged; media-relative image and LQIP paths are prefixed exactly once; a post without `media_subpath` keeps its source unchanged. |
| The dependency range is `jekyll-theme-chirpy ~> 7.0`, `Gemfile.lock` is ignored, and the workflow uses `bundler-cache` (`1635c41:Gemfile:3`, `1635c41:.gitignore:7-10`, `1635c41:.github/workflows/pages-deploy.yml:27-38`). The latest matching release observed during recon is 7.6.0, while this repository overrides version-sensitive theme files. | Keep automatic patch updates within the audited 7.6 line, but require deliberate review before a later minor/major theme upgrade. Do not claim fully reproducible transitive dependencies without a lockfile. | Replace the broad range with `~> 7.6.0`; retain the lockfile gap rather than hand-authoring one. | `Gemfile`, `.gitignore`, `.github/workflows/pages-deploy.yml` | A Ruby 3.1+ dependency resolution selects Chirpy 7.6.x; a future 7.7+ release is not selected implicitly. |
| `http_parser.rb ~> 0.6.0` is forced on every runtime (`1635c41:Gemfile:22-23`), although this legacy constraint exists for JRuby's Java implementation. The deployment workflow uses MRI Ruby 3.2. | Let MRI resolve a current compatible native gem while retaining the JRuby compatibility pin. | Restrict the 0.6.x dependency to the `jruby` platform. | `Gemfile` | Bundler on MRI Ruby 3.2 does not select `http_parser.rb` 0.6.x because of this direct dependency; JRuby still does. |
| The config advertises updated timestamps (`1635c41:_config.yml:91-92`), but the repository has no `_plugins/posts-lastmod-hook.rb`; Chirpy's gem packaging does not supply repository plugins. The existing post has two Git revisions. | Populate `last_modified_at` from the post's latest Git commit when it has changed after creation. | Add the missing hook, using argument-array process execution instead of interpolating paths into a shell command. | `_plugins/posts-lastmod-hook.rb` | Initializing the existing post yields `2026-09-15 21:56:35 +0800`; an untracked or single-revision post is left unchanged. |

## Relevant flow

`index.html` selects the repository's `_layouts/home.html`. That override composes the custom hero with posts selected from `site.posts` and Jekyll's `paginator`, then relies on the Chirpy gem for default layout, includes, taxonomy layouts, assets, and navigation. `_config.yml` owns the collection URLs and archive generation; the GitHub Actions workflow installs the matching gems under Ruby 3.2, builds `_site`, uploads it, and deploys it.

## Remaining gap

The initial `bundle exec jekyll build --trace` probe stopped before compilation because no Jekyll executable is installed locally. The installed Ruby is 2.6.10, while Chirpy 7.6.0 requires Ruby `~> 3.1`. This does not block the localized config/template repair, but a real production build remains required before integration can be marked ready. No local mock is treated as equivalent to that build.

## Implementation and validation — 2026-09-20

Implemented the scoped repairs in `Gemfile`, `_config.yml`, `_layouts/home.html`, and `_plugins/posts-lastmod-hook.rb`: a 7.6.x theme compatibility boundary, a JRuby-only legacy parser constraint, archive plugin/config activation, root-level tab permalinks, bounded pinned/normal pagination, Chirpy-compatible image/LQIP path handling, and Git-derived post update timestamps. The custom homepage hero, post-card presentation, site content, and styles were not changed.

The investigation expanded once to `_plugins/` after the official starter contract identified it as repository-owned and the advertised update-time behavior could not be implemented by the theme gem alone.

Validation actually run:

- Parsed `_config.yml` with Ruby YAML and asserted the tab permalink, archive types, archive permalinks, and plugin activation: passed.
- Parsed all Liquid control tags in the overridden home layout as a stack and verified balanced `if`/`unless`/`for` blocks: passed.
- Exercised the new page-selection arithmetic with 23 pinned and 17 normal posts at 10 posts per page; observed page splits `[10+0, 10+0, 3+7, 0+10]` with no page overflow: passed.
- Ran `ruby -c Gemfile` and asserted the Chirpy 7.6.x and JRuby-only parser constraints: passed.
- Loaded the post-update hook against the existing two-revision post and asserted the Git-derived timestamp: passed.
- Ran `git diff --check`: passed.
- Re-ran no production Jekyll build because the required Ruby/Jekyll runtime remains unavailable locally.

Implementation readiness is **READY** for the scoped repair. Integration readiness is **READY_WITH_GAPS** pending one Ruby 3.1+ production build and generated-link verification; release readiness was not assessed.
