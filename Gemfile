source "https://rubygems.org"

gem "jekyll-theme-chirpy", "~> 7.6.0"

group :jekyll_plugins do
  gem "jekyll-paginate"
  gem "jekyll-redirect-from"
  gem "jekyll-seo-tag"
  gem "jekyll-sitemap"
  gem "jekyll-feed"
end

# Windows and JRuby does not include zoneinfo files
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# Newer http_parser.rb releases do not provide the Java implementation used by JRuby.
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]

# Jekyll 4 requires webrick
gem "webrick", "~> 1.8"
