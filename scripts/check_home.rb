# frozen_string_literal: true

require "jekyll"

# Render the actual homepage Liquid. Peripheral theme includes are irrelevant
# to ordering and image attributes, so this regression harness omits them.
class EmptyInclude < Liquid::Tag
  def render(_context)
    ""
  end
end
Liquid::Template.register_tag("include", EmptyInclude)
Liquid::Template.register_filter(Jekyll::Filters)
source = File.expand_path("..", __dir__)
site = Jekyll::Site.new(Jekyll.configuration("source" => source, "quiet" => true))
template = File.read(File.join(source, "_layouts/home.html")).sub(/\A---.*?---/m, "")

def render(template, site, posts, page, per_page)
  Liquid::Template.parse(template).render!(
    { "site" => { "posts" => posts, "data" => {} },
      "paginator" => { "page" => page, "per_page" => per_page, "total_pages" => 3 } },
    registers: { site: site }
  )
end

def assert(condition, message)
  raise message unless condition
end

[0, 2, 10, 12].each do |pinned_count|
  posts = (1..23).map do |n|
    { "title" => "Post #{n}", "url" => "/posts/p#{n}/", "pin" => n <= pinned_count }
  end
  hidden = { "title" => "Hidden", "url" => "/posts/hidden/", "pin" => true, "hidden" => true }
  results = (1..3).map do |page|
    html = render(template, site, [hidden] + posts, page, 10)
    html.scan(/href="(\/posts\/[^\"]+)"/).flatten
  end
  assert(results.map(&:size) == [10, 10, 3], "Page counts differ for #{pinned_count} pinned posts")
  assert(results.flatten == posts.map { |p| p["url"] }, "Posts duplicated, omitted or hidden post leaked")
end
posts = [
  { "title" => "Image 1", "url" => "/1/", "media_subpath" => "/assets/articles",
    "image" => { "path" => "a.png", "lqip" => "https://example.com/preview.png" } },
  { "title" => "Image 2", "url" => "/2/", "media_subpath" => "/assets/articles",
    "image" => { "path" => "https://example.com/full.png" } },
  { "title" => "Image 3", "url" => "/3/", "media_subpath" => "/assets/articles",
    "image" => { "path" => "b.png", "lqip" => "small.png" } }
]
images = render(template, site, posts, 1, 10).scan(/<img[^>]+>/)
assert(images[0].include?('src="/assets/articles/a.png"'), "Relative image prefix missing")
assert(images[0].include?('lqip="https://example.com/preview.png"'), "Remote LQIP corrupted")
assert(!images[1].include?("lqip="), "Previous LQIP leaked into next card")
assert(images[1].include?('src="https://example.com/full.png"'), "Remote image corrupted")
assert(images[2].include?('lqip="/assets/articles/small.png"'), "Relative LQIP prefix missing")
puts "PASS: pinned pagination, hidden posts, image URLs and LQIP isolation"
