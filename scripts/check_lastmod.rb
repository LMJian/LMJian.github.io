# frozen_string_literal: true

require "jekyll"
require "tmpdir"
require "open3"
load File.expand_path("../_plugins/posts-lastmod-hook.rb", __dir__)

Dir.mktmpdir("blog-lastmod-") do |root|
  def git(root, *args)
    output, status = Open3.capture2e("git", *args, chdir: root)
    raise output unless status.success?
    output.strip
  end
  git(root, "init", "-q")
  git(root, "config", "user.name", "Local Test")
  git(root, "config", "user.email", "test@example.invalid")
  path = File.join(root, "post.md")
  File.write(path, "first\n")
  git(root, "add", "post.md")
  git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "first")
  post = Struct.new(:site, :path, :data).new(Struct.new(:source).new(root), path, {})
  Jekyll::Hooks.trigger(:posts, :post_init, post)
  raise "First commit should not show an update date" if post.data["last_modified_at"]
  File.write(path, "second\n")
  git(root, "add", "post.md")
  git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "second")
  expected = git(root, "log", "-1", "--format=%ad", "--date=iso", "--", path)
  Dir.chdir(File.dirname(root)) { Jekyll::Hooks.trigger(:posts, :post_init, post) }
  raise "Hook used caller's working directory" unless post.data["last_modified_at"] == expected
  post.data["last_modified_at"] = "2026-10-01 08:00:00 +0800"
  Jekyll::Hooks.trigger(:posts, :post_init, post)
  raise "Explicit date overwritten" unless post.data["last_modified_at"] == "2026-10-01 08:00:00 +0800"
end
puts "PASS: first commit, repository-scoped Git history and explicit update date"
